"""Tests for the F5 validity matrix at architecture A4 / requirements N1.

One of N1's required tests. The parametrization derives from the
Cartesian product of the four field-domain constants in
:mod:`seq_sklearn.config._domains` minus the legal cells; every illegal
cell raises with a message naming the offending fields. v1.1 task-type
rows carry ``pytest.mark.xfail(strict=True)`` so a future change that
accidentally accepts a v1.1 task type fails CI rather than silently
converting to a pass.

The legal cells are imported from :mod:`seq_sklearn.config._validity` so
the test stays in sync with the production validator (a duplicate copy
would silently drift on future matrix changes).
"""

from typing import get_args

import pytest
from pydantic import BaseModel, ValidationError

from seq_sklearn.config._domains import (
    CALIBRATION_STRATEGIES,
    IMBALANCE_STRATEGIES,
    LOSS_STRATEGIES,
    TASK_TYPES,
    V1_1_TASK_TYPES,
)
from seq_sklearn.config._validity import (
    _LEGAL_CELLS,
    check_combo,
    legal_task_loss_pairs,
)
from seq_sklearn.config.base import BaseModelConfig
from seq_sklearn.config.loss import LossConfig
from seq_sklearn.config.sampler import SamplerConfig


def _enumerate_legal_v1_cells() -> list[tuple[str, str, str, str]]:
    cells: list[tuple[str, str, str, str]] = []
    for (task, loss), (imbs, cals) in _LEGAL_CELLS.items():
        if task in V1_1_TASK_TYPES:
            continue
        for imb in imbs:
            for cal in cals:
                cells.append((task, loss, imb, cal))
    return cells


def _enumerate_v1_illegal_cells() -> list[tuple[str, str, str, str]]:
    legal = set(_enumerate_legal_v1_cells())
    v1_tasks = [t for t in TASK_TYPES if t not in V1_1_TASK_TYPES]
    illegal: list[tuple[str, str, str, str]] = []
    for task in v1_tasks:
        for loss in LOSS_STRATEGIES:
            for imb in IMBALANCE_STRATEGIES:
                for cal in CALIBRATION_STRATEGIES:
                    cell = (task, loss, imb, cal)
                    if cell not in legal:
                        illegal.append(cell)
    return illegal


def _enumerate_v1_1_cells() -> list[tuple[str, str, str, str]]:
    cells: list[tuple[str, str, str, str]] = []
    for task in V1_1_TASK_TYPES:
        for loss in LOSS_STRATEGIES:
            for imb in IMBALANCE_STRATEGIES:
                for cal in CALIBRATION_STRATEGIES:
                    cells.append((task, loss, imb, cal))
    return cells


# ---- check_combo (pure function): v1 legal cells pass


@pytest.mark.parametrize(("task", "loss", "imb", "cal"), _enumerate_legal_v1_cells())
def test_check_combo_legal_v1_cells_pass(task: str, loss: str, imb: str, cal: str) -> None:
    check_combo(task, loss, imb, cal)


# ---- check_combo (pure function): v1 illegal cells raise


@pytest.mark.parametrize(("task", "loss", "imb", "cal"), _enumerate_v1_illegal_cells())
def test_check_combo_illegal_v1_cells_raise(task: str, loss: str, imb: str, cal: str) -> None:
    with pytest.raises(ValueError, match=r"is not legal|scheduled for v1") as excinfo:
        check_combo(task, loss, imb, cal)
    msg = str(excinfo.value)
    # The error must name at least one of the offending values so the
    # caller can find the issue without re-reading the matrix.
    assert any(token in msg for token in (task, loss, imb, cal))


# ---- BaseModelConfig (user-facing): same illegal cells raise ValidationError
# Pydantic catches the underlying ValueError from check_combo and re-raises
# as ValidationError; this verifies the user-visible contract at config
# construction.


@pytest.mark.parametrize(("task", "loss", "imb", "cal"), _enumerate_v1_illegal_cells())
def test_base_model_config_illegal_v1_cells_raise_validation_error(
    task: str, loss: str, imb: str, cal: str
) -> None:
    with pytest.raises(ValidationError):
        BaseModelConfig(
            task_type=task,  # type: ignore[arg-type]
            loss=LossConfig(strategy=loss),  # type: ignore[arg-type]
            sampler=SamplerConfig(strategy=imb),  # type: ignore[arg-type]
            calibration_strategy=cal,  # type: ignore[arg-type]
        )


# ---- v1.1 task types: rejected with a "scheduled for v1.1" message


@pytest.mark.parametrize(("task", "loss", "imb", "cal"), _enumerate_v1_1_cells())
@pytest.mark.xfail(
    strict=True,
    reason="v1.1 task types must be rejected in v1; xfail-strict guards "
    "against accidental v1.1 acceptance until v1.1 ships.",
)
def test_v1_1_task_types_currently_rejected(task: str, loss: str, imb: str, cal: str) -> None:
    # When v1.1 enables a cell, check_combo stops raising for it; the
    # xfail flips to XPASS and CI fails. That is the intent.
    check_combo(task, loss, imb, cal)


def test_v1_1_message_names_v1_1_explicitly() -> None:
    """The rejection message must name v1.1 so callers know where it lands."""
    with pytest.raises(ValueError, match=r"v1\.1"):
        check_combo("multilabel", "cross_entropy", "none", "none")


def test_unknown_task_loss_pair_raises_helpful_message() -> None:
    """Legal task with illegal loss surfaces the legal-loss list."""
    with pytest.raises(ValueError, match="Legal losses"):
        check_combo("regression_point", "cross_entropy", "none", "none")


# ---- Literal-to-domain invariant
#
# Only LossConfig.strategy and SamplerConfig.strategy feed check_combo,
# so the F5 failure-mode contract depends on their Literals matching the
# _domains tuples exactly. If a Literal drifts, an illegal-cell value
# would fail at sub-config construction (pydantic Literal error) instead
# of at BaseModelConfig._check_validity_matrix (F5 ValueError ->
# ValidationError), silently changing the documented failure mode.
# OptimizerConfig.name / SchedulerConfig.name do not feed check_combo
# and have no _domains tuple to pair against, so they are out of scope.


@pytest.mark.parametrize(
    ("config_cls", "field", "domain"),
    [
        (LossConfig, "strategy", LOSS_STRATEGIES),
        (SamplerConfig, "strategy", IMBALANCE_STRATEGIES),
    ],
)
def test_family_config_strategy_literals_match_domains(
    config_cls: type[BaseModel], field: str, domain: tuple[str, ...]
) -> None:
    annotation = config_cls.model_fields[field].annotation
    assert get_args(annotation) == tuple(domain)


def test_legal_task_loss_pairs_excludes_v1_1_by_default() -> None:
    # The accessor's reason for existing is F5/Optuna closure: it must
    # not leak v1.1 pairs by default. A mutation flipping the filter to
    # admit v1.1 must fail here (the only other guard is a separate
    # one in build_loss, which does not cover this function's contract).
    pairs = legal_task_loss_pairs()
    assert all(task not in V1_1_TASK_TYPES for task, _ in pairs)
    assert ("multilabel", "cross_entropy") not in pairs
    assert ("regression_multioutput", "mse") not in pairs
    # Every default pair traces back to a non-v1.1 _LEGAL_CELLS entry.
    assert pairs == frozenset((t, lo) for (t, lo) in _LEGAL_CELLS if t not in V1_1_TASK_TYPES)
    assert ("binary", "cross_entropy") in pairs
    assert ("regression_quantile", "pinball") in pairs


def test_legal_task_loss_pairs_include_v1_1_adds_v1_1_pairs() -> None:
    default = legal_task_loss_pairs()
    full = legal_task_loss_pairs(include_v1_1=True)
    assert default < full  # strict superset
    assert full == frozenset(_LEGAL_CELLS)
    assert ("multilabel", "cross_entropy") in full
    assert any(task in V1_1_TASK_TYPES for task, _ in full)
