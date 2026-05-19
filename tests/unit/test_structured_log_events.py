"""F11 structured-log conformance + emission contract (N1).

The consolidating Phase 9 pass (per implementation_plan): one
parametrized assertion per F11-table event that :func:`emit` produces
exactly one ``caplog`` record with the right ``record.event``,
``record.payload`` key schema, and level. Per-code-path emission is
already covered by the phase-specific tests (NaN-loss variants, the
Optuna pruning e2e, the encoder unseen-category test, etc.); this test
owns the systematic table-vs-enum conformance check.

``train.hidden_norm`` is xfail-strict: it is a recurrent-only event
with no v1 emission path (the transformer family never produces it),
so the gap stays explicit in CI output rather than silently skipped.
"""

import logging
from pathlib import Path

import pytest

from seq_sklearn.logging import Event, emit

# Event members with at least one emission call-site in v1 source
# (logging.py holds only the enum definition, so it is excluded). An
# event with no call-site has no v1 emission path: that is exactly the
# train.hidden_norm (recurrent-only) gap the spec wants surfaced as a
# strict xfail rather than silently skipped.
_SRC = Path(__file__).parents[2] / "src" / "seq_sklearn"
_SRC_TEXT: str = "\n".join(p.read_text() for p in _SRC.rglob("*.py") if p.name != "logging.py")
# An event has a v1 emission path if its enum member is referenced
# (literal ``Event.NAME``) OR its dotted value appears as a string key
# (the TFT backbone emits the entropy events via a string-keyed payload
# dict, not the enum literal).
_EMITTING_EVENTS: frozenset[str] = frozenset(
    member.value
    for member in Event
    if f"Event.{member.name}" in _SRC_TEXT or f'"{member.value}"' in _SRC_TEXT
)

# The authoritative F11 table (requirements.md F11). Event value ->
# (logging level, frozenset of payload keys). calibration.fit has a
# task-dependent schema; both variants are listed as separate cases.
_F11_TABLE: dict[str, tuple[int, frozenset[str]]] = {
    "train.grad_norm": (logging.DEBUG, frozenset({"step", "grad_norm", "lr"})),
    "train.epoch": (
        logging.INFO,
        frozenset({"epoch", "train_loss", "val_loss", "val_metric"}),
    ),
    "train.var_selection_entropy": (
        logging.INFO,
        frozenset({"epoch", "static_entropy", "temporal_entropy"}),
    ),
    "train.attention_entropy": (
        logging.INFO,
        frozenset({"epoch", "entropy_per_head"}),
    ),
    "train.hidden_norm": (logging.INFO, frozenset({"epoch", "mean_hidden_norm"})),
    "train.nan_step_skipped": (
        logging.WARNING,
        frozenset({"step", "consecutive_nan_count"}),
    ),
    "train.mixed_precision_diverged": (
        logging.ERROR,
        frozenset({"step", "precision", "consecutive_skipped", "reason"}),
    ),
    "calibration.small_set": (
        logging.WARNING,
        frozenset({"strategy", "cal_size", "min_recommended"}),
    ),
    "optuna.trial_pruned": (
        logging.INFO,
        frozenset({"trial_number", "epoch", "reason"}),
    ),
    "data.duplicate_floor_breach_count": (
        logging.WARNING,
        frozenset({"count", "min_periods_predict"}),
    ),
    "data.unseen_categories": (logging.INFO, frozenset({"column", "count"})),
    "hardware.detect": (
        logging.INFO,
        frozenset({"tier", "cuda_compute_capability", "selected_precision"}),
    ),
}

# calibration.fit carries a task-dependent payload (classifier vs
# regressor); both schemas are part of the F11 contract.
_CALIBRATION_FIT_SCHEMAS: dict[str, frozenset[str]] = {
    "classifier": frozenset({"strategy", "cal_size", "pre_ece", "post_ece"}),
    "regressor": frozenset({"strategy", "cal_size", "pre_coverage", "post_coverage"}),
}

_HIDDEN_NORM_XFAIL = pytest.mark.xfail(
    reason="no v1 emission path; recurrent family ships in v3",
    strict=True,
)


def _params() -> list[object]:
    cases: list[object] = []
    for value, (_level, _keys) in _F11_TABLE.items():
        if value == "train.hidden_norm":
            cases.append(pytest.param(value, marks=_HIDDEN_NORM_XFAIL))
        else:
            cases.append(value)
    return cases


def test_event_enum_matches_f11_table() -> None:
    """The Event enum is exactly the F11 table (no untracked extras).

    Adding or removing an Event without updating the F11 table fails
    here. ``calibration.fit`` is in the table with a task-dependent
    payload so it is parametrized separately, not in ``_F11_TABLE``.
    """
    enum_values = {e.value for e in Event}
    accounted = set(_F11_TABLE) | {"calibration.fit"}
    assert enum_values == accounted, (
        f"Event enum vs F11 table drift. Only-in-enum: "
        f"{sorted(enum_values - accounted)}; only-in-table: "
        f"{sorted(accounted - enum_values)}."
    )


@pytest.mark.parametrize("event_value", _params())
def test_event_emits_with_schema(event_value: str, caplog: pytest.LogCaptureFixture) -> None:
    level, keys = _F11_TABLE[event_value]
    event = Event(event_value)
    logger = logging.getLogger("seq_sklearn.test_f11")
    payload = {k: 0 for k in keys}
    with caplog.at_level(logging.DEBUG, logger="seq_sklearn.test_f11"):
        emit(logger, event, level=level, **payload)
    records = [r for r in caplog.records if getattr(r, "event", None) == event_value]
    assert len(records) == 1, f"{event_value}: expected exactly one record"
    record = records[0]
    assert record.levelno == level
    assert record.message == event_value
    assert set(record.payload) == set(keys)  # type: ignore[attr-defined]
    # A v1 emission path must exist. train.hidden_norm has none
    # (recurrent-only, v3): this assertion fails for it, which the
    # strict xfail marker turns into the explicit, expected CI gap.
    assert event_value in _EMITTING_EVENTS, (
        f"{event_value}: no v1 emission call-site in src/seq_sklearn"
    )


@pytest.mark.parametrize("variant", sorted(_CALIBRATION_FIT_SCHEMAS))
def test_calibration_fit_emits_per_task_schema(
    variant: str, caplog: pytest.LogCaptureFixture
) -> None:
    keys = _CALIBRATION_FIT_SCHEMAS[variant]
    logger = logging.getLogger("seq_sklearn.test_f11")
    payload = {k: 0 for k in keys}
    with caplog.at_level(logging.INFO, logger="seq_sklearn.test_f11"):
        emit(logger, Event.CALIBRATION_FIT, level=logging.INFO, **payload)
    records = [r for r in caplog.records if getattr(r, "event", None) == "calibration.fit"]
    assert len(records) == 1
    assert set(records[0].payload) == set(keys)  # type: ignore[attr-defined]
