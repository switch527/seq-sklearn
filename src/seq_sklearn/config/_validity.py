"""F5 validity-matrix cross-field validator.

Encodes the legal (task_type, loss_strategy, imbalance_strategy,
calibration_strategy) cells per requirements F5 / docs/architecture.md
A4. The :func:`check_combo` pure function is shared between
:class:`seq_sklearn.config.base.BaseModelConfig`'s model_validator and
the Optuna search-space sampler (so the sampler can stay closed under
the validity matrix by construction).
"""

from typing import Final

from seq_sklearn.config._domains import V1_1_TASK_TYPES

__all__ = ["check_combo", "legal_strategies_for", "legal_task_loss_pairs"]


def legal_task_loss_pairs(*, include_v1_1: bool = False) -> frozenset[tuple[str, str]]:
    """Return the legal ``(task_type, loss_strategy)`` pairs.

    Derived from the single ``_LEGAL_CELLS`` table so callers (the
    config validator, the loss factory, the Optuna sampler) never
    hand-maintain a parallel copy of the F5 validity matrix. v1.1 task
    pairs are excluded unless ``include_v1_1`` is set.
    """
    return frozenset(
        (task, loss) for (task, loss) in _LEGAL_CELLS if include_v1_1 or task not in V1_1_TASK_TYPES
    )


# Legal cells per the F5 validity matrix. Keyed by (task_type,
# loss_strategy); each value is (legal_imbalance_strategies,
# legal_calibration_strategies). The two v1.1 task types
# (``multilabel``, ``regression_multioutput``) have entries here too so
# any future v1.1 enablement only needs to remove the v1.1 guard below
# and keep this table intact.
_LEGAL_CELLS: Final[dict[tuple[str, str], tuple[tuple[str, ...], tuple[str, ...]]]] = {
    ("binary", "cross_entropy"): (
        ("none", "class_weighted", "oversample_minority", "undersample_majority"),
        ("none", "temperature", "platt", "isotonic"),
    ),
    ("binary", "focal"): (
        ("none", "oversample_minority", "undersample_majority"),
        ("none", "temperature", "platt", "isotonic"),
    ),
    ("multiclass", "cross_entropy"): (
        ("none", "class_weighted", "oversample_minority", "undersample_majority"),
        ("none", "temperature", "isotonic"),
    ),
    ("multiclass", "focal"): (
        ("none", "oversample_minority", "undersample_majority"),
        ("none", "temperature", "isotonic"),
    ),
    ("multilabel", "cross_entropy"): (
        ("none", "oversample_minority", "undersample_majority"),
        ("none", "temperature", "isotonic"),
    ),
    ("multilabel", "focal"): (
        ("none", "oversample_minority", "undersample_majority"),
        ("none", "temperature", "isotonic"),
    ),
    ("regression_point", "mse"): (("none",), ("none",)),
    ("regression_point", "mae"): (("none",), ("none",)),
    ("regression_point", "huber"): (("none",), ("none",)),
    ("regression_quantile", "pinball"): (
        ("none",),
        ("none", "conformal", "isotonic_quantile"),
    ),
    ("regression_multioutput", "mse"): (("none",), ("none",)),
    ("regression_multioutput", "mae"): (("none",), ("none",)),
    ("regression_multioutput", "huber"): (("none",), ("none",)),
}


def legal_strategies_for(
    task_type: str, loss_strategy: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(legal_imbalance, legal_calibration)`` for an F5 cell.

    The sanctioned public accessor for the per-cell legal subsets so the
    Optuna sampler stays closed under the validity matrix without
    reaching into the private ``_LEGAL_CELLS`` table or hand-copying it.

    Raises:
        ValueError: if ``task_type`` is a v1.1 task, or
            ``(task_type, loss_strategy)`` is not a legal cell. The
            message mirrors :func:`check_combo`.
    """
    if task_type in V1_1_TASK_TYPES:
        raise ValueError(
            f"task_type={task_type!r} is scheduled for v1.1; v1 supports "
            "single-output classification and regression only."
        )
    cell = _LEGAL_CELLS.get((task_type, loss_strategy))
    if cell is None:
        legal_losses = sorted({lt for (tt, lt) in _LEGAL_CELLS if tt == task_type})
        raise ValueError(
            f"loss_strategy={loss_strategy!r} is not legal for "
            f"task_type={task_type!r}. Legal losses for this task: "
            f"{legal_losses}."
        )
    return cell


def check_combo(
    task_type: str,
    loss_strategy: str,
    imbalance_strategy: str,
    calibration_strategy: str,
) -> None:
    """Raise :class:`ValueError` if the four-field combination is illegal.

    Pure function. Returns ``None`` on legal combinations.

    v1.1 task types raise with a "scheduled for v1.1" message so callers
    on v1 get a clear forward-looking error rather than a downstream
    shape mismatch.
    """
    legal_imbalance, legal_calibration = legal_strategies_for(task_type, loss_strategy)
    if imbalance_strategy not in legal_imbalance:
        raise ValueError(
            f"imbalance_strategy={imbalance_strategy!r} is not legal for "
            f"(task_type={task_type!r}, loss_strategy={loss_strategy!r}). "
            f"Legal imbalance strategies for this cell: {list(legal_imbalance)}."
        )
    if calibration_strategy not in legal_calibration:
        raise ValueError(
            f"calibration_strategy={calibration_strategy!r} is not legal "
            f"for (task_type={task_type!r}, loss_strategy={loss_strategy!r}). "
            f"Legal calibration strategies for this cell: "
            f"{list(legal_calibration)}."
        )
