"""Friedman + Holm correction over the (model, dataset) matrix (B7.5).

Design B7.5 names the global rank-difference test for the
HPO-uplift report:

> "Across datasets, the input matrix is one cell per (model, dataset)
> equal to the seed-mean of the B5 primary loss.
> `scipy.stats.friedmanchisquare` over that matrix tests the global
> rank-difference null; a Nemenyi critical-difference diagram
> visualizes it. ... Holm correction over the set of seq-model /
> baseline ensemble pairs tested (not over the dataset
> cross-product)."

This module ships the two primitives the report consumes:

- `friedman_holm`: drives the matrix-level Friedman omnibus plus the
  Holm-adjusted pairwise comparisons against a designated reference
  model (the seq_sklearn family in the report). The omnibus follows
  scipy's `friedmanchisquare` exactly; the pairwise step uses the
  Wilcoxon signed-rank statistic across datasets per the design.
- `holm_correction`: the reusable Holm-Bonferroni adjustment
  routine, used both by `friedman_holm` internally and by callers
  that already have a list of raw p-values.

Both helpers are pure numerics; the report module is the single
consumer that knows how to extract the (model, dataset) matrix from
the manifest.
"""

import logging

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

__all__ = [
    "FriedmanHolmResult",
    "HolmAdjustedPValue",
    "friedman_holm",
    "holm_correction",
]


logger = logging.getLogger(__name__)


class HolmAdjustedPValue(BaseModel):
    """One pairwise comparison's Wilcoxon statistic + Holm-adjusted
    p-value.

    `comparison_name` is the label the caller assigns to the
    (reference_model, other_model) pair (e.g.
    ``"tft_classifier vs xgb_classifier"``); the module doesn't
    interpret it, just round-trips it for the report.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    comparison_name: str
    raw_p_value: float = Field(ge=0.0, le=1.0)
    holm_adjusted_p_value: float = Field(ge=0.0, le=1.0)
    wilcoxon_statistic: float
    n_datasets: int = Field(ge=0)


class FriedmanHolmResult(BaseModel):
    """Full matrix-level test result.

    `friedman_statistic` and `friedman_p_value` come from
    `scipy.stats.friedmanchisquare`; `pairwise` holds one
    Holm-adjusted entry per `(reference_model, other_model)` pair
    requested by the caller. `family_size` is the number of
    comparisons the Holm correction was applied across (B7.5: the
    set of seq-model / baseline pairs tested, not the dataset
    cross-product), recorded so the report can disclose it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_models: int = Field(ge=0)
    n_datasets: int = Field(ge=0)
    friedman_statistic: float | None = None
    friedman_p_value: float | None = None
    pairwise: tuple[HolmAdjustedPValue, ...] = ()
    family_size: int = Field(ge=0)


def holm_correction(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni step-down correction to raw p-values.

    The Holm procedure sorts the m p-values ascending, scales the
    smallest by m, the next by m-1, ..., the largest by 1, and
    enforces monotonicity by taking the running max. Adjusted
    p-values are clamped to [0, 1].

    Returns the adjusted p-values in the SAME order as the input
    (the caller doesn't need to track the sort permutation).

    An empty input returns ``[]``. Single-element input returns
    the value unchanged (a 1-comparison family needs no correction).

    Raises:
        ValueError: any p-value is outside [0, 1] or NaN.
    """
    if not p_values:
        return []
    arr = np.asarray(p_values, dtype=float)
    if np.any(np.isnan(arr)):
        raise ValueError(f"holm_correction: NaN in p-values: {p_values}")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"holm_correction: p-values must be in [0, 1]: {p_values}")

    m = arr.size
    order = np.argsort(arr, kind="stable")
    scales = np.arange(m, 0, -1, dtype=float)  # m, m-1, ..., 1
    scaled_sorted = arr[order] * scales
    # Enforce monotone non-decreasing along the sort order, then
    # clamp to [0, 1]. The running max guarantees the adjusted
    # p-values do not decrease as the raw rank increases.
    monotone_sorted = np.maximum.accumulate(scaled_sorted)
    adjusted_sorted = np.minimum(monotone_sorted, 1.0)
    # Unsort: place each adjusted value back at its original index.
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return [float(v) for v in adjusted]


def _build_matrix(
    matrix: pd.DataFrame,
) -> tuple[list[str], list[str], np.ndarray]:
    """Coerce the input DataFrame to (models, datasets, values).

    `matrix` is shaped ``(n_models, n_datasets)``: one row per model,
    one column per dataset, cell = seed-mean of the primary loss.
    Returns the model labels (row index, sorted), dataset labels
    (column index, sorted), and the numpy matrix in that order. NaN
    cells propagate so the caller can decide how to handle them.
    """
    if matrix.empty:
        return [], [], np.empty((0, 0), dtype=float)
    sorted_models = sorted(map(str, matrix.index))
    sorted_datasets = sorted(map(str, matrix.columns))
    sub = matrix.loc[sorted_models, sorted_datasets].astype(float)
    values = sub.to_numpy()
    return sorted_models, sorted_datasets, values


def friedman_holm(
    matrix: pd.DataFrame,
    *,
    reference_model: str,
) -> FriedmanHolmResult:
    """Friedman omnibus + Wilcoxon-signed-rank pairwise (vs reference)
    with Holm correction.

    The omnibus needs at least 3 models (scipy
    `friedmanchisquare`'s documented minimum) and 2 datasets; below
    that the result is returned with the structural fields
    populated but `friedman_statistic` and `friedman_p_value` as
    ``None`` and `pairwise` empty. Datasets with any NaN cell are
    dropped from the matrix before both tests; if fewer than 2
    datasets remain, the same null-result shape is returned.

    The pairwise comparisons are between `reference_model` and every
    other model in the matrix. `family_size` is the number of
    comparisons the Holm correction is applied across (== number of
    other models). A single pairwise comparison gets Holm-adjusted
    p-value equal to its raw p-value.

    Args:
        matrix: ``(n_models, n_datasets)`` DataFrame of seed-mean
            primary-loss values. Lower-is-better is the convention
            but the test is rank-invariant under monotone transforms,
            so direction doesn't change ranks.
        reference_model: row label to compare every other row
            against. Must appear in ``matrix.index``.

    Raises:
        ValueError: ``reference_model`` is not in ``matrix.index``.
    """
    models, datasets, values = _build_matrix(matrix)
    n_models = len(models)
    n_datasets_in = len(datasets)
    if n_models > 0 and reference_model not in set(models):
        raise ValueError(
            f"friedman_holm: reference_model {reference_model!r} not in matrix.index={models}"
        )

    if n_models < 3 or n_datasets_in < 2:
        return FriedmanHolmResult(
            n_models=n_models,
            n_datasets=n_datasets_in,
            friedman_statistic=None,
            friedman_p_value=None,
            pairwise=(),
            family_size=0,
        )

    # Drop dataset columns that carry any NaN across the model rows;
    # both `friedmanchisquare` and `wilcoxon` reject NaN inputs and a
    # silent NaN-propagation here would be a hidden bug source.
    column_has_nan: np.ndarray = np.asarray(np.isnan(values).any(axis=0))
    if bool(column_has_nan.any()):
        keep_mask: list[bool] = [bool(v) for v in column_has_nan.tolist()]
        keep_mask = [not v for v in keep_mask]  # invert: True == keep
        kept_datasets = [d for d, keep in zip(datasets, keep_mask, strict=True) if keep]
        kept_values = values[:, keep_mask]
        logger.info(
            "friedman_holm: dropped %d dataset column(s) carrying NaN (kept %d): dropped=%s",
            int(column_has_nan.sum()),
            len(kept_datasets),
            [d for d, keep in zip(datasets, keep_mask, strict=True) if not keep],
        )
    else:
        kept_datasets = datasets
        kept_values = values

    if len(kept_datasets) < 2:
        return FriedmanHolmResult(
            n_models=n_models,
            n_datasets=len(kept_datasets),
            friedman_statistic=None,
            friedman_p_value=None,
            pairwise=(),
            family_size=0,
        )

    # Friedman omnibus across the (model, dataset) matrix. scipy's
    # `friedmanchisquare(*sequences)` takes one sequence per
    # treatment (== one per model here); each sequence is the per-
    # dataset measurement vector.
    friedman = stats.friedmanchisquare(*[kept_values[i, :] for i in range(n_models)])

    ref_idx = models.index(reference_model)
    other_idxs = [i for i in range(n_models) if i != ref_idx]
    raw_pvalues: list[float] = []
    raw_statistics: list[float] = []
    comparison_labels: list[str] = []
    for j in other_idxs:
        diffs = kept_values[ref_idx, :] - kept_values[j, :]
        # Wilcoxon signed-rank: H0 = paired diffs symmetric about 0
        # (i.e., no consistent rank advantage). All-zero diffs are
        # not testable; surface a sentinel and let Holm receive 1.0
        # (the most-conservative neutral outcome).
        nonzero = np.count_nonzero(diffs)
        if nonzero == 0:
            raw_pvalues.append(1.0)
            raw_statistics.append(0.0)
        else:
            # `zero_method="wilcox"` matches scipy's documented default
            # for non-zero diffs and drops exact zeros (which is the
            # signed-rank contract). `alternative="two-sided"` is the
            # design's two-sided null.
            res = stats.wilcoxon(
                diffs,
                zero_method="wilcox",
                alternative="two-sided",
            )
            raw_pvalues.append(float(res.pvalue))  # pyright: ignore[reportAttributeAccessIssue]
            raw_statistics.append(float(res.statistic))  # pyright: ignore[reportAttributeAccessIssue]
        comparison_labels.append(f"{reference_model} vs {models[j]}")

    adjusted = holm_correction(raw_pvalues)
    pairwise = tuple(
        HolmAdjustedPValue(
            comparison_name=label,
            raw_p_value=raw,
            holm_adjusted_p_value=adj,
            wilcoxon_statistic=stat,
            n_datasets=len(kept_datasets),
        )
        for label, raw, adj, stat in zip(
            comparison_labels, raw_pvalues, adjusted, raw_statistics, strict=True
        )
    )

    return FriedmanHolmResult(
        n_models=n_models,
        n_datasets=len(kept_datasets),
        friedman_statistic=float(friedman.statistic),
        friedman_p_value=float(friedman.pvalue),
        pairwise=pairwise,
        family_size=len(other_idxs),
    )
