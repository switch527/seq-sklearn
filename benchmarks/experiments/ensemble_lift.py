"""Phase B11 ensemble-lift experiment driver (B6.2.5).

Closes the B6-followup deferred deliverable named in the design
(B6.2.5): build a `GBM-only` ensemble and a `GBM+seq` ensemble on
the same folds, then run a Wilcoxon signed-rank test paired over
datasets on the per-dataset Δloss between them with Holm
correction over the seq-model / baseline pairs tested. The
headline test answers "does adding the seq-sklearn deep model to
a GBM ensemble lower the primary loss?".

Unlike B5 / B8, this is a REPORT-ONLY phase: every cell the
ensembles consume already lives in the predictions shards the B5
driver writes for variant="default" cells. The driver reads those
shards, builds per-fold averaged probability (classification) or
mean-prediction (regression) ensembles, computes the per-cell
primary loss for each ensemble, and aggregates per-dataset.

Scope (v1):

- Ensemble construction: SIMPLE AVERAGE of base-model predictions
  on the same (seed, fold, panel_row_index) rows. Classification
  averages probabilities; regression averages point predictions.
  The design also names a "properly nested stacked meta-learner"
  variant; that ships in a B11-followup branch (the meta-learner
  needs an extra inner CV split on top of the B5 fold layout).

- Loss metric: log_loss for binary + multiclass, RMSE for
  regression_point. Quantile cells inherit the B5 skip.

- Comparison: per (dataset, seed-fold pair), compute
  `loss_GBM_only(seed, fold)` and `loss_GBM_plus_seq(seed, fold)`,
  then `Δloss = loss_GBM_only - loss_GBM_plus_seq`. Aggregate per
  dataset by seed-mean. The Wilcoxon signed-rank test runs paired
  over datasets on the per-dataset Δloss.

- Holm correction: the family-size argument is the number of
  (seq-model, baseline-family) pairs tested. B11 ships one
  combination (seq_sklearn TFT classifier + GBM family) so the
  family size is 1; future seq models extend it.

- Oracle upper bound: per-sample best across (GBM-only,
  GBM+seq) per cell. This is the "if we always picked the right
  ensemble per-row" reference; the report quotes it as a
  ceiling for the lift.

B11-followup deferrals (carried forward):

- Stacked meta-learner ensemble: B11 ships the simple-average
  ensemble only. The stacked variant needs a nested-CV
  meta-learner training surface that doesn't exist today.
- Cross-family-set lift: B11's GBM family is a single roster
  (LightGBM + XGBoost + CatBoost averaged). A future branch can
  add an Optuna-tuned variant or per-library lifts.
"""

import logging
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments.raw_loss import RunEnvironment
from benchmarks.manifest import cell_key, load_run
from benchmarks.predictions import (
    has_predictions,
    load_predictions,
    proba_matrix,
)
from benchmarks.registry import get_model
from benchmarks.registry.models import ModelNotRegisteredError
from benchmarks.stats.friedman import holm_correction

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BASELINE_FAMILY",
    "DEFAULT_SEQ_FAMILY",
    "ComputePerCellLiftDeltasResult",
    "EnsembleLiftExperimentResult",
    "PerCellLiftDelta",
    "PerDatasetLift",
    "WilcoxonResult",
    "build_cells_table",
    "compute_per_cell_lift_deltas",
    "model_families",
    "run_ensemble_lift",
]


# Default seq family + baseline family. B11 ships one comparison
# (seq_sklearn vs gbm); a future branch wires multi-family
# combinations by extending these defaults at the spec level.
DEFAULT_SEQ_FAMILY = "seq_sklearn"
DEFAULT_BASELINE_FAMILY = "gbm"

# Primary-loss column per task. Quantile cells inherit B5's skip.
_PRIMARY_LOSS_BY_TASK: dict[str, str] = {
    "binary": "log_loss",
    "multiclass": "log_loss",
    "regression_point": "rmse",
}


class WilcoxonResult(BaseModel):
    """Wilcoxon signed-rank test outcome on per-dataset Δloss.

    `statistic`, `p_value`, and `n_datasets` come from
    `scipy.stats.wilcoxon` on the per-dataset Δloss vector after
    dropping NaNs. `holm_adjusted_p_value` is the
    Holm-Bonferroni adjustment over the (seq-model, baseline-
    family) pair family; B11 ships one pair so the adjusted value
    equals the raw value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    statistic: float | None = None
    p_value: float | None = None
    holm_adjusted_p_value: float | None = None
    n_datasets: int = Field(ge=0)
    family_size: int = Field(ge=0)


class PerDatasetLift(BaseModel):
    """One dataset's ensemble-lift summary.

    `delta_loss` is `loss(GBM-only) - loss(GBM+seq)`: a positive
    value means adding the deep model lowered the loss.
    `oracle_delta_loss` is `loss(GBM-only) - loss(per-sample best)`,
    the ceiling for the lift on that dataset.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    task_type: str
    primary_loss_column: str
    n_cells_paired: int = Field(ge=0)
    loss_gbm_only_mean: float | None = None
    loss_gbm_plus_seq_mean: float | None = None
    delta_loss_mean: float | None = None
    delta_loss_std: float | None = None
    oracle_loss_mean: float | None = None
    oracle_delta_loss_mean: float | None = None
    # Footnote-able states: a (dataset, seed, fold) cell where one
    # of the two ensembles couldn't be built (e.g., no GBM cells in
    # the B5 manifest for that dataset) yields `delta_loss=None`
    # and lands in the report's "Incomplete groups" block.
    no_gbm_predictions: bool = False
    no_seq_predictions: bool = False


class EnsembleLiftExperimentResult(BaseModel):
    """Summary returned by `run_ensemble_lift`.

    `wilcoxon` is the matrix-level test across datasets; `rows`
    carries the per-dataset Δlosses + oracle bounds.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    seq_family: str
    baseline_family: str
    rows: tuple[PerDatasetLift, ...]
    wilcoxon: WilcoxonResult


def _assert_ensemble_lift_configured(experiments: Iterable[ExperimentSpec]) -> None:
    """Raise if no `ensemble_lift` spec is declared.

    The driver aggregates over the (seed, fold) pairs found in the
    B5 manifest rather than the spec's `seeds`, so this validator
    is side-effect-only.
    """
    for spec in experiments:
        if spec.kind == "ensemble_lift":
            return
    raise ValueError(
        "run_ensemble_lift: BenchmarkConfig declares no "
        "ensemble_lift experiment; add ExperimentSpec(kind="
        "'ensemble_lift') to the config"
    )


def model_families(manifest: pd.DataFrame) -> dict[str, str]:
    """Map every `model_name` in the manifest to its registered
    family.

    Lookups are best-effort: a model that was registered at run
    time but is no longer in the registry maps to the empty
    string so the partitioning logic at the call site can
    surface it as a footnote group.
    """
    out: dict[str, str] = {}
    for name in manifest["model_name"].dropna().unique():
        try:
            spec = get_model(str(name))
        except ModelNotRegisteredError:
            out[str(name)] = ""
            continue
        out[str(name)] = spec.family
    return out


def _per_cell_primary_loss(
    *,
    task_type: str,
    y_true: np.ndarray,
    proba: np.ndarray | None,
    y_pred: np.ndarray | None,
) -> float | None:
    """Per-cell primary loss for the ensemble.

    Classification: `log_loss` from the averaged probability
    matrix. Regression: RMSE from the averaged point prediction.
    Quantile: not supported (caller filters those cells before
    reaching this helper).
    """
    if task_type in {"binary", "multiclass"}:
        if proba is None:
            return None
        # Clip to avoid log(0). sklearn's log_loss does the same;
        # we inline the safe-log so the ensemble math stays
        # within this module.
        eps = 1e-15
        proba_clipped = np.clip(proba, eps, 1.0 - eps)
        # `y_true` is float-encoded class index (B6.2.1
        # convention: predictions are written with `y_true`
        # cast to float64). Cast back to int for indexing.
        true_idx = y_true.astype(np.int64)
        # Bound the index to the proba column count to defend
        # against a future class-label encoding regression.
        if true_idx.min() < 0 or true_idx.max() >= proba_clipped.shape[1]:
            return None
        per_row = -np.log(proba_clipped[np.arange(len(true_idx)), true_idx])
        return float(per_row.mean())
    if task_type == "regression_point":
        if y_pred is None:
            return None
        # The B5 contract guarantees `y_true` is finite, but a buggy
        # upstream adapter could write NaN-tinged predictions. Skip
        # NaN rows rather than poisoning the dataset mean (which
        # would propagate through Δloss into the Wilcoxon vector).
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if not mask.any():
            return None
        diff = y_true[mask] - y_pred[mask]
        return float(np.sqrt(np.mean(diff**2)))
    return None


def _per_sample_oracle_loss(
    *,
    task_type: str,
    y_true: np.ndarray,
    proba_a: np.ndarray | None,
    proba_b: np.ndarray | None,
    pred_a: np.ndarray | None,
    pred_b: np.ndarray | None,
) -> float | None:
    """Per-sample best-of-two oracle loss.

    For each row, take the lower of the two ensemble's per-row
    contributions. This is the design-named "oracle (per-sample
    best) upper bound" the report quotes alongside the achieved
    lift.
    """
    if task_type in {"binary", "multiclass"}:
        if proba_a is None or proba_b is None:
            return None
        eps = 1e-15
        clip_a = np.clip(proba_a, eps, 1.0 - eps)
        clip_b = np.clip(proba_b, eps, 1.0 - eps)
        true_idx = y_true.astype(np.int64)
        if (
            true_idx.min() < 0
            or true_idx.max() >= clip_a.shape[1]
            or true_idx.max() >= clip_b.shape[1]
        ):
            return None
        rows = np.arange(len(true_idx))
        nll_a = -np.log(clip_a[rows, true_idx])
        nll_b = -np.log(clip_b[rows, true_idx])
        per_row = np.minimum(nll_a, nll_b)
        return float(per_row.mean())
    if task_type == "regression_point":
        if pred_a is None or pred_b is None:
            return None
        mask = ~(np.isnan(y_true) | np.isnan(pred_a) | np.isnan(pred_b))
        if not mask.any():
            return None
        sqerr_a = (y_true[mask] - pred_a[mask]) ** 2
        sqerr_b = (y_true[mask] - pred_b[mask]) ** 2
        per_row = np.minimum(sqerr_a, sqerr_b)
        return float(np.sqrt(per_row.mean()))
    return None


def _join_predictions(
    *,
    cells: pd.DataFrame,
    output_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Inner-join the predictions shards for every cell in
    `cells` on `panel_row_index`.

    Returns `(panel_row_index, y_true, ensemble_value)` where the
    ensemble value is either the averaged probability matrix
    (classification, 2D) or the averaged point prediction
    (regression, 1D). Returns None if any cell's shard is missing
    or no panel rows survive the inner join (`ensemble_value` is
    never None inside a returned tuple; the whole tuple is None
    on failure).
    """
    if cells.empty:
        return None
    # Pull every cell's predictions; bail if any shard is missing.
    frames: list[pd.DataFrame] = []
    for _, row in cells.iterrows():
        key = str(row["cell_key"])
        if not has_predictions(output_root, key):
            return None
        frames.append(load_predictions(output_root, key))
    # Per-cell prefix join: build one DataFrame per cell with
    # `c{i}_*` column names then inner-join all on
    # `panel_row_index`. Per-cell prefixing avoids pandas's
    # `lsuffix/rsuffix` join which only handles two frames.
    keyed_prefixed: list[pd.DataFrame] = []
    for i, frame in enumerate(frames):
        prefixed = frame.add_prefix(f"c{i}_")
        prefixed = prefixed.rename(columns={f"c{i}_panel_row_index": "panel_row_index"})
        prefixed = prefixed.set_index("panel_row_index")
        # A duplicate panel_row_index in any shard would cause
        # `DataFrame.join` below to produce a silent Cartesian
        # product, inflating row counts and biasing the averaged
        # probabilities. Reject the shard rather than averaging
        # mis-aligned rows.
        if prefixed.index.has_duplicates:
            logger.warning(
                "_join_predictions: cell %d shard has duplicate panel_row_index "
                "values; rejecting (B5 contract violation)",
                i,
            )
            return None
        keyed_prefixed.append(prefixed)
    if not keyed_prefixed:
        return None
    joined_clean = keyed_prefixed[0]
    for next_prefixed in keyed_prefixed[1:]:
        joined_clean = joined_clean.join(next_prefixed, how="inner")
    if joined_clean.empty:
        return None
    panel_row_index = joined_clean.index.to_numpy()
    y_true = joined_clean["c0_y_true"].to_numpy()
    # Verify y_true alignment across cells; a mismatch indicates a
    # B5 contract violation (predictions on the same panel rows
    # should share y_true).
    for i in range(1, len(frames)):
        col = f"c{i}_y_true"
        if not np.array_equal(joined_clean[col].to_numpy(), y_true):
            logger.warning(
                "_join_predictions: y_true mismatch between cell 0 and cell %d "
                "on panel_row_index intersection; B5 manifest may be "
                "inconsistent",
                i,
            )
            return None
    # Classification path: average probability matrices.
    proba_blocks: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        proba_i = proba_matrix(frame)
        if proba_i is None:
            proba_blocks = []
            break
        # Re-align proba_i to the joined index order. The original
        # frame has `panel_row_index` as a column, so its row order
        # may differ from the joined index; reindex via the cell's
        # prefix view.
        aligned_idx = keyed_prefixed[i].index.get_indexer(panel_row_index)
        # `aligned_idx` maps the joined position to the original
        # cell's row position. Negative values indicate misses
        # (impossible after inner join, but defensive).
        if (aligned_idx < 0).any():
            proba_blocks = []
            break
        proba_blocks.append(proba_i[aligned_idx])
    if proba_blocks:
        # All frames have probabilities + identical column count
        # is guaranteed by the B5 contract for a given (dataset,
        # task). Average rowwise.
        col_counts = {p.shape[1] for p in proba_blocks}
        if len(col_counts) != 1:
            logger.warning(
                "_join_predictions: cells disagree on proba column count: %s",
                sorted(col_counts),
            )
            return None
        averaged_proba = np.mean(np.stack(proba_blocks, axis=0), axis=0)
        return panel_row_index, y_true, averaged_proba

    # Regression path: average point predictions.
    pred_blocks: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        # Same realignment as the proba path.
        aligned_idx = keyed_prefixed[i].index.get_indexer(panel_row_index)
        if (aligned_idx < 0).any():
            return None
        pred_blocks.append(frame["y_pred"].to_numpy()[aligned_idx])
    averaged_pred = np.mean(np.stack(pred_blocks, axis=0), axis=0)
    return panel_row_index, y_true, averaged_pred


def build_cells_table(
    manifest: pd.DataFrame, *, families: dict[str, str], target_family: str
) -> pd.DataFrame:
    """Filter the manifest to OK cells in `target_family`."""
    ok = manifest.loc[manifest["skipped_reason"].isna()].copy()
    ok["model_family"] = ok["model_name"].map(families).fillna("")
    keep = ok.loc[ok["model_family"] == target_family]
    # Cell key is reconstructed for the predictions-shard lookup.
    keep = keep.copy()
    # Empty DataFrame: `apply(axis=1)` returns a DataFrame rather
    # than a Series, which would crash the column assignment.
    # Short-circuit with an empty string Series.
    if keep.empty:
        keep["cell_key"] = pd.Series([], dtype=str)
    else:

        def _row_cell_key(row: pd.Series) -> str:
            return cell_key(
                dataset_name=str(row["dataset_name"]),
                model_name=str(row["model_name"]),
                seed=int(row["seed"]),
                variant=str(row["variant"]),
                fold_index=int(row["fold_index"]),
            )

        keep["cell_key"] = keep.apply(_row_cell_key, axis=1)
    return keep[
        [
            "dataset_name",
            "model_name",
            "seed",
            "fold_index",
            "task_type",
            "variant",
            "cell_key",
        ]
    ]


class PerCellLiftDelta(BaseModel):
    """One paired-cell record from
    `compute_per_cell_lift_deltas` (B16 refactor extraction).

    `loss_gbm` and `loss_gbm_plus_seq` are the per-cell primary
    losses; `delta_loss = loss_gbm - loss_gbm_plus_seq`.
    `oracle_loss` is None when the per-sample oracle could not be
    computed (e.g., one of the ensembles produced no usable
    rows). The bootstrap aggregator consumes the `delta_loss`
    array; B11's `_per_dataset_lift` aggregates the same records
    by mean + std.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    fold_index: int
    loss_gbm: float
    loss_gbm_plus_seq: float
    delta_loss: float
    oracle_loss: float | None = None


class ComputePerCellLiftDeltasResult(BaseModel):
    """Return shape for `compute_per_cell_lift_deltas`.

    `cells` is the paired-cell records; `seen_no_gbm` /
    `seen_no_seq` mirror B11's incomplete-block flags so both
    `_per_dataset_lift` (B11) and the B16 aggregator can surface
    the existing sentinel states without recomputing them.

    This is the public return type of a public function; the
    name carries no leading underscore so cross-module
    consumers can rely on its field set.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cells: tuple[PerCellLiftDelta, ...]
    seen_no_gbm: bool = False
    seen_no_seq: bool = False
    selector: str


def compute_per_cell_lift_deltas(
    *,
    dataset_name: str,
    task_type: str,
    seed_fold_pairs: list[tuple[int, int]],
    gbm_cells: pd.DataFrame,
    seq_cells: pd.DataFrame,
    output_root: Path,
) -> ComputePerCellLiftDeltasResult:
    """Per-(seed, fold) paired-cell delta computation.

    Iterates `seed_fold_pairs` and at each pair: builds the
    GBM-only and GBM+seq ensembles via the existing
    `_join_predictions` helper, computes the per-cell primary
    loss for each, and emits one `PerCellLiftDelta` if both
    losses are finite. Returns the records list plus the same
    `seen_no_gbm` / `seen_no_seq` flags that B11's
    `_per_dataset_lift` previously tracked inline.

    B11's `_per_dataset_lift` is a thin aggregation over this
    helper's output; the B16 bootstrap aggregator consumes the
    same records to bootstrap the per-cell `delta_loss` array.
    """
    selector = _PRIMARY_LOSS_BY_TASK.get(task_type)
    if selector is None:
        return ComputePerCellLiftDeltasResult(cells=(), selector="unknown")

    records: list[PerCellLiftDelta] = []
    seen_no_gbm = False
    seen_no_seq = False

    for seed, fold in seed_fold_pairs:
        gbm_subset = gbm_cells.loc[(gbm_cells["seed"] == seed) & (gbm_cells["fold_index"] == fold)]
        seq_subset = seq_cells.loc[(seq_cells["seed"] == seed) & (seq_cells["fold_index"] == fold)]
        if gbm_subset.empty:
            seen_no_gbm = True
            continue
        if seq_subset.empty:
            seen_no_seq = True
            continue

        gbm_only = _join_predictions(cells=gbm_subset, output_root=output_root)
        # GBM+seq is the inner join of the GBM rosters + seq cells.
        combined = pd.concat([gbm_subset, seq_subset], ignore_index=True)
        gbm_plus_seq = _join_predictions(cells=combined, output_root=output_root)
        if gbm_only is None or gbm_plus_seq is None:
            continue
        idx_a, y_true_a, value_a = gbm_only
        idx_b, y_true_b, value_b = gbm_plus_seq

        # Restrict both ensembles to the panel-row intersection so
        # `loss_a`, `loss_b`, and the oracle sit on the SAME rows.
        # The seq adapter strips below-floor rows (A9.1) but the
        # GBM adapters do not, so `len(idx_b) <= len(idx_a)` in
        # production; without this intersection, `delta_loss`
        # would compare means over different populations and the
        # oracle would crash on row-misaligned arrays.
        common, pos_a, pos_b = np.intersect1d(idx_a, idx_b, return_indices=True)
        if common.size == 0:
            continue
        y_true_common = y_true_b[pos_b]
        if not np.array_equal(y_true_a[pos_a], y_true_common):
            logger.warning(
                "compute_per_cell_lift_deltas: y_true disagrees on the "
                "gbm/gbm+seq intersection for dataset=%s seed=%d fold=%d; "
                "skipping pair",
                dataset_name,
                seed,
                fold,
            )
            continue
        value_a_common = value_a[pos_a]
        value_b_common = value_b[pos_b]

        # Per-task ensemble value type.
        if task_type in {"binary", "multiclass"}:
            proba_a = value_a_common if value_a_common.ndim == 2 else None
            proba_b = value_b_common if value_b_common.ndim == 2 else None
            pred_a = None
            pred_b = None
        else:
            proba_a = None
            proba_b = None
            pred_a = value_a_common if value_a_common.ndim == 1 else None
            pred_b = value_b_common if value_b_common.ndim == 1 else None

        loss_a = _per_cell_primary_loss(
            task_type=task_type, y_true=y_true_common, proba=proba_a, y_pred=pred_a
        )
        loss_b = _per_cell_primary_loss(
            task_type=task_type, y_true=y_true_common, proba=proba_b, y_pred=pred_b
        )
        if loss_a is None or loss_b is None:
            continue
        oracle = _per_sample_oracle_loss(
            task_type=task_type,
            y_true=y_true_common,
            proba_a=proba_a,
            proba_b=proba_b,
            pred_a=pred_a,
            pred_b=pred_b,
        )
        records.append(
            PerCellLiftDelta(
                seed=int(seed),
                fold_index=int(fold),
                loss_gbm=loss_a,
                loss_gbm_plus_seq=loss_b,
                delta_loss=loss_a - loss_b,
                oracle_loss=oracle,
            )
        )

    return ComputePerCellLiftDeltasResult(
        cells=tuple(records),
        seen_no_gbm=seen_no_gbm,
        seen_no_seq=seen_no_seq,
        selector=selector,
    )


def _per_dataset_lift(
    *,
    dataset_name: str,
    task_type: str,
    seed_fold_pairs: list[tuple[int, int]],
    gbm_cells: pd.DataFrame,
    seq_cells: pd.DataFrame,
    output_root: Path,
) -> PerDatasetLift:
    """Per-dataset Δloss aggregation.

    Thin aggregation over `compute_per_cell_lift_deltas`: takes
    the per-(seed, fold) records, drops the partial-loss cells
    (the helper already returns only finite-loss records), and
    computes the per-dataset mean Δloss + std + oracle bounds.
    """
    computed = compute_per_cell_lift_deltas(
        dataset_name=dataset_name,
        task_type=task_type,
        seed_fold_pairs=seed_fold_pairs,
        gbm_cells=gbm_cells,
        seq_cells=seq_cells,
        output_root=output_root,
    )
    selector = computed.selector

    if not computed.cells:
        return PerDatasetLift(
            dataset_name=dataset_name,
            task_type=task_type,
            primary_loss_column=selector,
            n_cells_paired=0,
            no_gbm_predictions=computed.seen_no_gbm,
            no_seq_predictions=computed.seen_no_seq,
        )

    losses_gbm = [c.loss_gbm for c in computed.cells]
    losses_gbm_seq = [c.loss_gbm_plus_seq for c in computed.cells]
    losses_oracle = [c.oracle_loss for c in computed.cells if c.oracle_loss is not None]

    delta = np.array(losses_gbm) - np.array(losses_gbm_seq)
    delta_std: float | None = float(delta.std(ddof=1)) if delta.size > 1 else None
    oracle_mean = float(np.mean(losses_oracle)) if losses_oracle else None
    oracle_delta_mean = (
        float(np.mean(losses_gbm) - np.array(losses_oracle).mean()) if losses_oracle else None
    )

    return PerDatasetLift(
        dataset_name=dataset_name,
        task_type=task_type,
        primary_loss_column=selector,
        n_cells_paired=len(losses_gbm),
        loss_gbm_only_mean=float(np.mean(losses_gbm)),
        loss_gbm_plus_seq_mean=float(np.mean(losses_gbm_seq)),
        delta_loss_mean=float(np.mean(delta)),
        delta_loss_std=delta_std,
        oracle_loss_mean=oracle_mean,
        oracle_delta_loss_mean=oracle_delta_mean,
    )


def _wilcoxon_across_datasets(rows: list[PerDatasetLift], *, family_size: int) -> WilcoxonResult:
    """Wilcoxon signed-rank on per-dataset Δloss.

    Holm correction with `family_size=1` returns the raw p-value;
    a future multi-pair test family extends the size.
    """
    # Filter `None` AND NaN: an upstream cell loss can produce NaN
    # when an adapter writes NaN-tinged predictions (e.g., a buggy
    # comparator). NaN flowing into `scipy.stats.wilcoxon` yields a
    # NaN p-value that crashes the Holm correction sort downstream.
    deltas = [
        r.delta_loss_mean
        for r in rows
        if r.delta_loss_mean is not None and not np.isnan(r.delta_loss_mean)
    ]
    arr = np.asarray(deltas, dtype=float)
    # Wilcoxon signed-rank needs >= 2 paired observations AFTER
    # zero-discarding for any inferential content. scipy's
    # `zero_method="wilcox"` drops zeros, so a vector like
    # `[0.0, 1.0]` would otherwise bypass this guard and operate on
    # n=1 effective. Guard on `count_nonzero` so the semantic floor
    # matches scipy's internal behavior.
    if np.count_nonzero(arr) < 2:
        # All-zero or single-non-zero: surface the neutral all-zero
        # path when there are at least two zero entries (matches
        # the friedman_holm convention); otherwise route to the
        # informational skip path.
        if arr.size >= 2 and np.count_nonzero(arr) == 0:
            return WilcoxonResult(
                statistic=0.0,
                p_value=1.0,
                holm_adjusted_p_value=1.0,
                n_datasets=int(arr.size),
                family_size=family_size,
            )
        return WilcoxonResult(
            statistic=None,
            p_value=None,
            holm_adjusted_p_value=None,
            n_datasets=int(arr.size),
            family_size=family_size,
        )
    res = stats.wilcoxon(arr, zero_method="wilcox", alternative="two-sided")
    raw_p = float(res.pvalue)  # pyright: ignore[reportAttributeAccessIssue]
    statistic = float(res.statistic)  # pyright: ignore[reportAttributeAccessIssue]
    # Holm correction over the configured pair family.
    if family_size <= 1:
        adjusted = raw_p
    else:
        adjusted_list = holm_correction([raw_p] + [1.0] * (family_size - 1))
        adjusted = adjusted_list[0]
    return WilcoxonResult(
        statistic=statistic,
        p_value=raw_p,
        holm_adjusted_p_value=adjusted,
        n_datasets=int(arr.size),
        family_size=family_size,
    )


def run_ensemble_lift(
    config: BenchmarkConfig,
    *,
    output_root: Path,
    env: RunEnvironment,
) -> EnsembleLiftExperimentResult:
    """Read the B5 manifest under `output_root`, build the
    per-(dataset, seed, fold) GBM-only and GBM+seq ensembles
    from the cached predictions, compute the per-dataset Δloss +
    oracle, and run the Wilcoxon signed-rank across datasets.

    The seq + baseline family pair is fixed at v1 (seq_sklearn vs
    gbm); a cross-family-set lift driver is a B11-followup.

    Returns a structured summary; the report-rendering layer
    (`benchmarks/report/ensemble_lift.py`) turns the summary into
    the Markdown deliverable.

    Raises:
        ValueError: no `ensemble_lift` experiment in the config,
            or the B5 manifest under `output_root` is empty.
    """
    seq_family = DEFAULT_SEQ_FAMILY
    baseline_family = DEFAULT_BASELINE_FAMILY
    _assert_ensemble_lift_configured(config.experiments)
    manifest = load_run(output_root)
    if manifest.empty:
        raise ValueError(
            f"run_ensemble_lift: B5 manifest at {output_root} is "
            f"empty; run `--experiment=raw_loss` first"
        )

    started_at = time.perf_counter()
    families = model_families(manifest)
    gbm_cells = build_cells_table(manifest, families=families, target_family=baseline_family)
    seq_cells = build_cells_table(manifest, families=families, target_family=seq_family)
    logger.info(
        "run_ensemble_lift: baseline=%s cells=%d, seq=%s cells=%d",
        baseline_family,
        len(gbm_cells),
        seq_family,
        len(seq_cells),
    )

    per_dataset: list[PerDatasetLift] = []
    dataset_names = sorted({str(name) for name in manifest["dataset_name"].dropna().unique()})
    for dataset_name in dataset_names:
        ds_manifest = manifest.loc[manifest["dataset_name"] == dataset_name]
        task_type_values = ds_manifest["task_type"].dropna().unique()
        if len(task_type_values) != 1:
            logger.warning(
                "run_ensemble_lift: dataset %s has %d distinct task_types %s; "
                "skipping (config drift across runs)",
                dataset_name,
                len(task_type_values),
                sorted(task_type_values),
            )
            continue
        task_type = str(task_type_values[0])
        seed_fold_pairs: list[tuple[int, int]] = sorted(
            {
                (int(s), int(f))
                for s, f in zip(
                    ds_manifest["seed"].to_numpy(),
                    ds_manifest["fold_index"].to_numpy(),
                    strict=True,
                )
                if pd.notna(s) and pd.notna(f)
            }
        )
        gbm_ds = gbm_cells.loc[gbm_cells["dataset_name"] == dataset_name]
        seq_ds = seq_cells.loc[seq_cells["dataset_name"] == dataset_name]
        per_dataset.append(
            _per_dataset_lift(
                dataset_name=dataset_name,
                task_type=task_type,
                seed_fold_pairs=seed_fold_pairs,
                gbm_cells=gbm_ds,
                seq_cells=seq_ds,
                output_root=output_root,
            )
        )

    wilcoxon = _wilcoxon_across_datasets(per_dataset, family_size=1)
    elapsed = time.perf_counter() - started_at
    logger.info(
        "run_ensemble_lift: %d datasets aggregated in %.2fs; wilcoxon p=%s",
        len(per_dataset),
        elapsed,
        wilcoxon.p_value,
    )
    return EnsembleLiftExperimentResult(
        run_id=env.run_id,
        seq_family=seq_family,
        baseline_family=baseline_family,
        rows=tuple(per_dataset),
        wilcoxon=wilcoxon,
    )
