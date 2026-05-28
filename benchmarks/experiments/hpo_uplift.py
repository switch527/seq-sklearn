"""Phase B8 HPO-uplift experiment driver (B6.4).

Quantifies the gain from full hyperparameter tuning over the
default config per design B6.4. Per cell, runs an Optuna study on
an inner train/validation split (B6.4.2: the test fold is untouched
until the final fit), then refits the best config on the full train
fold and evaluates on the test fold. Writes one ResultRow with
`variant="tuned"` per cell.

Resumability (B7.2): tuned cells use the same atomic shard-then-
sentinel contract as B5; the cell key includes `variant="tuned"`
so default and tuned cells coexist in the same `results/` directory
without colliding.

Default arm: reuses the B5 manifest (variant="default" rows). The
report module (`benchmarks/report/hpo_uplift.py`) joins default and
tuned rows per (dataset_name, model_name, seed, fold) to compute the
Δ statistic. The driver does NOT re-run the default arm.

HPO budget (B7.4): the (n_trials, timeout_seconds) tier is derived
from the run's profile (smoke: 0 trials, standard: 25 trials,
full: 100 trials) and can be overridden per experiment via
`ExperimentSpec.n_trials` / `.timeout_seconds`. The budget is
written on every tuned row so uplift can only be compared within a
tier (B7.4 contract).

Search-space parity (B6.4.0): every tuned row records
`hpo_search_space_size` from the registered `HPOSpace`. The report
renders the size next to the Δ so cross-family comparisons surface
their search-breadth caveat. v1 explicitly does not normalize size
across families (design B6.4.0).

B8-followup deferrals (carried into B9 / later):

- GBM and TSC search spaces ship with their adapter families (per
  `benchmarks/adapters/__init__.py`). Cells whose model family has
  no HPO space registered today skip cleanly with
  `skipped_reason="hpo_family_not_registered"`.
- Nemenyi critical-difference diagram (design B7.5): the Friedman
  + Holm matrix is rendered as a table at B8; the diagram is the
  B9 visualization pass.
- Predictions persistence for tuned cells: B6.2.1 writes a
  per-cell predictions shard for B5. B8 does NOT write the
  predictions shard for the tuned arm to avoid the ensemble report
  inadvertently mixing default + tuned variants into one pairwise
  cell; B9 wires variant-aware predictions if needed.
"""

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.exceptions import TrialPruned
from pydantic import BaseModel, ConfigDict

from benchmarks.adapters._base import (
    OptionalDependencyMissingError,
    ProbaUnsupportedError,
    QuantilesUnsupportedError,
    SeqSklearnAdapter,
)
from benchmarks.config import BenchmarkConfig, DatasetSpec, ExperimentSpec, TaskType
from benchmarks.datasets._base import PanelDataset
from benchmarks.experiments.raw_loss import (
    RunEnvironment,
    _fit_with_seed,  # pyright: ignore[reportPrivateUsage]
    _iter_folds,  # pyright: ignore[reportPrivateUsage]
    _maybe_predict_proba,  # pyright: ignore[reportPrivateUsage]
    _resolve_labels,  # pyright: ignore[reportPrivateUsage]
    _resolve_names,  # pyright: ignore[reportPrivateUsage]
    _strip_below_floor_rows,  # pyright: ignore[reportPrivateUsage]
    _take_panel_rows,  # pyright: ignore[reportPrivateUsage]
    _utc_now,  # pyright: ignore[reportPrivateUsage]
)
from benchmarks.hpo._base import HPOFamilyNotRegisteredError, get_hpo_space
from benchmarks.manifest import (
    ResultRow,
    cell_key,
    is_cell_complete,
    list_completed_keys,
    metrics_to_row_fields,
    write_cell,
)
from benchmarks.metrics import compute_all
from benchmarks.metrics.resource import FitResource, measure_fit
from benchmarks.protocol import (
    fingerprint_folds,
    make_splitter,
    resolve_lookback,
)
from benchmarks.registry import (
    get_dataset,
    get_loader,
    get_model,
    instantiate_adapter,
    list_datasets,
    list_models,
)
from seq_sklearn import NotFittedError

logger = logging.getLogger(__name__)


_DEFAULT_N_SPLITS = 5
_HPO_VARIANT = "tuned"
_INNER_VAL_FRACTION = 0.2  # tail fraction of train fold held out for trial scoring

# Per-profile HPO budget defaults (B7.1, B7.4). `(n_trials, timeout_seconds)`.
# Smoke is 0 trials by design (no HPO); a config that overrides via
# `ExperimentSpec.n_trials` still runs.
_HPO_BUDGETS_BY_PROFILE: dict[str, tuple[int, float]] = {
    "smoke": (0, 0.0),
    "standard": (25, 1800.0),
    "full": (100, 21600.0),
}
_HPO_TIER_BY_PROFILE: dict[str, str] = {
    "smoke": "none",
    "standard": "light",
    "full": "full",
}

# Skipped-reason prefixes.
_SKIP_REASON_TASK_TYPE_MISMATCH = "task_type_mismatch"
_SKIP_REASON_PROBA_UNAVAILABLE = "probabilities_required_for_classification"
_SKIP_REASON_PROBA_RUNTIME_UNAVAILABLE = "probabilities_required_but_runtime_unavailable"
_SKIP_REASON_QUANTILE_FOLLOWUP = "regression_quantile_b5_followup"
_SKIP_REASON_HPO_FAMILY_NOT_REGISTERED = "hpo_family_not_registered"
_SKIP_REASON_HPO_BUDGET_ZERO = "hpo_budget_zero_for_profile"
_SKIP_REASON_HPO_ALL_TRIALS_PRUNED = "hpo_all_trials_pruned"
_SKIP_REASON_ADAPTER_ERROR = "adapter_error"
_SKIP_REASON_OPTIONAL_DEP_MISSING = "optional_dep_missing"


class HPOUpliftExperimentResult(BaseModel):
    """Summary returned by `run_hpo_uplift`.

    Counts the tuned-arm cells the driver touched (the default arm
    is the B5 manifest's job); the full Δ table lives in the report
    layer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    cells_attempted: int  # tuned arm: best config refit + evaluated
    cells_skipped_task_mismatch: int
    cells_skipped_proba_unavailable: int
    cells_skipped_proba_runtime_unavailable: int
    cells_skipped_quantile_followup: int
    cells_skipped_hpo_family_not_registered: int
    cells_skipped_hpo_budget_zero: int
    cells_skipped_hpo_all_trials_pruned: int
    cells_skipped_adapter_error: int
    cells_skipped_optional_dep_missing: int  # B12: adapter raised OptionalDependencyMissingError
    cells_already_complete: int


def _resolve_hpo_seeds(experiments: Sequence[ExperimentSpec]) -> tuple[int, ...]:
    """Pull the deduplicated, sorted seed tuple from the
    `hpo_uplift` experiment spec(s).

    Raises:
        ValueError: no `hpo_uplift` experiment is configured.
    """
    seeds: set[int] = set()
    for spec in experiments:
        if spec.kind == "hpo_uplift":
            seeds.update(spec.seeds)
    if not seeds:
        raise ValueError(
            "run_hpo_uplift: BenchmarkConfig declares no hpo_uplift "
            "experiment; add ExperimentSpec(kind='hpo_uplift') to the "
            "config"
        )
    return tuple(sorted(seeds))


def _resolve_hpo_budget(
    experiments: Sequence[ExperimentSpec], profile: str
) -> tuple[int, float, str]:
    """Return `(n_trials, timeout_seconds, hpo_tier)` for the run.

    Profile-tier defaults from `_HPO_BUDGETS_BY_PROFILE`. The first
    `ExperimentSpec(kind="hpo_uplift")` carrying a non-None
    `n_trials` / `timeout_seconds` overrides the corresponding
    default. When the overrides yield a non-default tier, `hpo_tier`
    is the override sentinel `"custom"` so the report can tell
    overridden runs apart.
    """
    n_trials, timeout_seconds = _HPO_BUDGETS_BY_PROFILE.get(profile, (0, 0.0))
    tier = _HPO_TIER_BY_PROFILE.get(profile, "none")
    overridden = False
    for spec in experiments:
        if spec.kind != "hpo_uplift":
            continue
        if spec.n_trials is not None:
            n_trials = int(spec.n_trials)
            overridden = True
        if spec.timeout_seconds is not None:
            timeout_seconds = float(spec.timeout_seconds)
            overridden = True
        break
    if overridden:
        tier = "custom"
    return n_trials, timeout_seconds, tier


def _primary_loss_from_metrics(metrics_dump: dict[str, Any], task_type: TaskType) -> float:
    """Return the primary loss for a metric dump.

    Per B6.1: log_loss for binary + multiclass; rmse for
    regression_point; mean of pinball values for regression_quantile.

    Raises:
        ValueError: the metric dump does not carry the primary loss
            for `task_type` (e.g. a classifier dump given to a
            regression task).
    """
    if task_type in {"binary", "multiclass"}:
        value = metrics_dump.get("log_loss")
        if value is None:
            raise ValueError(
                f"_primary_loss_from_metrics: task_type={task_type} "
                f"requires `log_loss`; got {metrics_dump.keys()}"
            )
        return float(value)
    if task_type == "regression_point":
        value = metrics_dump.get("rmse")
        if value is None:
            raise ValueError(
                "_primary_loss_from_metrics: task_type=regression_point "
                f"requires `rmse`; got {metrics_dump.keys()}"
            )
        return float(value)
    if task_type == "regression_quantile":
        # Mean across the pinball levels. The metric layer returns a
        # `pinball: dict[str, float]`; averaging matches B6.1's
        # treatment of quantile loss as the primary rank key.
        pinball = metrics_dump.get("pinball")
        if not pinball:
            raise ValueError(
                "_primary_loss_from_metrics: task_type=regression_quantile "
                "requires `pinball`; got empty/None"
            )
        return float(np.mean(list(pinball.values())))
    raise ValueError(f"_primary_loss_from_metrics: unsupported task_type={task_type!r}")


def _inner_train_val_split(train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a fold's train indices into inner_train + val by tail
    fraction.

    Time-ordered split: the LAST `_INNER_VAL_FRACTION` of `train_idx`
    becomes the validation set. The harness's outer splitter
    (`benchmarks.protocol.make_splitter`) is already time-ordered;
    splitting the train tail keeps the val/test ordering consistent.

    Returns `(inner_train_idx, val_idx)` as numpy arrays. A train
    fold with fewer than 2 rows raises `ValueError` (an HPO trial
    on 0 or 1 rows is not meaningful).
    """
    n_train = train_idx.shape[0]
    if n_train < 2:
        raise ValueError(
            f"_inner_train_val_split: train fold has {n_train} rows; "
            f"need >= 2 to carve an inner val split"
        )
    n_val = max(1, round(n_train * _INNER_VAL_FRACTION))
    n_val = min(n_val, n_train - 1)  # leave at least 1 row in inner_train
    return train_idx[:-n_val], train_idx[-n_val:]


def _evaluate_on(
    *,
    adapter: SeqSklearnAdapter,
    spec: DatasetSpec,
    panel: pd.DataFrame,
    y: np.ndarray,
    labels: np.ndarray | None,
) -> dict[str, Any]:
    """Fit-time inference + metric compute on a single (panel, y).

    Returns the `compute_all` model_dump. Caller wraps the call in
    the adapter-failure-mode try/except. `y_proba` is None for
    regressor adapters and the metric layer's None-proba branches
    are exercised on the inner val + final test pieces alike.
    """
    y_pred = adapter.predict(panel)
    y_proba = _maybe_predict_proba(adapter, panel)
    y_true_clean, y_pred_clean, y_proba_clean, _, _ = _strip_below_floor_rows(y, y_pred, y_proba)
    if y_proba is None and spec.task_type in {"binary", "multiclass"}:
        raise ProbaUnsupportedError(
            f"_evaluate_on: model={spec.name} runtime predict_proba "
            "unavailable on a classification task"
        )
    metrics = compute_all(
        task_type=spec.task_type,
        y_true=y_true_clean,
        y_pred=y_pred_clean,
        y_proba=y_proba_clean,
        pos_label=spec.positive_label,
        labels=labels,
    )
    return metrics.model_dump()


def _build_tuned_row(
    *,
    env: RunEnvironment,
    spec: DatasetSpec,
    model_name: str,
    seed: int,
    fold_index: int,
    n_folds: int,
    split_fingerprint: str,
    l_resolved: int,
    started_at_utc: str,
    hpo_tier: str,
    hpo_n_trials_completed: int | None,
    hpo_n_trials_pruned: int | None,
    hpo_search_space_size: int | None,
    hpo_timeout_seconds: float | None,
    hpo_best_trial_number: int | None,
    hpo_time_to_best_seconds: float | None,
    hpo_best_val_loss: float | None,
    skipped_reason: str | None,
    metrics_dump: dict[str, Any] | None,
    fit_resource: FitResource | None,
    n_below_floor_dropped: int | None = None,
) -> ResultRow:
    """Assemble a tuned-variant `ResultRow` from the per-cell pieces.

    `variant="tuned"` is pinned here; the cell key the manifest
    persists is `(dataset, model, seed, "tuned", fold)`.
    """
    fields: dict[str, Any] = {
        "library_git_sha": env.library_git_sha,
        "run_id": env.run_id,
        "started_at_utc": started_at_utc,
        "dataset_name": spec.name,
        "model_name": model_name,
        "seed": seed,
        "variant": _HPO_VARIANT,
        "task_type": spec.task_type,
        "fold_index": fold_index,
        "n_folds": n_folds,
        "split_fingerprint": split_fingerprint,
        "l_resolved": l_resolved,
        "hardware_tier": env.hardware_tier,
        "profile": env.profile,
        "hpo_tier": hpo_tier,
        "hpo_n_trials_completed": hpo_n_trials_completed,
        "hpo_n_trials_pruned": hpo_n_trials_pruned,
        "hpo_search_space_size": hpo_search_space_size,
        "hpo_timeout_seconds": hpo_timeout_seconds,
        "hpo_best_trial_number": hpo_best_trial_number,
        "hpo_time_to_best_seconds": hpo_time_to_best_seconds,
        "hpo_best_val_loss": hpo_best_val_loss,
        "skipped_reason": skipped_reason,
        "n_below_floor_dropped": n_below_floor_dropped,
    }
    if metrics_dump is not None:
        fields.update(metrics_to_row_fields(metrics_dump))
    if fit_resource is not None:
        fields["wall_seconds"] = fit_resource.wall_seconds
        fields["peak_rss_bytes"] = fit_resource.peak_rss_bytes
        fields["peak_cuda_bytes"] = fit_resource.peak_cuda_bytes
    return ResultRow(**fields)


def _run_hpo_study(
    *,
    spec: DatasetSpec,
    model_name: str,
    hpo_sampler: Any,
    panel_dataset: PanelDataset,
    inner_train_idx: np.ndarray,
    val_idx: np.ndarray,
    labels: np.ndarray | None,
    n_trials: int,
    timeout_seconds: float,
    seed: int,
) -> tuple[optuna.Study, int, int, int | None, float | None, float | None, dict[str, Any] | None]:
    """Drive an Optuna study against the inner train/val split.

    Returns:
        `(study, n_completed, n_pruned, best_trial_number,
        time_to_best, best_val_loss, best_hyperparameters)`.
        The `n_completed` count includes only trials that finished
        without pruning; `n_pruned` counts pruned trials. The
        `best_trial_number` is `None` when no trial completed.
    """
    inner_train_panel, inner_train_y = _take_panel_rows(
        panel_dataset.panel, panel_dataset.y, inner_train_idx
    )
    val_panel, val_y = _take_panel_rows(panel_dataset.panel, panel_dataset.y, val_idx)
    model_spec = get_model(model_name)

    # Optuna's TPE sampler is reproducible when seeded; pin to the
    # cell's seed so a (dataset, model, seed, fold) study is
    # deterministic across runs (B8.1 reproducibility).
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    best_val_loss: float | None = None
    best_trial_number: int | None = None
    best_hyperparameters: dict[str, Any] | None = None
    n_completed = 0

    def _objective(trial: optuna.trial.Trial) -> float:
        nonlocal best_val_loss, best_trial_number, best_hyperparameters, n_completed
        # The sampler is expected to delegate F8 wrapping to the
        # library (`ConfigError` -> `TrialPruned` via
        # `optuna_trial_guard`). A `ValueError` here is the legacy
        # path some samplers raise on illegal cells; convert it to
        # a pruned trial so the study continues, while letting
        # anything else (a real bug) fail fast.
        try:
            hyperparameters = hpo_sampler(trial, model_spec, spec)
        except ValueError as exc:
            raise TrialPruned(str(exc)) from exc

        # Build a fresh adapter per trial; the trial's
        # hyperparameters override the adapter's defaults via the
        # `hyperparameters=` channel.
        trial_adapter = instantiate_adapter(model_name, spec=spec, hyperparameters=hyperparameters)
        # Numerical and adapter-side failures prune; structural
        # bugs propagate (matches B5's adapter-error contract).
        try:
            _fit_with_seed(trial_adapter, inner_train_panel, inner_train_y, seed)
            metrics_dump = _evaluate_on(
                adapter=trial_adapter,
                spec=spec,
                panel=val_panel,
                y=val_y,
                labels=labels,
            )
        except (
            RuntimeError,
            MemoryError,
            NotFittedError,
            ProbaUnsupportedError,
            QuantilesUnsupportedError,
        ) as exc:
            raise TrialPruned(f"{type(exc).__name__}: {exc}") from exc

        val_loss = _primary_loss_from_metrics(metrics_dump, spec.task_type)
        n_completed += 1
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_trial_number = trial.number
            best_hyperparameters = dict(hyperparameters)
        return val_loss

    if n_trials > 0:
        study.optimize(
            _objective,
            n_trials=n_trials,
            timeout=timeout_seconds if timeout_seconds > 0 else None,
            catch=(),
            show_progress_bar=False,
        )
    # `study.trials` includes both COMPLETE and PRUNED trials; the
    # counter inside `_objective` only increments on COMPLETE. Count
    # the pruned trials here.
    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)

    time_to_best_final = _time_to_best_seconds(study, best_trial_number)

    return (
        study,
        n_completed,
        n_pruned,
        best_trial_number,
        time_to_best_final,
        best_val_loss,
        best_hyperparameters,
    )


def _time_to_best_seconds(study: optuna.Study, best_trial_number: int | None) -> float | None:
    """Compute wall-clock seconds from study start to the best
    trial's completion (design B6.4.1 "time-to-best-trial").

    Uses Optuna's `datetime_start` on the first trial as the
    study-start anchor and `datetime_complete` on the best trial.
    Returns None when either timestamp is absent or no trial
    completed (`best_trial_number is None`). Looks up the best
    trial by trial-number (not by list position) so a future
    parallel/distributed study cannot scramble the lookup
    (code-I1).
    """
    if best_trial_number is None or not study.trials:
        return None
    first_trial_start = study.trials[0].datetime_start
    if first_trial_start is None:
        return None
    best_trial = next((t for t in study.trials if t.number == best_trial_number), None)
    if best_trial is None or best_trial.datetime_complete is None:
        return None
    return (best_trial.datetime_complete - first_trial_start).total_seconds()


def _run_one_hpo_cell(
    *,
    env: RunEnvironment,
    panel_dataset: PanelDataset,
    spec: DatasetSpec,
    model_name: str,
    seed: int,
    fold_index: int,
    n_folds: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_fingerprint: str,
    l_resolved: int,
    labels: np.ndarray | None,
    hpo_tier: str,
    n_trials: int,
    timeout_seconds: float,
) -> ResultRow:
    """Run one (dataset, model, seed, fold) tuned-arm cell.

    Mirrors B5's `_run_one_cell` shape: surfaces typed
    `skipped_reason` rows for the known failure modes and lets
    harness-side bugs propagate.
    """
    started_at_utc = _utc_now()

    # B5-followup carryover: regression_quantile cells need the
    # quantile-level protocol surface to compute pinball metrics.
    if spec.task_type == "regression_quantile":
        return _build_tuned_row(
            env=env,
            spec=spec,
            model_name=model_name,
            seed=seed,
            fold_index=fold_index,
            n_folds=n_folds,
            split_fingerprint=split_fingerprint,
            l_resolved=l_resolved,
            started_at_utc=started_at_utc,
            hpo_tier=hpo_tier,
            hpo_n_trials_completed=None,
            hpo_n_trials_pruned=None,
            hpo_search_space_size=None,
            hpo_timeout_seconds=None,
            hpo_best_trial_number=None,
            hpo_time_to_best_seconds=None,
            hpo_best_val_loss=None,
            skipped_reason=_SKIP_REASON_QUANTILE_FOLLOWUP,
            metrics_dump=None,
            fit_resource=None,
        )

    model_spec = get_model(model_name)
    try:
        hpo_space, hpo_sampler = get_hpo_space(model_spec.family)
    except HPOFamilyNotRegisteredError:
        return _build_tuned_row(
            env=env,
            spec=spec,
            model_name=model_name,
            seed=seed,
            fold_index=fold_index,
            n_folds=n_folds,
            split_fingerprint=split_fingerprint,
            l_resolved=l_resolved,
            started_at_utc=started_at_utc,
            hpo_tier=hpo_tier,
            hpo_n_trials_completed=None,
            hpo_n_trials_pruned=None,
            hpo_search_space_size=None,
            hpo_timeout_seconds=None,
            hpo_best_trial_number=None,
            hpo_time_to_best_seconds=None,
            hpo_best_val_loss=None,
            skipped_reason=(
                f"{_SKIP_REASON_HPO_FAMILY_NOT_REGISTERED}: model_spec.family={model_spec.family!r}"
            ),
            metrics_dump=None,
            fit_resource=None,
        )

    if n_trials <= 0:
        return _build_tuned_row(
            env=env,
            spec=spec,
            model_name=model_name,
            seed=seed,
            fold_index=fold_index,
            n_folds=n_folds,
            split_fingerprint=split_fingerprint,
            l_resolved=l_resolved,
            started_at_utc=started_at_utc,
            hpo_tier=hpo_tier,
            hpo_n_trials_completed=0,
            hpo_n_trials_pruned=0,
            hpo_search_space_size=hpo_space.search_space_size,
            hpo_timeout_seconds=timeout_seconds,
            hpo_best_trial_number=None,
            hpo_time_to_best_seconds=None,
            hpo_best_val_loss=None,
            skipped_reason=(
                f"{_SKIP_REASON_HPO_BUDGET_ZERO}: profile={env.profile}; "
                f"override via ExperimentSpec.n_trials"
            ),
            metrics_dump=None,
            fit_resource=None,
        )

    try:
        inner_train_idx, val_idx = _inner_train_val_split(train_idx)
        (
            _study,
            n_completed,
            n_pruned,
            best_trial_number,
            time_to_best,
            best_val_loss,
            best_hyperparameters,
        ) = _run_hpo_study(
            spec=spec,
            model_name=model_name,
            hpo_sampler=hpo_sampler,
            panel_dataset=panel_dataset,
            inner_train_idx=inner_train_idx,
            val_idx=val_idx,
            labels=labels,
            n_trials=n_trials,
            timeout_seconds=timeout_seconds,
            seed=seed,
        )
    except OptionalDependencyMissingError as exc:
        # B12: trial fit raised OptionalDependencyMissingError. The
        # outer cell-level skip keeps the HPO budget from burning
        # on a missing-dep error that would never resolve. Caught
        # BEFORE the generic adapter-error tuple so it wins on
        # ordering.
        logger.info(
            "run_hpo_uplift: cell %s/%s seed=%d fold=%d skipped: optional dependency %r missing",
            spec.name,
            model_name,
            seed,
            fold_index,
            exc.package_name,
        )
        return _build_tuned_row(
            env=env,
            spec=spec,
            model_name=model_name,
            seed=seed,
            fold_index=fold_index,
            n_folds=n_folds,
            split_fingerprint=split_fingerprint,
            l_resolved=l_resolved,
            started_at_utc=started_at_utc,
            hpo_tier=hpo_tier,
            hpo_n_trials_completed=None,
            hpo_n_trials_pruned=None,
            hpo_search_space_size=hpo_space.search_space_size,
            hpo_timeout_seconds=timeout_seconds,
            hpo_best_trial_number=None,
            hpo_time_to_best_seconds=None,
            hpo_best_val_loss=None,
            skipped_reason=f"{_SKIP_REASON_OPTIONAL_DEP_MISSING}: {exc.package_name}",
            metrics_dump=None,
            fit_resource=None,
        )
    except (
        RuntimeError,
        MemoryError,
        NotFittedError,
        ProbaUnsupportedError,
        QuantilesUnsupportedError,
    ) as exc:
        logger.warning(
            "run_hpo_uplift: study setup crashed on %s/%s seed=%d fold=%d: %s: %s",
            spec.name,
            model_name,
            seed,
            fold_index,
            type(exc).__name__,
            exc,
        )
        return _build_tuned_row(
            env=env,
            spec=spec,
            model_name=model_name,
            seed=seed,
            fold_index=fold_index,
            n_folds=n_folds,
            split_fingerprint=split_fingerprint,
            l_resolved=l_resolved,
            started_at_utc=started_at_utc,
            hpo_tier=hpo_tier,
            hpo_n_trials_completed=None,
            hpo_n_trials_pruned=None,
            hpo_search_space_size=hpo_space.search_space_size,
            hpo_timeout_seconds=timeout_seconds,
            hpo_best_trial_number=None,
            hpo_time_to_best_seconds=None,
            hpo_best_val_loss=None,
            skipped_reason=f"{_SKIP_REASON_ADAPTER_ERROR}: {type(exc).__name__}: {exc}",
            metrics_dump=None,
            fit_resource=None,
        )

    if best_hyperparameters is None:
        return _build_tuned_row(
            env=env,
            spec=spec,
            model_name=model_name,
            seed=seed,
            fold_index=fold_index,
            n_folds=n_folds,
            split_fingerprint=split_fingerprint,
            l_resolved=l_resolved,
            started_at_utc=started_at_utc,
            hpo_tier=hpo_tier,
            hpo_n_trials_completed=n_completed,
            hpo_n_trials_pruned=n_pruned,
            hpo_search_space_size=hpo_space.search_space_size,
            hpo_timeout_seconds=timeout_seconds,
            hpo_best_trial_number=None,
            hpo_time_to_best_seconds=None,
            hpo_best_val_loss=None,
            skipped_reason=(
                f"{_SKIP_REASON_HPO_ALL_TRIALS_PRUNED}: completed={n_completed} pruned={n_pruned}"
            ),
            metrics_dump=None,
            fit_resource=None,
        )

    # Refit on the full train fold with the best config; evaluate on
    # the untouched test fold (B6.4.2 contract).
    train_panel, train_y = _take_panel_rows(panel_dataset.panel, panel_dataset.y, train_idx)
    test_panel, test_y = _take_panel_rows(panel_dataset.panel, panel_dataset.y, test_idx)
    try:
        final_adapter = instantiate_adapter(
            model_name, spec=spec, hyperparameters=best_hyperparameters
        )
        fit_resource = measure_fit(
            lambda: _fit_with_seed(final_adapter, train_panel, train_y, seed),
            cuda_available=False,
        )
        y_pred = final_adapter.predict(test_panel)
        y_proba = _maybe_predict_proba(final_adapter, test_panel)
        if y_proba is None and spec.task_type in {"binary", "multiclass"}:
            return _build_tuned_row(
                env=env,
                spec=spec,
                model_name=model_name,
                seed=seed,
                fold_index=fold_index,
                n_folds=n_folds,
                split_fingerprint=split_fingerprint,
                l_resolved=l_resolved,
                started_at_utc=started_at_utc,
                hpo_tier=hpo_tier,
                hpo_n_trials_completed=n_completed,
                hpo_n_trials_pruned=n_pruned,
                hpo_search_space_size=hpo_space.search_space_size,
                hpo_timeout_seconds=timeout_seconds,
                hpo_best_trial_number=best_trial_number,
                hpo_time_to_best_seconds=time_to_best,
                hpo_best_val_loss=best_val_loss,
                skipped_reason=(
                    f"{_SKIP_REASON_PROBA_RUNTIME_UNAVAILABLE}: best config "
                    f"refit raised ProbaUnsupportedError at runtime"
                ),
                metrics_dump=None,
                fit_resource=fit_resource,
            )

        (
            y_true_clean,
            y_pred_clean,
            y_proba_clean,
            n_dropped,
            _keep_mask,
        ) = _strip_below_floor_rows(test_y, y_pred, y_proba)
        metrics = compute_all(
            task_type=spec.task_type,
            y_true=y_true_clean,
            y_pred=y_pred_clean,
            y_proba=y_proba_clean,
            pos_label=spec.positive_label,
            labels=labels,
        )
    except (
        RuntimeError,
        MemoryError,
        NotFittedError,
        ProbaUnsupportedError,
        QuantilesUnsupportedError,
    ) as exc:
        logger.warning(
            "run_hpo_uplift: final fit failed on %s/%s seed=%d fold=%d: %s: %s",
            spec.name,
            model_name,
            seed,
            fold_index,
            type(exc).__name__,
            exc,
        )
        return _build_tuned_row(
            env=env,
            spec=spec,
            model_name=model_name,
            seed=seed,
            fold_index=fold_index,
            n_folds=n_folds,
            split_fingerprint=split_fingerprint,
            l_resolved=l_resolved,
            started_at_utc=started_at_utc,
            hpo_tier=hpo_tier,
            hpo_n_trials_completed=n_completed,
            hpo_n_trials_pruned=n_pruned,
            hpo_search_space_size=hpo_space.search_space_size,
            hpo_timeout_seconds=timeout_seconds,
            hpo_best_trial_number=best_trial_number,
            hpo_time_to_best_seconds=time_to_best,
            hpo_best_val_loss=best_val_loss,
            skipped_reason=f"{_SKIP_REASON_ADAPTER_ERROR}: {type(exc).__name__}: {exc}",
            metrics_dump=None,
            fit_resource=None,
        )

    return _build_tuned_row(
        env=env,
        spec=spec,
        model_name=model_name,
        seed=seed,
        fold_index=fold_index,
        n_folds=n_folds,
        split_fingerprint=split_fingerprint,
        l_resolved=l_resolved,
        started_at_utc=started_at_utc,
        hpo_tier=hpo_tier,
        hpo_n_trials_completed=n_completed,
        hpo_n_trials_pruned=n_pruned,
        hpo_search_space_size=hpo_space.search_space_size,
        hpo_timeout_seconds=timeout_seconds,
        hpo_best_trial_number=best_trial_number,
        hpo_time_to_best_seconds=time_to_best,
        hpo_best_val_loss=best_val_loss,
        skipped_reason=None,
        metrics_dump=metrics.model_dump(),
        fit_resource=fit_resource,
        n_below_floor_dropped=n_dropped,
    )


def _expected_hpo_cell_keys(
    *,
    dataset_name: str,
    model_names: Sequence[str],
    seeds: Sequence[int],
    n_folds: int,
) -> set[str]:
    """Set of `variant="tuned"` cell keys the run will produce."""
    return {
        cell_key(
            dataset_name=dataset_name,
            model_name=model_name,
            seed=seed,
            variant=_HPO_VARIANT,
            fold_index=fold_index,
        )
        for model_name in model_names
        for seed in seeds
        for fold_index in range(n_folds)
    }


def run_hpo_uplift(
    config: BenchmarkConfig,
    *,
    output_root: Path,
    env: RunEnvironment,
) -> HPOUpliftExperimentResult:
    """Execute the HPO-uplift tuned arm across the config's cells.

    Per cell: run an Optuna study on an inner train/val split,
    refit the best config on the full train fold, evaluate on the
    test fold, write one `variant="tuned"` ResultRow.

    The default arm reuses B5's manifest; this driver does NOT
    re-run it.

    Raises:
        DatasetNotRegisteredError, ModelNotRegisteredError: a name
            in the config has no registry entry.
        ValueError: no `hpo_uplift` experiment is configured, or
            `cache_dir` is unset.
    """
    dataset_names = _resolve_names(config.datasets, list_datasets())
    model_names = _resolve_names(config.models, list_models())
    seeds = _resolve_hpo_seeds(config.experiments)
    n_trials, timeout_seconds, hpo_tier = _resolve_hpo_budget(config.experiments, env.profile)
    logger.info(
        "run_hpo_uplift: env.run_id=%s datasets=%d models=%d seeds=%s "
        "n_trials=%d timeout=%.1fs hpo_tier=%s",
        env.run_id,
        len(dataset_names),
        len(model_names),
        seeds,
        n_trials,
        timeout_seconds,
        hpo_tier,
    )
    cache_dir = config.cache_dir
    if cache_dir is None:
        raise ValueError(
            "run_hpo_uplift: BenchmarkConfig.cache_dir must be set; the "
            "loader callables consume it for the on-disk cache root"
        )

    attempted = 0
    skipped_task_mismatch = 0
    skipped_proba_unavailable = 0
    skipped_proba_runtime_unavailable = 0
    skipped_quantile_followup = 0
    skipped_hpo_family_not_registered = 0
    skipped_hpo_budget_zero = 0
    skipped_hpo_all_trials_pruned = 0
    skipped_adapter_error = 0
    skipped_optional_dep_missing = 0
    already_complete = 0
    completed_keys = set(list_completed_keys(output_root))

    for dataset_name in sorted(dataset_names):
        spec = get_dataset(dataset_name)
        if spec.excluded:
            logger.info(
                "run_hpo_uplift: dataset %s skipped (excluded=True; reason=%s)",
                dataset_name,
                spec.exclusion_reason,
            )
            continue

        expected_keys = _expected_hpo_cell_keys(
            dataset_name=dataset_name,
            model_names=model_names,
            seeds=seeds,
            n_folds=_DEFAULT_N_SPLITS,
        )
        if expected_keys.issubset(completed_keys):
            logger.info(
                "run_hpo_uplift: dataset %s all %d cells already complete; skipping loader",
                dataset_name,
                len(expected_keys),
            )
            already_complete += len(expected_keys)
            continue

        loader = get_loader(dataset_name)
        load_start = time.perf_counter()
        panel_dataset = loader(cache_dir)
        logger.info(
            "run_hpo_uplift: dataset %s loaded in %.2fs (%d rows)",
            dataset_name,
            time.perf_counter() - load_start,
            len(panel_dataset.panel),
        )

        l_resolved = resolve_lookback(spec)
        splitter = make_splitter(spec, lookback=l_resolved, n_splits=_DEFAULT_N_SPLITS)
        folds = list(_iter_folds(splitter, panel_dataset.panel))
        if not folds:
            logger.warning(
                "run_hpo_uplift: dataset %s produced no folds; skipping",
                dataset_name,
            )
            continue
        n_folds = len(folds)
        split_fingerprint = fingerprint_folds(idx for _, idx in folds)
        labels = _resolve_labels(panel_dataset, spec.task_type)

        for model_name in sorted(model_names):
            model_spec = get_model(model_name)
            applicable = spec.task_type in model_spec.task_types
            proba_required_but_unavailable = (
                applicable
                and spec.task_type in {"binary", "multiclass"}
                and not model_spec.supports_proba
            )
            for seed in seeds:
                for fold_index, (train_idx, test_idx) in folds:
                    key_started = time.perf_counter()
                    key = cell_key(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        seed=seed,
                        variant=_HPO_VARIANT,
                        fold_index=fold_index,
                    )
                    if is_cell_complete(output_root, key):
                        already_complete += 1
                        continue

                    if not applicable:
                        skipped_task_mismatch += 1
                        row = _build_tuned_row(
                            env=env,
                            spec=spec,
                            model_name=model_name,
                            seed=seed,
                            fold_index=fold_index,
                            n_folds=n_folds,
                            split_fingerprint=split_fingerprint,
                            l_resolved=l_resolved,
                            started_at_utc=_utc_now(),
                            hpo_tier=hpo_tier,
                            hpo_n_trials_completed=None,
                            hpo_n_trials_pruned=None,
                            hpo_search_space_size=None,
                            hpo_timeout_seconds=None,
                            hpo_best_trial_number=None,
                            hpo_time_to_best_seconds=None,
                            hpo_best_val_loss=None,
                            skipped_reason=(
                                f"{_SKIP_REASON_TASK_TYPE_MISMATCH}: "
                                f"model.task_types={model_spec.task_types} "
                                f"dataset.task_type={spec.task_type}"
                            ),
                            metrics_dump=None,
                            fit_resource=None,
                        )
                    elif proba_required_but_unavailable:
                        skipped_proba_unavailable += 1
                        row = _build_tuned_row(
                            env=env,
                            spec=spec,
                            model_name=model_name,
                            seed=seed,
                            fold_index=fold_index,
                            n_folds=n_folds,
                            split_fingerprint=split_fingerprint,
                            l_resolved=l_resolved,
                            started_at_utc=_utc_now(),
                            hpo_tier=hpo_tier,
                            hpo_n_trials_completed=None,
                            hpo_n_trials_pruned=None,
                            hpo_search_space_size=None,
                            hpo_timeout_seconds=None,
                            hpo_best_trial_number=None,
                            hpo_time_to_best_seconds=None,
                            hpo_best_val_loss=None,
                            skipped_reason=(
                                f"{_SKIP_REASON_PROBA_UNAVAILABLE}: model="
                                f"{model_name} declares supports_proba=False"
                            ),
                            metrics_dump=None,
                            fit_resource=None,
                        )
                    else:
                        row = _run_one_hpo_cell(
                            env=env,
                            panel_dataset=panel_dataset,
                            spec=spec,
                            model_name=model_name,
                            seed=seed,
                            fold_index=fold_index,
                            n_folds=n_folds,
                            train_idx=train_idx,
                            test_idx=test_idx,
                            split_fingerprint=split_fingerprint,
                            l_resolved=l_resolved,
                            labels=labels,
                            hpo_tier=hpo_tier,
                            n_trials=n_trials,
                            timeout_seconds=timeout_seconds,
                        )
                        if row.skipped_reason is None:
                            attempted += 1
                        elif row.skipped_reason.startswith(_SKIP_REASON_QUANTILE_FOLLOWUP):
                            skipped_quantile_followup += 1
                        elif row.skipped_reason.startswith(_SKIP_REASON_HPO_FAMILY_NOT_REGISTERED):
                            skipped_hpo_family_not_registered += 1
                        elif row.skipped_reason.startswith(_SKIP_REASON_HPO_BUDGET_ZERO):
                            skipped_hpo_budget_zero += 1
                        elif row.skipped_reason.startswith(_SKIP_REASON_HPO_ALL_TRIALS_PRUNED):
                            skipped_hpo_all_trials_pruned += 1
                        elif row.skipped_reason.startswith(_SKIP_REASON_PROBA_RUNTIME_UNAVAILABLE):
                            skipped_proba_runtime_unavailable += 1
                        elif row.skipped_reason.startswith(_SKIP_REASON_OPTIONAL_DEP_MISSING):
                            skipped_optional_dep_missing += 1
                        else:
                            skipped_adapter_error += 1

                    write_cell(output_root, row)
                    logger.info(
                        "run_hpo_uplift: cell %s done in %.2fs (reason=%s)",
                        key,
                        time.perf_counter() - key_started,
                        row.skipped_reason or "ok",
                    )

    summary = HPOUpliftExperimentResult(
        run_id=env.run_id,
        cells_attempted=attempted,
        cells_skipped_task_mismatch=skipped_task_mismatch,
        cells_skipped_proba_unavailable=skipped_proba_unavailable,
        cells_skipped_proba_runtime_unavailable=skipped_proba_runtime_unavailable,
        cells_skipped_quantile_followup=skipped_quantile_followup,
        cells_skipped_hpo_family_not_registered=skipped_hpo_family_not_registered,
        cells_skipped_hpo_budget_zero=skipped_hpo_budget_zero,
        cells_skipped_hpo_all_trials_pruned=skipped_hpo_all_trials_pruned,
        cells_skipped_adapter_error=skipped_adapter_error,
        cells_skipped_optional_dep_missing=skipped_optional_dep_missing,
        cells_already_complete=already_complete,
    )
    logger.info("run_hpo_uplift: complete: %s", summary.model_dump())
    return summary


__all__ = [
    "HPOUpliftExperimentResult",
    "run_hpo_uplift",
]
