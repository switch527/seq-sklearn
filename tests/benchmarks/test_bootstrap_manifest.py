"""Phase B13 bootstrap-manifest tests (RollupRow round-trip,
write_rollup atomic semantics, rollup_path format).
"""

from pathlib import Path

import pytest
from benchmarks.bootstrap_manifest import (
    PairwiseRollupRow,
    RollupRow,
    TrainingTimeRollupRow,
    load_pairwise_rollup,
    load_rollup,
    load_training_time_rollup,
    pairwise_aggregator_failed_sentinel_path,
    pairwise_rollup_path,
    rollup_path,
    training_time_aggregator_failed_sentinel_path,
    training_time_rollup_path,
    write_pairwise_rollup,
    write_rollup,
    write_training_time_rollup,
)


def _make_row(**overrides: object) -> RollupRow:
    """Build a `RollupRow` with EVERY field populated, including
    the new Gemini-pass fields (`manifest_fingerprint`,
    `bootstrap_numpy_version`) AND `bootstrap_rng_algorithm` AND a
    `bootstrap_skipped_reason` string. Overrides are merged per
    key."""
    defaults: dict[str, object] = {
        "dataset_name": "fake_binary",
        "model_name": "lightgbm_classifier",
        "task_type": "binary",
        "primary_metric": "log_loss",
        "n_seeds": 3,
        "n_cells_evaluated": 9,
        "n_skipped_cells": 0,
        "n_rows": 240,
        "n_entities": 12,
        "primary_metric_mean": 0.234,
        "primary_metric_ci_lo": 0.221,
        "primary_metric_ci_hi": 0.247,
        "bootstrap_seed": 42,
        "bootstrap_n_resamples": 10_000,
        "bootstrap_confidence": 0.95,
        "bootstrap_rng_algorithm": "PCG64",
        "bootstrap_numpy_version": "2.3.0",
        "bootstrap_skipped_reason": None,
        "manifest_fingerprint": "deadbeef" * 8,
    }
    defaults.update(overrides)
    return RollupRow(**defaults)  # type: ignore[arg-type]


def test_rollup_path_format(tmp_path: Path) -> None:
    """`rollup_path(root)` returns `{root}/bootstrap_rollup.parquet`."""
    assert rollup_path(tmp_path) == tmp_path / "bootstrap_rollup.parquet"


def test_write_rollup_then_load_rollup_round_trips_all_fields(
    tmp_path: Path,
) -> None:
    """Build a `RollupRow` with EVERY field populated including
    `manifest_fingerprint`, `bootstrap_numpy_version`,
    `bootstrap_rng_algorithm`, and `bootstrap_skipped_reason`;
    write via `write_rollup`, load via `load_rollup`, and assert
    every field round-trips exactly. Pins the new pydantic
    schema's parquet serialization for ALL Gemini-added fields."""
    rows = [
        _make_row(),
        _make_row(
            dataset_name="fake_regression_point",
            model_name="lightgbm_regressor",
            task_type="regression_point",
            primary_metric="rmse",
            primary_metric_mean=1.23,
            primary_metric_ci_lo=1.1,
            primary_metric_ci_hi=1.35,
        ),
        _make_row(
            model_name="bad_model",
            n_cells_evaluated=0,
            n_skipped_cells=9,
            n_rows=0,
            n_entities=0,
            primary_metric_mean=None,
            primary_metric_ci_lo=None,
            primary_metric_ci_hi=None,
            bootstrap_skipped_reason="all_cells_skipped_in_manifest",
        ),
    ]
    write_rollup(tmp_path, rows)
    loaded = load_rollup(tmp_path)
    assert len(loaded) == len(rows)
    for orig, got in zip(rows, loaded, strict=True):
        assert got.dataset_name == orig.dataset_name
        assert got.model_name == orig.model_name
        assert got.task_type == orig.task_type
        assert got.primary_metric == orig.primary_metric
        assert got.n_seeds == orig.n_seeds
        assert got.n_cells_evaluated == orig.n_cells_evaluated
        assert got.n_skipped_cells == orig.n_skipped_cells
        assert got.n_rows == orig.n_rows
        assert got.n_entities == orig.n_entities
        assert got.primary_metric_mean == orig.primary_metric_mean
        assert got.primary_metric_ci_lo == orig.primary_metric_ci_lo
        assert got.primary_metric_ci_hi == orig.primary_metric_ci_hi
        assert got.bootstrap_seed == orig.bootstrap_seed
        assert got.bootstrap_n_resamples == orig.bootstrap_n_resamples
        assert got.bootstrap_confidence == orig.bootstrap_confidence
        assert got.bootstrap_rng_algorithm == orig.bootstrap_rng_algorithm
        assert got.bootstrap_numpy_version == orig.bootstrap_numpy_version
        assert got.bootstrap_skipped_reason == orig.bootstrap_skipped_reason
        assert got.manifest_fingerprint == orig.manifest_fingerprint


def test_write_rollup_atomic_replace_on_overwrite(tmp_path: Path) -> None:
    """Write twice with different row counts; assert the file at
    the second write is the second write's content, not a partial
    mix. Mirrors `test_run_manifest.py:test_write_run_manifest_atomic_replace_on_overwrite`."""
    write_rollup(tmp_path, [_make_row(), _make_row(model_name="m2")])
    first = load_rollup(tmp_path)
    assert len(first) == 2

    write_rollup(tmp_path, [_make_row(model_name="m3")])
    second = load_rollup(tmp_path)
    assert len(second) == 1
    assert second[0].model_name == "m3"


def test_load_rollup_returns_empty_when_file_absent(tmp_path: Path) -> None:
    """Absent rollup is the renderer's signal to use the std
    variant; `load_rollup` returns `[]` rather than raising."""
    assert load_rollup(tmp_path) == []


def test_write_rollup_empty_rows_writes_empty_shard(tmp_path: Path) -> None:
    """An empty rollup (e.g., no datasets matched the dispatch)
    writes an empty DataFrame so a subsequent load returns []."""
    write_rollup(tmp_path, [])
    assert load_rollup(tmp_path) == []


def test_rollup_row_extra_forbid_rejects_unknown_field() -> None:
    """The pydantic schema is `extra="forbid"`; passing an
    unknown field raises `ValidationError` so a future schema
    drift surfaces at construction time."""
    with pytest.raises(Exception, match="extra"):
        RollupRow.model_validate({**_make_row().model_dump(), "ghost": 1})


# --- ExperimentSpec extension validation -----------------------------------


def test_experiment_spec_bootstrap_n_resamples_below_one_raises_validation() -> None:
    """N1: `bootstrap_n_resamples` has `Field(ge=1)`; 0 must
    raise pydantic `ValidationError` at construction."""
    from benchmarks.config import ExperimentSpec
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="bootstrap_n_resamples"):
        ExperimentSpec(kind="raw_loss", bootstrap_n_resamples=0)


# --- Phase B14: PairwiseRollupRow + TrainingTimeRollupRow schemas ----------


def _make_pairwise_row(**overrides: object) -> PairwiseRollupRow:
    defaults: dict[str, object] = {
        "dataset_name": "fake_binary",
        "model_a": "lightgbm_classifier",
        "model_b": "tft_classifier",
        "task_type": "binary",
        "primary_metric": "complementarity_score",
        "n_seeds": 3,
        "n_cells_evaluated": 9,
        "n_skipped_cells": 0,
        "primary_metric_mean": 0.612,
        "primary_metric_ci_lo": 0.581,
        "primary_metric_ci_hi": 0.643,
        "bootstrap_seed": 42,
        "bootstrap_n_resamples": 10_000,
        "bootstrap_confidence": 0.95,
        "bootstrap_rng_algorithm": "PCG64",
        "bootstrap_numpy_version": "2.3.0",
        "bootstrap_skipped_reason": None,
        "manifest_fingerprint": "feedface" * 8,
    }
    defaults.update(overrides)
    return PairwiseRollupRow(**defaults)  # type: ignore[arg-type]


def _make_training_time_row(**overrides: object) -> TrainingTimeRollupRow:
    defaults: dict[str, object] = {
        "dataset_name": "fake_binary",
        "model_name": "lightgbm_classifier",
        "hardware_tier": "cpu",
        "task_type": "binary",
        "primary_metric": "wall_seconds",
        "n_seeds": 3,
        "n_cells_evaluated": 9,
        "n_skipped_cells": 0,
        "primary_metric_mean": 1.234,
        "primary_metric_ci_lo": 1.10,
        "primary_metric_ci_hi": 1.35,
        "bootstrap_seed": 42,
        "bootstrap_n_resamples": 10_000,
        "bootstrap_confidence": 0.95,
        "bootstrap_rng_algorithm": "PCG64",
        "bootstrap_numpy_version": "2.3.0",
        "bootstrap_skipped_reason": None,
        "manifest_fingerprint": "cafef00d" * 8,
    }
    defaults.update(overrides)
    return TrainingTimeRollupRow(**defaults)  # type: ignore[arg-type]


def test_pairwise_rollup_path_format(tmp_path: Path) -> None:
    """B14: `pairwise_rollup_path(root)` returns
    `{root}/bootstrap_pairwise_rollup.parquet`."""
    assert pairwise_rollup_path(tmp_path) == tmp_path / "bootstrap_pairwise_rollup.parquet"


def test_training_time_rollup_path_format(tmp_path: Path) -> None:
    """B14: `training_time_rollup_path(root)` returns
    `{root}/bootstrap_training_time_rollup.parquet`."""
    assert (
        training_time_rollup_path(tmp_path) == tmp_path / "bootstrap_training_time_rollup.parquet"
    )


def test_pairwise_aggregator_failed_sentinel_path_format(tmp_path: Path) -> None:
    """B14: pairwise sentinel filename is
    `bootstrap_pairwise_aggregator_failed.txt`."""
    assert (
        pairwise_aggregator_failed_sentinel_path(tmp_path)
        == tmp_path / "bootstrap_pairwise_aggregator_failed.txt"
    )


def test_training_time_aggregator_failed_sentinel_path_format(tmp_path: Path) -> None:
    """B14: training-time sentinel filename is
    `bootstrap_training_time_aggregator_failed.txt`."""
    assert (
        training_time_aggregator_failed_sentinel_path(tmp_path)
        == tmp_path / "bootstrap_training_time_aggregator_failed.txt"
    )


def test_write_pairwise_rollup_then_load_round_trips_all_fields(
    tmp_path: Path,
) -> None:
    """B14: write + load the PairwiseRollupRow shard; every field
    round-trips. Pins the new pydantic schema's parquet
    serialization including the `bootstrap_skipped_reason`
    sentinel branch."""
    rows = [
        _make_pairwise_row(),
        _make_pairwise_row(
            task_type="regression_point",
            primary_metric="complementarity_score",
            primary_metric_mean=None,
            primary_metric_ci_lo=None,
            primary_metric_ci_hi=None,
            bootstrap_skipped_reason="regression_complementarity_undefined",
        ),
    ]
    write_pairwise_rollup(tmp_path, rows)
    loaded = load_pairwise_rollup(tmp_path)
    assert len(loaded) == len(rows)
    for orig, got in zip(rows, loaded, strict=True):
        assert got.model_dump() == orig.model_dump()


def test_write_training_time_rollup_then_load_round_trips_all_fields(
    tmp_path: Path,
) -> None:
    """B14: write + load the TrainingTimeRollupRow shard; every
    field round-trips including the hardware_tier column."""
    rows = [
        _make_training_time_row(),
        _make_training_time_row(
            hardware_tier="gpu_single",
            primary_metric_mean=0.876,
            primary_metric_ci_lo=0.82,
            primary_metric_ci_hi=0.93,
        ),
        _make_training_time_row(
            n_cells_evaluated=0,
            n_skipped_cells=9,
            primary_metric_mean=None,
            primary_metric_ci_lo=None,
            primary_metric_ci_hi=None,
            bootstrap_skipped_reason="all_cells_skipped_in_manifest",
        ),
    ]
    write_training_time_rollup(tmp_path, rows)
    loaded = load_training_time_rollup(tmp_path)
    assert len(loaded) == len(rows)
    for orig, got in zip(rows, loaded, strict=True):
        assert got.model_dump() == orig.model_dump()


def test_pairwise_rollup_row_extra_forbid_rejects_unknown_field() -> None:
    """B14: `PairwiseRollupRow.model_config = extra="forbid"`;
    a future schema-drift extra field raises ValidationError."""
    with pytest.raises(Exception, match="extra"):
        PairwiseRollupRow.model_validate({**_make_pairwise_row().model_dump(), "ghost": 1})


def test_training_time_rollup_row_extra_forbid_rejects_unknown_field() -> None:
    """B14: `TrainingTimeRollupRow.model_config = extra="forbid"`;
    a future schema-drift extra field raises ValidationError."""
    with pytest.raises(Exception, match="extra"):
        TrainingTimeRollupRow.model_validate({**_make_training_time_row().model_dump(), "ghost": 1})


def test_write_pairwise_rollup_empty_rows_writes_empty_shard(
    tmp_path: Path,
) -> None:
    """B14: empty rows -> empty parquet shard; subsequent load returns []."""
    write_pairwise_rollup(tmp_path, [])
    assert load_pairwise_rollup(tmp_path) == []


def test_write_training_time_rollup_empty_rows_writes_empty_shard(
    tmp_path: Path,
) -> None:
    """B14: empty rows -> empty parquet shard; subsequent load returns []."""
    write_training_time_rollup(tmp_path, [])
    assert load_training_time_rollup(tmp_path) == []


def test_load_pairwise_rollup_returns_empty_when_file_absent(tmp_path: Path) -> None:
    """B14: absent pairwise shard -> []; matches the B13 dispatch
    convention so the renderer can detect 'no rollup ran' without
    raising."""
    assert load_pairwise_rollup(tmp_path) == []


def test_load_training_time_rollup_returns_empty_when_file_absent(
    tmp_path: Path,
) -> None:
    """B14: absent training-time shard -> []."""
    assert load_training_time_rollup(tmp_path) == []


# --- B14 ExperimentSpec extension ------------------------------------------


def test_experiment_spec_bootstrap_pairwise_enabled_defaults_true() -> None:
    """B14: `bootstrap_pairwise_enabled` defaults True so the CI
    rollup runs by default when an ensemble spec is configured."""
    from benchmarks.config import ExperimentSpec

    spec = ExperimentSpec(kind="ensemble")
    assert spec.bootstrap_pairwise_enabled is True


def test_experiment_spec_bootstrap_training_time_enabled_defaults_true() -> None:
    """B14: `bootstrap_training_time_enabled` defaults True so the
    CI rollup runs by default when a training_time spec is configured."""
    from benchmarks.config import ExperimentSpec

    spec = ExperimentSpec(kind="training_time")
    assert spec.bootstrap_training_time_enabled is True


def test_experiment_spec_bootstrap_pairwise_enabled_opt_out() -> None:
    """B14: setting `bootstrap_pairwise_enabled=False` opts out of
    the CI rollup; the wrapper will see this and skip."""
    from benchmarks.config import ExperimentSpec

    spec = ExperimentSpec(kind="ensemble", bootstrap_pairwise_enabled=False)
    assert spec.bootstrap_pairwise_enabled is False


def test_experiment_spec_bootstrap_training_time_enabled_opt_out() -> None:
    """B14: setting `bootstrap_training_time_enabled=False` opts out."""
    from benchmarks.config import ExperimentSpec

    spec = ExperimentSpec(kind="training_time", bootstrap_training_time_enabled=False)
    assert spec.bootstrap_training_time_enabled is False


# --- Phase B15: HPOUpliftRollupRow + ExperimentSpec extension --------------

from benchmarks.bootstrap_manifest import HPOUpliftRollupRow as _HPOUpliftRollupRow  # noqa: E402


def _make_hpo_uplift_row(**overrides: object) -> _HPOUpliftRollupRow:
    defaults: dict[str, object] = {
        "dataset_name": "fake_binary",
        "model_name": "fake_constant_binary",
        "task_type": "binary",
        "primary_metric": "delta",
        "primary_loss_column": "log_loss",
        "n_seeds": 3,
        "n_folds": 2,
        "n_cells_paired": 6,
        "n_skipped_cells": 0,
        "primary_metric_mean": 0.20,
        "primary_metric_ci_lo": 0.15,
        "primary_metric_ci_hi": 0.25,
        "bootstrap_seed": 42,
        "bootstrap_n_resamples": 10_000,
        "bootstrap_confidence": 0.95,
        "bootstrap_rng_algorithm": "PCG64",
        "bootstrap_numpy_version": "2.3.0",
        "bootstrap_skipped_reason": None,
        "manifest_fingerprint": "feedbeef" * 8,
    }
    defaults.update(overrides)
    return _HPOUpliftRollupRow(**defaults)  # type: ignore[arg-type]


def test_hpo_uplift_rollup_path_format(tmp_path: Path) -> None:
    """B15: `hpo_uplift_rollup_path(root)` returns
    `{root}/bootstrap_hpo_uplift_rollup.parquet`."""
    from benchmarks.bootstrap_manifest import hpo_uplift_rollup_path

    assert hpo_uplift_rollup_path(tmp_path) == tmp_path / "bootstrap_hpo_uplift_rollup.parquet"


def test_hpo_uplift_aggregator_failed_sentinel_path_format(tmp_path: Path) -> None:
    """B15: sentinel filename is
    `bootstrap_hpo_uplift_aggregator_failed.txt`."""
    from benchmarks.bootstrap_manifest import (
        hpo_uplift_aggregator_failed_sentinel_path,
    )

    assert (
        hpo_uplift_aggregator_failed_sentinel_path(tmp_path)
        == tmp_path / "bootstrap_hpo_uplift_aggregator_failed.txt"
    )


def test_write_hpo_uplift_rollup_then_load_round_trips_all_fields(
    tmp_path: Path,
) -> None:
    """B15: write + load the HPOUpliftRollupRow shard; every field
    round-trips. Pins `primary_loss_column`, `n_seeds`, `n_folds`,
    `n_cells_paired`, and the sentinel-reason branch."""
    from benchmarks.bootstrap_manifest import (
        load_hpo_uplift_rollup,
        write_hpo_uplift_rollup,
    )

    rows = [
        _make_hpo_uplift_row(),
        _make_hpo_uplift_row(
            task_type="regression_point",
            primary_loss_column="rmse",
            primary_metric_mean=-0.05,
            primary_metric_ci_lo=-0.10,
            primary_metric_ci_hi=0.00,
        ),
        _make_hpo_uplift_row(
            model_name="bad_model",
            n_cells_paired=0,
            n_skipped_cells=0,
            primary_metric_mean=None,
            primary_metric_ci_lo=None,
            primary_metric_ci_hi=None,
            bootstrap_skipped_reason="no_paired_cells",
        ),
    ]
    write_hpo_uplift_rollup(tmp_path, rows)
    loaded = load_hpo_uplift_rollup(tmp_path)
    assert len(loaded) == len(rows)
    for orig, got in zip(rows, loaded, strict=True):
        assert got.model_dump() == orig.model_dump()
        assert got.primary_loss_column == orig.primary_loss_column
        assert got.n_seeds == orig.n_seeds
        assert got.n_folds == orig.n_folds
        assert got.n_cells_paired == orig.n_cells_paired


def test_hpo_uplift_rollup_row_extra_forbid_rejects_unknown_field() -> None:
    """B15: extra="forbid" -> unknown field raises ValidationError."""
    with pytest.raises(Exception, match="extra"):
        _HPOUpliftRollupRow.model_validate({**_make_hpo_uplift_row().model_dump(), "ghost": 1})


def test_write_hpo_uplift_rollup_empty_rows_writes_empty_shard(
    tmp_path: Path,
) -> None:
    """B15: empty rows -> empty shard; load returns []."""
    from benchmarks.bootstrap_manifest import (
        load_hpo_uplift_rollup,
        write_hpo_uplift_rollup,
    )

    write_hpo_uplift_rollup(tmp_path, [])
    assert load_hpo_uplift_rollup(tmp_path) == []


def test_load_hpo_uplift_rollup_returns_empty_when_file_absent(
    tmp_path: Path,
) -> None:
    """B15: absent file -> [] (dispatch convention)."""
    from benchmarks.bootstrap_manifest import load_hpo_uplift_rollup

    assert load_hpo_uplift_rollup(tmp_path) == []


def test_experiment_spec_bootstrap_hpo_uplift_enabled_defaults_true() -> None:
    """B15: default-True so CI rollup runs when configured."""
    from benchmarks.config import ExperimentSpec

    spec = ExperimentSpec(kind="hpo_uplift")
    assert spec.bootstrap_hpo_uplift_enabled is True


def test_experiment_spec_bootstrap_hpo_uplift_enabled_opt_out() -> None:
    """B15: explicit opt-out."""
    from benchmarks.config import ExperimentSpec

    spec = ExperimentSpec(kind="hpo_uplift", bootstrap_hpo_uplift_enabled=False)
    assert spec.bootstrap_hpo_uplift_enabled is False


# --- Phase B16: EnsembleLiftRollupRow + ExperimentSpec extension -----------

from benchmarks.bootstrap_manifest import (  # noqa: E402
    EnsembleLiftRollupRow as _EnsembleLiftRollupRow,
)


def _make_ensemble_lift_row(**overrides: object) -> _EnsembleLiftRollupRow:
    defaults: dict[str, object] = {
        "dataset_name": "fake_binary",
        "task_type": "binary",
        "primary_metric": "delta_loss",
        "primary_loss_column": "log_loss",
        "n_seeds": 3,
        "n_folds": 2,
        "n_cells_paired": 6,
        # B19 / D-B16.7: factory default mirrors `n_cells_paired` so the
        # symmetric-roster happy-path default matches pre-B19 behavior
        # (renderer's partial-flag computation evaluates to False when
        # n_cells_paired == n_pair_grid).
        "n_pair_grid": 6,
        # B20 / D-B16.1: factory default mirrors `n_cells_paired`
        # (symmetric-roster happy path: every paired cell had a
        # computable oracle_loss). Concrete oracle CI bounds chosen
        # to be distinct from the primary metric so a cross-wire
        # mutation that writes primary into oracle fails any direct
        # equality assertion.
        "n_oracle_cells_paired": 6,
        "n_skipped_cells": 0,
        "primary_metric_mean": 0.20,
        "primary_metric_ci_lo": 0.15,
        "primary_metric_ci_hi": 0.25,
        "oracle_metric_mean": 0.10,
        "oracle_metric_ci_lo": 0.08,
        "oracle_metric_ci_hi": 0.12,
        "bootstrap_seed": 42,
        "bootstrap_n_resamples": 10_000,
        "bootstrap_confidence": 0.95,
        "bootstrap_rng_algorithm": "PCG64",
        "bootstrap_numpy_version": "2.3.0",
        "bootstrap_skipped_reason": None,
        "manifest_fingerprint": "beeff00d" * 8,
    }
    defaults.update(overrides)
    return _EnsembleLiftRollupRow(**defaults)  # type: ignore[arg-type]


def test_ensemble_lift_rollup_path_format(tmp_path: Path) -> None:
    """B16: `ensemble_lift_rollup_path(root)` returns
    `{root}/bootstrap_ensemble_lift_rollup.parquet`."""
    from benchmarks.bootstrap_manifest import ensemble_lift_rollup_path

    assert (
        ensemble_lift_rollup_path(tmp_path) == tmp_path / "bootstrap_ensemble_lift_rollup.parquet"
    )


def test_ensemble_lift_aggregator_failed_sentinel_path_format(
    tmp_path: Path,
) -> None:
    """B16: sentinel filename is
    `bootstrap_ensemble_lift_aggregator_failed.txt`. Independent
    from the B5/B6/B7/B8 sentinels."""
    from benchmarks.bootstrap_manifest import (
        ensemble_lift_aggregator_failed_sentinel_path,
    )

    assert (
        ensemble_lift_aggregator_failed_sentinel_path(tmp_path)
        == tmp_path / "bootstrap_ensemble_lift_aggregator_failed.txt"
    )


def test_write_ensemble_lift_rollup_then_load_round_trips_all_fields(
    tmp_path: Path,
) -> None:
    """B16: write + load the EnsembleLiftRollupRow shard; every
    field round-trips. Pins `primary_loss_column`, `n_seeds`,
    `n_folds`, `n_cells_paired`, and each sentinel-reason branch."""
    from benchmarks.bootstrap_manifest import (
        load_ensemble_lift_rollup,
        write_ensemble_lift_rollup,
    )

    rows = [
        _make_ensemble_lift_row(),
        _make_ensemble_lift_row(
            dataset_name="fake_regression",
            task_type="regression_point",
            primary_loss_column="rmse",
            primary_metric_mean=-0.05,
            primary_metric_ci_lo=-0.10,
            primary_metric_ci_hi=0.00,
        ),
        _make_ensemble_lift_row(
            dataset_name="bad_dataset_a",
            n_cells_paired=0,
            n_skipped_cells=0,
            primary_metric_mean=None,
            primary_metric_ci_lo=None,
            primary_metric_ci_hi=None,
            bootstrap_skipped_reason="no_gbm_predictions",
        ),
        _make_ensemble_lift_row(
            dataset_name="bad_dataset_b",
            n_cells_paired=0,
            n_skipped_cells=0,
            primary_metric_mean=None,
            primary_metric_ci_lo=None,
            primary_metric_ci_hi=None,
            bootstrap_skipped_reason="no_seq_predictions",
        ),
        _make_ensemble_lift_row(
            dataset_name="bad_dataset_c",
            n_cells_paired=0,
            n_skipped_cells=0,
            primary_metric_mean=None,
            primary_metric_ci_lo=None,
            primary_metric_ci_hi=None,
            bootstrap_skipped_reason="all_cells_skipped_in_manifest",
        ),
    ]
    write_ensemble_lift_rollup(tmp_path, rows)
    loaded = load_ensemble_lift_rollup(tmp_path)
    assert len(loaded) == len(rows)
    for orig, got in zip(rows, loaded, strict=True):
        assert got.model_dump() == orig.model_dump()
        assert got.primary_loss_column == orig.primary_loss_column
        assert got.n_seeds == orig.n_seeds
        assert got.n_folds == orig.n_folds
        assert got.n_cells_paired == orig.n_cells_paired


def test_ensemble_lift_rollup_row_extra_forbid_rejects_unknown_field() -> None:
    """B16: extra="forbid" -> unknown field raises ValidationError."""
    with pytest.raises(Exception, match="extra"):
        _EnsembleLiftRollupRow.model_validate(
            {**_make_ensemble_lift_row().model_dump(), "ghost": 1}
        )


def test_write_ensemble_lift_rollup_empty_rows_writes_empty_shard(
    tmp_path: Path,
) -> None:
    """B16: empty rows -> empty shard; load returns []."""
    from benchmarks.bootstrap_manifest import (
        load_ensemble_lift_rollup,
        write_ensemble_lift_rollup,
    )

    write_ensemble_lift_rollup(tmp_path, [])
    assert load_ensemble_lift_rollup(tmp_path) == []


def test_load_ensemble_lift_rollup_returns_empty_when_file_absent(
    tmp_path: Path,
) -> None:
    """B16: absent file -> [] (dispatch convention)."""
    from benchmarks.bootstrap_manifest import load_ensemble_lift_rollup

    assert load_ensemble_lift_rollup(tmp_path) == []


def test_experiment_spec_bootstrap_ensemble_lift_enabled_defaults_true() -> None:
    """B16: default-True so CI rollup runs when configured."""
    from benchmarks.config import ExperimentSpec

    spec = ExperimentSpec(kind="ensemble_lift")
    assert spec.bootstrap_ensemble_lift_enabled is True


def test_experiment_spec_bootstrap_ensemble_lift_enabled_opt_out() -> None:
    """B16: explicit opt-out."""
    from benchmarks.config import ExperimentSpec

    spec = ExperimentSpec(kind="ensemble_lift", bootstrap_ensemble_lift_enabled=False)
    assert spec.bootstrap_ensemble_lift_enabled is False


# --- Phase B17 / D-B16.5: Guard A (schema-field invariant) ----------------

from benchmarks.bootstrap_manifest import (  # noqa: E402
    EnsembleLiftRollupRow as _EnsembleLiftRollupRowAll,
)
from benchmarks.bootstrap_manifest import (  # noqa: E402
    HPOUpliftRollupRow as _HPOUpliftRollupRowAll,
)
from benchmarks.bootstrap_manifest import (  # noqa: E402
    PairwiseRollupRow as _PairwiseRollupRowAll,
)
from benchmarks.bootstrap_manifest import (  # noqa: E402
    RollupRow as _RollupRowAll,
)
from benchmarks.bootstrap_manifest import (  # noqa: E402
    TrainingTimeRollupRow as _TrainingTimeRollupRowAll,
)
from pydantic import BaseModel  # noqa: E402


@pytest.mark.parametrize(
    "schema",
    [
        _RollupRowAll,
        _PairwiseRollupRowAll,
        _TrainingTimeRollupRowAll,
        _HPOUpliftRollupRowAll,
        _EnsembleLiftRollupRowAll,
    ],
)
def test_rollup_row_has_no_stray_primary_loss_field(
    schema: type[BaseModel],
) -> None:
    """B17 R-B17-5 Guard A invariant: no rollup row may declare a
    `primary_loss_*` field other than the dedicated
    `primary_loss_column` audit field. The bootstrap math uses
    `primary_metric_*`; mixing the two leads back to the B13
    confusion this delta closes."""
    stray = [
        name
        for name in schema.model_fields
        if "primary_loss" in name and name != "primary_loss_column"
    ]
    assert stray == [], (
        f"{schema.__name__} has stray primary_loss_* field(s): "
        f"{stray}; use primary_metric_* per D-B16.5 / B17"
    )
