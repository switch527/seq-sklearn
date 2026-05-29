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
        "primary_loss_mean": 0.234,
        "primary_loss_ci_lo": 0.221,
        "primary_loss_ci_hi": 0.247,
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
            primary_loss_mean=1.23,
            primary_loss_ci_lo=1.1,
            primary_loss_ci_hi=1.35,
        ),
        _make_row(
            model_name="bad_model",
            n_cells_evaluated=0,
            n_skipped_cells=9,
            n_rows=0,
            n_entities=0,
            primary_loss_mean=None,
            primary_loss_ci_lo=None,
            primary_loss_ci_hi=None,
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
        assert got.primary_loss_mean == orig.primary_loss_mean
        assert got.primary_loss_ci_lo == orig.primary_loss_ci_lo
        assert got.primary_loss_ci_hi == orig.primary_loss_ci_hi
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
        "primary_loss_mean": 0.612,
        "primary_loss_ci_lo": 0.581,
        "primary_loss_ci_hi": 0.643,
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
        "primary_loss_mean": 1.234,
        "primary_loss_ci_lo": 1.10,
        "primary_loss_ci_hi": 1.35,
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
        training_time_rollup_path(tmp_path)
        == tmp_path / "bootstrap_training_time_rollup.parquet"
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
            primary_loss_mean=None,
            primary_loss_ci_lo=None,
            primary_loss_ci_hi=None,
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
            primary_loss_mean=0.876,
            primary_loss_ci_lo=0.82,
            primary_loss_ci_hi=0.93,
        ),
        _make_training_time_row(
            n_cells_evaluated=0,
            n_skipped_cells=9,
            primary_loss_mean=None,
            primary_loss_ci_lo=None,
            primary_loss_ci_hi=None,
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
        PairwiseRollupRow.model_validate(
            {**_make_pairwise_row().model_dump(), "ghost": 1}
        )


def test_training_time_rollup_row_extra_forbid_rejects_unknown_field() -> None:
    """B14: `TrainingTimeRollupRow.model_config = extra="forbid"`;
    a future schema-drift extra field raises ValidationError."""
    with pytest.raises(Exception, match="extra"):
        TrainingTimeRollupRow.model_validate(
            {**_make_training_time_row().model_dump(), "ghost": 1}
        )


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
