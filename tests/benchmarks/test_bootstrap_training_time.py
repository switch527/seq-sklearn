"""Phase B14 D-B13.2 training-time bootstrap-CI aggregator tests.

Covers `aggregate_bootstrap_training_time_rollup`: happy path,
all-cells-skipped sentinel, mixed-skip, empty manifest, malformed
cell raise, per-spec n_resamples override, OOM gate, fingerprint
round-trip, n-entities-zero-loss path (NOT single-entity
degenerate), single-entity degenerate primitive path, and the
bootstrap-degeneracy oracle (Round-1 qa-C1 closure).
"""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from benchmarks.bootstrap_manifest import (
    load_training_time_rollup,
    training_time_rollup_path,
)
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments import build_run_environment, run_raw_loss
from benchmarks.report.bootstrap_rollup import RawRollupError
from benchmarks.report.bootstrap_training_time import (
    aggregate_bootstrap_training_time_rollup,
)
from benchmarks.run_manifest import (
    RunManifest,
    build_run_manifest,
    write_run_manifest,
)

from tests.benchmarks._fakes import register_all_fakes_and_get_panels


def _make_config(
    tmp_path: Path,
    *,
    seeds: tuple[int, ...] = (0, 1),
    bootstrap_n_resamples: int | None = None,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(
            ExperimentSpec(kind="raw_loss", seeds=seeds),
            ExperimentSpec(
                kind="training_time",
                seeds=seeds,
                bootstrap_n_resamples=bootstrap_n_resamples,
            ),
        ),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )


def _setup_and_run_b5(
    tmp_path: Path,
    *,
    seeds: tuple[int, ...] = (0, 1),
    bootstrap_n_resamples: int | None = None,
) -> tuple[BenchmarkConfig, Path, RunManifest]:
    register_all_fakes_and_get_panels()
    config = _make_config(tmp_path, seeds=seeds, bootstrap_n_resamples=bootstrap_n_resamples)
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    env = build_run_environment(profile="smoke")
    run_raw_loss(config, output_root=output_root, env=env)
    manifest = build_run_manifest(
        config=config,
        run_id="b14-tt-test",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)
    return config, output_root, manifest


# --- Empty manifest ---------------------------------------------------------


def test_aggregate_bootstrap_training_time_rollup_empty_manifest_returns_empty_list(
    tmp_path: Path,
) -> None:
    """Absent B5 manifest -> []; no rollup file written."""
    register_all_fakes_and_get_panels()
    config = _make_config(tmp_path)
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        config=config,
        run_id="b14-empty",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)
    env = build_run_environment(profile="smoke")

    rows = aggregate_bootstrap_training_time_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert rows == []
    assert not training_time_rollup_path(output_root).exists()


# --- Classification happy path ----------------------------------------------


def test_aggregate_bootstrap_training_time_rollup_happy_path_emits_ci(
    tmp_path: Path,
) -> None:
    """Run B5 (writes wall_seconds to the manifest), then run
    the training-time CI aggregator; assert the rollup row has
    `n_cells_evaluated >= n_seeds * n_folds`, a positive mean,
    and ci_lo <= mean <= ci_hi."""
    config, output_root, manifest = _setup_and_run_b5(tmp_path)
    env = build_run_environment(profile="smoke")

    rows = aggregate_bootstrap_training_time_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert len(rows) >= 1
    row = rows[0]
    assert row.dataset_name == "fake_binary"
    assert row.model_name == "fake_constant_binary"
    assert row.primary_metric == "wall_seconds"
    assert row.n_cells_evaluated >= 1
    assert row.primary_loss_mean is not None
    assert row.primary_loss_ci_lo is not None
    assert row.primary_loss_ci_hi is not None
    assert row.primary_loss_mean >= 0.0
    assert row.primary_loss_ci_lo <= row.primary_loss_mean <= row.primary_loss_ci_hi


def test_aggregate_bootstrap_training_time_rollup_ci_width_nonzero_with_multiple_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage-3 qa-I2 closure (strict-width assertion): pin
    ci_hi - ci_lo > 0 on a multi-cell group with non-identical
    wall_seconds. The standard fixture's wall_seconds vary with
    system jitter, which is too noisy to assert strict > 0
    against, so we substitute a deterministic non-constant
    wall_seconds vector via monkeypatch on load_run."""
    import pandas as pd

    config, output_root, manifest = _setup_and_run_b5(tmp_path)

    rows_df = pd.DataFrame(
        [
            {
                "dataset_name": "fake_binary",
                "model_name": "fake_constant_binary",
                "hardware_tier": "cpu",
                "task_type": "binary",
                "seed": s,
                "fold_index": f,
                "wall_seconds": 1.0 + 0.1 * (2 * s + f),
                "skipped_reason": None,
            }
            for s in (0, 1, 2)
            for f in (0, 1)
        ]
    )

    import benchmarks.report.bootstrap_training_time as _module

    monkeypatch.setattr(_module, "load_run", lambda _root: rows_df)

    env = build_run_environment(profile="smoke")
    rows = aggregate_bootstrap_training_time_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.n_cells_evaluated == 6
    assert row.primary_loss_mean is not None
    assert row.primary_loss_ci_lo is not None
    assert row.primary_loss_ci_hi is not None
    # Strict assertion: CI width must be > 0. An entity-id
    # degeneracy bug (entity_ids=np.zeros) would produce
    # ci_lo == ci_hi and fail this test.
    assert row.primary_loss_ci_hi - row.primary_loss_ci_lo > 0.0


def test_aggregate_bootstrap_training_time_rollup_zero_wall_seconds_produces_zero_width_ci(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-1 qa-I3 reframe: when every cell has
    wall_seconds=0.0 (smoke-tier degenerate), the rollup row
    has primary_loss_mean = ci_lo = ci_hi = 0.0. This is the
    n-entities-zero-loss path (each cell is its own entity),
    NOT the single-entity degenerate path at
    benchmarks/metrics/bootstrap.py:133-136."""
    config, output_root, manifest = _setup_and_run_b5(tmp_path, seeds=(0, 1, 2))

    captured: dict[str, object] = {}

    def _zero_primitive(
        losses: np.ndarray,
        entity_ids: np.ndarray,
        **kwargs: object,
    ) -> tuple[float, float, float]:
        # Force all losses to zero so the bootstrap collapses to (0, 0, 0).
        zero_losses = np.zeros_like(losses)
        captured["n_entities"] = int(np.unique(entity_ids).shape[0])
        captured["n_cells"] = int(zero_losses.shape[0])
        return (0.0, 0.0, 0.0)

    import benchmarks.report.bootstrap_training_time as _module

    monkeypatch.setattr(_module, "entity_block_bootstrap_ci", _zero_primitive)

    env = build_run_environment(profile="smoke")
    rows = aggregate_bootstrap_training_time_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert len(rows) >= 1
    row = rows[0]
    assert row.primary_loss_mean == 0.0
    assert row.primary_loss_ci_lo == 0.0
    assert row.primary_loss_ci_hi == 0.0
    # Confirm the code path: each cell its own entity (n_entities == n_cells).
    assert captured["n_entities"] == captured["n_cells"]
    assert int(cast(int, captured["n_cells"])) >= 2  # multi-entity, NOT single-entity degenerate


def test_aggregate_bootstrap_training_time_rollup_single_entity_degenerate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage-3 qa-C1 closure: pin the primitive's n_entities==1
    degenerate path at benchmarks/metrics/bootstrap.py:133-136.
    Forces a 1-row manifest via monkeypatch so the assertion is
    unconditional (the prior test had an `if n_cells_evaluated
    == 1` guard that never fired under the standard fixture)."""
    import pandas as pd

    config, output_root, manifest = _setup_and_run_b5(tmp_path)

    # Construct a 1-row manifest that the aggregator will see.
    # Only one OK cell -> n_cells_evaluated == 1 -> primitive's
    # single-entity-degenerate path fires.
    single_row_df = pd.DataFrame(
        [
            {
                "dataset_name": "fake_binary",
                "model_name": "fake_constant_binary",
                "hardware_tier": "cpu",
                "task_type": "binary",
                "seed": 0,
                "fold_index": 0,
                "wall_seconds": 1.234,
                "skipped_reason": None,
            }
        ]
    )

    import benchmarks.report.bootstrap_training_time as _module

    monkeypatch.setattr(_module, "load_run", lambda _root: single_row_df)

    env = build_run_environment(profile="smoke")
    rows = aggregate_bootstrap_training_time_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.n_cells_evaluated == 1
    # Unconditional assertion: pins the single-entity-degenerate path.
    assert row.primary_loss_mean == pytest.approx(1.234, abs=1e-9)
    assert row.primary_loss_ci_lo == row.primary_loss_mean
    assert row.primary_loss_ci_hi == row.primary_loss_mean


# --- All-cells-skipped sentinel ---------------------------------------------


def test_aggregate_bootstrap_training_time_rollup_all_cells_skipped_emits_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every row in a (dataset, model, hardware_tier) group
    has a populated `skipped_reason`, the rollup emits a sentinel
    row with bootstrap_skipped_reason="all_cells_skipped_in_manifest"."""
    config, output_root, manifest = _setup_and_run_b5(tmp_path)
    # Mutate the B5 manifest: mark every OK row as skipped.
    import pandas as pd
    from benchmarks.manifest import load_run

    df = load_run(output_root)
    df["skipped_reason"] = "all_skipped_for_test"
    # Rewrite the results directory with the mutated rows.
    results_dir = output_root / "results"
    for shard in results_dir.glob("*.parquet"):
        shard.unlink()
    for i, row in df.iterrows():
        single = pd.DataFrame([row.to_dict()])
        single.to_parquet(results_dir / f"shard_{i:04d}.parquet", index=False)

    env = build_run_environment(profile="smoke")
    rows = aggregate_bootstrap_training_time_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert len(rows) >= 1
    row = rows[0]
    assert row.bootstrap_skipped_reason == "all_cells_skipped_in_manifest"
    assert row.n_cells_evaluated == 0


# --- Malformed cell ---------------------------------------------------------


def test_aggregate_bootstrap_training_time_rollup_malformed_cell_raises(
    tmp_path: Path,
) -> None:
    """An OK row (skipped_reason is None) with NaN wall_seconds
    is malformed and raises RawRollupError."""
    config, output_root, manifest = _setup_and_run_b5(tmp_path)
    # Mutate B5: set the FIRST OK row's wall_seconds to NaN.
    import pandas as pd
    from benchmarks.manifest import load_run

    df = load_run(output_root)
    ok_mask = df["skipped_reason"].isna()
    first_ok_index = df.index[ok_mask][0]
    df.at[first_ok_index, "wall_seconds"] = float("nan")
    results_dir = output_root / "results"
    for shard in results_dir.glob("*.parquet"):
        shard.unlink()
    for i, row in df.iterrows():
        single = pd.DataFrame([row.to_dict()])
        single.to_parquet(results_dir / f"shard_{i:04d}.parquet", index=False)

    env = build_run_environment(profile="smoke")
    with pytest.raises(RawRollupError, match="NaN wall_seconds"):
        aggregate_bootstrap_training_time_rollup(
            config, output_root=output_root, env=env, manifest=manifest
        )


# --- Per-spec n_resamples override ------------------------------------------


def test_aggregate_bootstrap_training_time_rollup_respects_per_spec_n_resamples_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ExperimentSpec(kind="training_time", bootstrap_n_resamples=N)
    overrides the profile default."""
    config, output_root, manifest = _setup_and_run_b5(tmp_path, bootstrap_n_resamples=137)

    captured: dict[str, int] = {}

    def _capturing_primitive(
        losses: object, entity_ids: object, *, n_resamples: int, **_kwargs: object
    ) -> tuple[float, float, float]:
        captured["n_resamples"] = n_resamples
        return (1.0, 0.9, 1.1)

    import benchmarks.report.bootstrap_training_time as _module

    monkeypatch.setattr(_module, "entity_block_bootstrap_ci", _capturing_primitive)

    env = build_run_environment(profile="smoke")
    aggregate_bootstrap_training_time_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert captured["n_resamples"] == 137


# --- OOM ceiling raise ------------------------------------------------------


def test_aggregate_bootstrap_training_time_rollup_missing_wall_seconds_column_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage-3 qa-I1 closure: when the B5 manifest has no
    `wall_seconds` column (schema drift), the aggregator raises
    RawRollupError with a clear message rather than silently
    proceeding."""
    import pandas as pd

    config, output_root, manifest = _setup_and_run_b5(tmp_path)

    # Manifest with all required columns EXCEPT wall_seconds.
    rows_df = pd.DataFrame(
        [
            {
                "dataset_name": "fake_binary",
                "model_name": "fake_constant_binary",
                "hardware_tier": "cpu",
                "task_type": "binary",
                "seed": 0,
                "fold_index": 0,
                "skipped_reason": None,
            }
        ]
    )

    import benchmarks.report.bootstrap_training_time as _module

    monkeypatch.setattr(_module, "load_run", lambda _root: rows_df)

    env = build_run_environment(profile="smoke")
    with pytest.raises(RawRollupError, match="no `wall_seconds` column"):
        aggregate_bootstrap_training_time_rollup(
            config, output_root=output_root, env=env, manifest=manifest
        )


def test_aggregate_bootstrap_training_time_rollup_oom_gate_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lower BOOTSTRAP_ROW_COUNT_CEILING to 1 so the tiny test
    fixture trips the OOM gate."""
    config, output_root, manifest = _setup_and_run_b5(tmp_path)

    import benchmarks.report.bootstrap_training_time as _module

    monkeypatch.setattr(_module, "BOOTSTRAP_ROW_COUNT_CEILING", 1)
    env = build_run_environment(profile="smoke")
    with pytest.raises(RawRollupError, match="ceiling"):
        aggregate_bootstrap_training_time_rollup(
            config, output_root=output_root, env=env, manifest=manifest
        )


# --- Fingerprint round-trip -------------------------------------------------


def test_aggregate_bootstrap_training_time_rollup_records_manifest_fingerprint(
    tmp_path: Path,
) -> None:
    """Every emitted row's `manifest_fingerprint` equals the
    live `RunManifest.fingerprint()`."""
    config, output_root, manifest = _setup_and_run_b5(tmp_path)
    env = build_run_environment(profile="smoke")
    rows = aggregate_bootstrap_training_time_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    fingerprint = manifest.fingerprint()
    assert all(row.manifest_fingerprint == fingerprint for row in rows)


# --- Parquet write ----------------------------------------------------------


def test_aggregate_bootstrap_training_time_rollup_writes_parquet_shard(
    tmp_path: Path,
) -> None:
    """The aggregator writes the parquet shard on a successful
    call; load_training_time_rollup round-trips the rows."""
    config, output_root, manifest = _setup_and_run_b5(tmp_path)
    env = build_run_environment(profile="smoke")
    rows_out = aggregate_bootstrap_training_time_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert training_time_rollup_path(output_root).exists()
    loaded = load_training_time_rollup(output_root)
    assert len(loaded) == len(rows_out)
    assert loaded[0].model_dump() == rows_out[0].model_dump()
