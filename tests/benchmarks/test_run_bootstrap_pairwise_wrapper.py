"""Phase B14 D-B13.1 CLI-wrapper tests for `_run_bootstrap_pairwise_rollup`.

Pins the wrapper's full contract:
- success path writes the parquet shard
- opt-out skip
- run_manifest absent skip
- run_manifest corrupt skip
- RawRollupError caught + partial deletion
- failure sentinel written with the exception class name
- stale failure sentinel unlinked on subsequent success
- cross-report sentinel isolation: a B6 failure does NOT touch
  the B5 or B7 sentinels (Round-1 qa-C3 closure)
"""

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.bootstrap_manifest import (
    aggregator_failed_sentinel_path,
    pairwise_aggregator_failed_sentinel_path,
    pairwise_rollup_path,
    training_time_aggregator_failed_sentinel_path,
)
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments import build_run_environment
from benchmarks.experiments.ensemble import pairwise_dir
from benchmarks.run import _run_bootstrap_pairwise_rollup
from benchmarks.run_manifest import (
    build_run_manifest,
    run_manifest_path,
    write_run_manifest,
)


def _build_config(
    *,
    tmp_path: Path,
    bootstrap_pairwise_enabled: bool = True,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(
            ExperimentSpec(kind="raw_loss", seeds=(0,)),
            ExperimentSpec(
                kind="ensemble",
                seeds=(0,),
                bootstrap_pairwise_enabled=bootstrap_pairwise_enabled,
            ),
        ),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )


def _write_minimal_pairwise(output_root: Path) -> None:
    target_dir = pairwise_dir(output_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "library_git_sha": "0" * 40,
                "run_id": "b14-wrapper-test",
                "started_at_utc": "2026-05-30T00:00:00+00:00",
                "dataset_name": "fake_binary",
                "model_a": "a",
                "model_b": "b",
                "seed": 0,
                "fold_index": 0,
                "task_type": "binary",
                "skipped_reason": None,
                "n_samples": 100,
                "n11": 40, "n10": 10, "n01": 15, "n00": 35,
                "yule_q": 0.7, "phi": 0.5,
                "disagreement_rate": 0.25,
                "double_fault_rate": 0.1,
                "pearson_pred_corr": 0.6,
                "spearman_pred_corr": 0.55,
                "pearson_error_corr": 0.30,
            }
        ]
    )
    df.to_parquet(target_dir / "shard_0000.parquet", index=False)


def _write_run_manifest(config: BenchmarkConfig, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        config=config,
        run_id="b14-wrapper-test",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)


# --- Success path -----------------------------------------------------------


def test_run_bootstrap_pairwise_rollup_writes_shard_on_happy_path(
    tmp_path: Path,
) -> None:
    """Default config (`bootstrap_pairwise_enabled=True`) plus
    a non-empty pairwise manifest -> rollup parquet written."""
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _write_minimal_pairwise(output_root)
    _write_run_manifest(config, output_root)

    env = build_run_environment(profile="smoke")
    _run_bootstrap_pairwise_rollup(config, env=env, output_root=output_root)
    assert pairwise_rollup_path(output_root).exists()


# --- Opt-out skip -----------------------------------------------------------


def test_run_bootstrap_pairwise_rollup_skips_via_opt_out(
    tmp_path: Path,
) -> None:
    """`bootstrap_pairwise_enabled=False` -> the wrapper skips
    and no rollup file lands."""
    config = _build_config(tmp_path=tmp_path, bootstrap_pairwise_enabled=False)
    output_root = tmp_path / "out"
    _write_minimal_pairwise(output_root)
    _write_run_manifest(config, output_root)

    env = build_run_environment(profile="smoke")
    _run_bootstrap_pairwise_rollup(config, env=env, output_root=output_root)
    assert not pairwise_rollup_path(output_root).exists()


# --- Missing run_manifest ---------------------------------------------------


def test_run_bootstrap_pairwise_rollup_skips_when_run_manifest_absent(
    tmp_path: Path,
) -> None:
    """No run_manifest.json -> wrapper logs and returns; no
    rollup file lands."""
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    _write_minimal_pairwise(output_root)
    # No write_run_manifest -> file absent.
    assert not run_manifest_path(output_root).exists()

    env = build_run_environment(profile="smoke")
    _run_bootstrap_pairwise_rollup(config, env=env, output_root=output_root)
    assert not pairwise_rollup_path(output_root).exists()


# --- Corrupt run_manifest ---------------------------------------------------


def test_run_bootstrap_pairwise_rollup_skips_when_run_manifest_load_fails(
    tmp_path: Path,
) -> None:
    """Corrupt run_manifest.json -> wrapper catches the load
    error and returns; no rollup file lands."""
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    _write_minimal_pairwise(output_root)
    run_manifest_path(output_root).write_bytes(b"{not valid json")

    env = build_run_environment(profile="smoke")
    _run_bootstrap_pairwise_rollup(config, env=env, output_root=output_root)
    assert not pairwise_rollup_path(output_root).exists()


# --- RawRollupError caught + sentinel written -------------------------------


def test_run_bootstrap_pairwise_rollup_catches_raw_rollup_error_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Monkeypatch the aggregator to raise `RawRollupError`;
    assert the wrapper catches it, deletes any partial output,
    writes the failure sentinel with the exception class name,
    and does NOT propagate the exception."""
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _write_minimal_pairwise(output_root)
    _write_run_manifest(config, output_root)

    # Pre-create a stale partial rollup file to verify deletion.
    pairwise_rollup_path(output_root).write_bytes(b"stale partial content")
    assert pairwise_rollup_path(output_root).exists()

    from benchmarks.report.bootstrap_rollup import RawRollupError as _Err

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise _Err("simulated pairwise aggregator failure")

    import benchmarks.run as _run_module

    monkeypatch.setattr(_run_module, "aggregate_bootstrap_pairwise_rollup", _boom)
    _run_bootstrap_pairwise_rollup(config, env=build_run_environment(profile="smoke"), output_root=output_root)
    # Partial output deleted.
    assert not pairwise_rollup_path(output_root).exists()
    # Sentinel written with the exception class name.
    sentinel = pairwise_aggregator_failed_sentinel_path(output_root)
    assert sentinel.exists()
    assert sentinel.read_text(encoding="utf-8").strip() == "RawRollupError"


# --- Stale sentinel unlink on success ---------------------------------------


def test_run_bootstrap_pairwise_rollup_unlinks_stale_failure_sentinel_on_success(
    tmp_path: Path,
) -> None:
    """A stale sentinel from a prior failed run is unlinked when
    the next aggregate call succeeds; otherwise the renderer
    would surface the std + failure footnote indefinitely."""
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _write_minimal_pairwise(output_root)
    _write_run_manifest(config, output_root)

    sentinel = pairwise_aggregator_failed_sentinel_path(output_root)
    sentinel.write_text("RawRollupError", encoding="utf-8")
    assert sentinel.exists()

    env = build_run_environment(profile="smoke")
    _run_bootstrap_pairwise_rollup(config, env=env, output_root=output_root)
    assert not sentinel.exists()
    assert pairwise_rollup_path(output_root).exists()
    assert pairwise_rollup_path(output_root).stat().st_size > 0


# --- Cross-report sentinel isolation (Round-1 qa-C3) ------------------------


def test_run_bootstrap_pairwise_rollup_failure_does_not_touch_b5_or_b7_sentinels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-1 qa-C3 closure: a B6 (pairwise) aggregator failure
    must write `bootstrap_pairwise_aggregator_failed.txt` and
    leave the B5 (`bootstrap_aggregator_failed.txt`) and B7
    (`bootstrap_training_time_aggregator_failed.txt`) sentinels
    untouched. This pins the three-way cross-report independence
    declared in B14.0."""
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _write_minimal_pairwise(output_root)
    _write_run_manifest(config, output_root)

    from benchmarks.report.bootstrap_rollup import RawRollupError as _Err

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise _Err("simulated pairwise aggregator failure")

    import benchmarks.run as _run_module

    monkeypatch.setattr(_run_module, "aggregate_bootstrap_pairwise_rollup", _boom)
    _run_bootstrap_pairwise_rollup(config, env=build_run_environment(profile="smoke"), output_root=output_root)

    # B6 sentinel exists.
    assert pairwise_aggregator_failed_sentinel_path(output_root).exists()
    # B5 sentinel does NOT exist (the B6 wrapper must not have
    # touched it).
    assert not aggregator_failed_sentinel_path(output_root).exists()
    # B7 sentinel does NOT exist.
    assert not training_time_aggregator_failed_sentinel_path(output_root).exists()
