"""Phase B14 D-B13.2 CLI-wrapper tests for `_run_bootstrap_training_time_rollup`.

Mirrors the pairwise wrapper tests against the training-time
sentinel. Pins the success path, opt-out, missing/corrupt
run_manifest, RawRollupError catch + sentinel write, stale-
sentinel unlink, and the cross-report sentinel isolation
(Round-1 qa-C3 closure: B7 failure must not touch B5 or B6
sentinels).
"""

from pathlib import Path

import pytest
from benchmarks.bootstrap_manifest import (
    aggregator_failed_sentinel_path,
    pairwise_aggregator_failed_sentinel_path,
    training_time_aggregator_failed_sentinel_path,
    training_time_rollup_path,
)
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments import build_run_environment, run_raw_loss
from benchmarks.run import _run_bootstrap_training_time_rollup
from benchmarks.run_manifest import (
    build_run_manifest,
    run_manifest_path,
    write_run_manifest,
)

from tests.benchmarks._fakes import register_all_fakes_and_get_panels


def _build_config(
    *,
    tmp_path: Path,
    bootstrap_training_time_enabled: bool = True,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(
            ExperimentSpec(kind="raw_loss", seeds=(0,)),
            ExperimentSpec(
                kind="training_time",
                seeds=(0,),
                bootstrap_training_time_enabled=bootstrap_training_time_enabled,
            ),
        ),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )


def _run_b5_and_manifest(tmp_path: Path, config: BenchmarkConfig) -> Path:
    register_all_fakes_and_get_panels()
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_raw_loss(config, output_root=output_root, env=env)
    manifest = build_run_manifest(
        config=config,
        run_id="b14-tt-wrapper-test",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)
    return output_root


# --- Happy path -------------------------------------------------------------


def test_run_bootstrap_training_time_rollup_writes_shard_on_happy_path(
    tmp_path: Path,
) -> None:
    """Default config + a B5 manifest -> rollup parquet written."""
    config = _build_config(tmp_path=tmp_path)
    output_root = _run_b5_and_manifest(tmp_path, config)
    env = build_run_environment(profile="smoke")
    _run_bootstrap_training_time_rollup(config, env=env, output_root=output_root)
    assert training_time_rollup_path(output_root).exists()


# --- Opt-out skip -----------------------------------------------------------


def test_run_bootstrap_training_time_rollup_skips_via_opt_out(
    tmp_path: Path,
) -> None:
    """bootstrap_training_time_enabled=False -> wrapper skips."""
    config = _build_config(tmp_path=tmp_path, bootstrap_training_time_enabled=False)
    output_root = _run_b5_and_manifest(tmp_path, config)
    env = build_run_environment(profile="smoke")
    _run_bootstrap_training_time_rollup(config, env=env, output_root=output_root)
    assert not training_time_rollup_path(output_root).exists()


# --- Missing B5 manifest ----------------------------------------------------


def test_run_bootstrap_training_time_rollup_skips_when_b5_manifest_absent(
    tmp_path: Path,
) -> None:
    """When the B5 manifest is absent (run_raw_loss never wrote
    one), the aggregator returns [] and the wrapper returns
    cleanly with no rollup file."""
    register_all_fakes_and_get_panels()
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    # Write only the run_manifest; do NOT run raw_loss.
    manifest = build_run_manifest(
        config=config,
        run_id="b14-no-b5",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)

    env = build_run_environment(profile="smoke")
    _run_bootstrap_training_time_rollup(config, env=env, output_root=output_root)
    assert not training_time_rollup_path(output_root).exists()
    assert not training_time_aggregator_failed_sentinel_path(output_root).exists()


# --- Missing run_manifest ---------------------------------------------------


def test_run_bootstrap_training_time_rollup_skips_when_run_manifest_absent(
    tmp_path: Path,
) -> None:
    """No run_manifest.json -> wrapper logs and returns."""
    register_all_fakes_and_get_panels()
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    env = build_run_environment(profile="smoke")
    run_raw_loss(config, output_root=output_root, env=env)
    # Do NOT write the run_manifest.
    assert not run_manifest_path(output_root).exists()

    _run_bootstrap_training_time_rollup(config, env=env, output_root=output_root)
    assert not training_time_rollup_path(output_root).exists()


# --- Corrupt run_manifest ---------------------------------------------------


def test_run_bootstrap_training_time_rollup_skips_when_run_manifest_load_fails(
    tmp_path: Path,
) -> None:
    """Corrupt run_manifest.json -> wrapper catches and returns."""
    register_all_fakes_and_get_panels()
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    env = build_run_environment(profile="smoke")
    run_raw_loss(config, output_root=output_root, env=env)
    run_manifest_path(output_root).write_bytes(b"{not valid json")

    _run_bootstrap_training_time_rollup(config, env=env, output_root=output_root)
    assert not training_time_rollup_path(output_root).exists()


# --- RawRollupError caught + sentinel deletion -----------------------------


def test_run_bootstrap_training_time_rollup_catches_raw_rollup_error_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Monkeypatch the aggregator to raise RawRollupError; assert
    the wrapper catches it, deletes any partial output, and does
    NOT propagate."""
    config = _build_config(tmp_path=tmp_path)
    output_root = _run_b5_and_manifest(tmp_path, config)

    # Pre-create a stale partial rollup file.
    training_time_rollup_path(output_root).write_bytes(b"stale partial")
    assert training_time_rollup_path(output_root).exists()

    from benchmarks.report.bootstrap_rollup import RawRollupError as _Err

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise _Err("simulated training_time aggregator failure")

    import benchmarks.run as _run_module

    monkeypatch.setattr(_run_module, "aggregate_bootstrap_training_time_rollup", _boom)
    _run_bootstrap_training_time_rollup(
        config, env=build_run_environment(profile="smoke"), output_root=output_root
    )
    # Partial output deleted.
    assert not training_time_rollup_path(output_root).exists()


# --- Sentinel write contract (dedicated pin) --------------------------------


def test_run_bootstrap_training_time_rollup_writes_failure_sentinel_on_raw_rollup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dedicated pin for the sentinel-write contract independent
    from the catch-and-continue contract."""
    config = _build_config(tmp_path=tmp_path)
    output_root = _run_b5_and_manifest(tmp_path, config)

    from benchmarks.report.bootstrap_rollup import RawRollupError as _Err

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise _Err("simulated aggregator failure")

    import benchmarks.run as _run_module

    monkeypatch.setattr(_run_module, "aggregate_bootstrap_training_time_rollup", _boom)
    _run_bootstrap_training_time_rollup(
        config, env=build_run_environment(profile="smoke"), output_root=output_root
    )
    sentinel = training_time_aggregator_failed_sentinel_path(output_root)
    assert sentinel.exists()
    assert sentinel.read_text(encoding="utf-8").strip() == "RawRollupError"


# --- Stale sentinel unlink on success ---------------------------------------


def test_run_bootstrap_training_time_rollup_unlinks_stale_failure_sentinel_on_success(
    tmp_path: Path,
) -> None:
    """A stale sentinel from a prior failed run is unlinked when
    the next aggregate call succeeds."""
    config = _build_config(tmp_path=tmp_path)
    output_root = _run_b5_and_manifest(tmp_path, config)

    sentinel = training_time_aggregator_failed_sentinel_path(output_root)
    sentinel.write_text("RawRollupError", encoding="utf-8")
    assert sentinel.exists()

    env = build_run_environment(profile="smoke")
    _run_bootstrap_training_time_rollup(config, env=env, output_root=output_root)
    assert not sentinel.exists()
    assert training_time_rollup_path(output_root).exists()
    assert training_time_rollup_path(output_root).stat().st_size > 0


# --- Cross-report sentinel isolation (Round-1 qa-C3 B7 direction) ---------


def test_run_bootstrap_training_time_rollup_failure_does_not_touch_b5_or_b6_sentinels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-1 qa-C3 B7-direction closure: a B7 (training-time)
    aggregator failure must write
    `bootstrap_training_time_aggregator_failed.txt` and leave the
    B5 (`bootstrap_aggregator_failed.txt`) and B6
    (`bootstrap_pairwise_aggregator_failed.txt`) sentinels
    untouched."""
    config = _build_config(tmp_path=tmp_path)
    output_root = _run_b5_and_manifest(tmp_path, config)

    from benchmarks.report.bootstrap_rollup import RawRollupError as _Err

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise _Err("simulated training_time aggregator failure")

    import benchmarks.run as _run_module

    monkeypatch.setattr(_run_module, "aggregate_bootstrap_training_time_rollup", _boom)
    _run_bootstrap_training_time_rollup(
        config, env=build_run_environment(profile="smoke"), output_root=output_root
    )
    # B7 sentinel exists.
    assert training_time_aggregator_failed_sentinel_path(output_root).exists()
    # B5 and B6 sentinels do NOT exist.
    assert not aggregator_failed_sentinel_path(output_root).exists()
    assert not pairwise_aggregator_failed_sentinel_path(output_root).exists()
