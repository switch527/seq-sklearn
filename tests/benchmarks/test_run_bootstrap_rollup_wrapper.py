"""Phase B13 CLI wrapper tests for _run_bootstrap_rollup.

Pins the wrapper's three contract pieces:
1. Catches RawRollupError and deletes any partial output.
2. Returns silently (the run as a whole still exits 0).
3. The leaderboard then falls back to the std variant with the
   "Bootstrap aggregator failed" footnote.

Plus the opt-out flag test (qa-C4): when
`ExperimentSpec(bootstrap_rollup_enabled=False)`, the wrapper
skips the aggregator and no rollup file lands.
"""

from pathlib import Path

import benchmarks.adapters  # type: ignore[reportUnusedImport]  # noqa: F401
import pytest
from benchmarks.bootstrap_manifest import rollup_path
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments import build_run_environment, run_raw_loss
from benchmarks.run import _run_bootstrap_rollup

from tests.benchmarks._fakes import register_all_fakes_and_get_panels


def _build_config(
    *,
    tmp_path: Path,
    bootstrap_rollup_enabled: bool = True,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(
            ExperimentSpec(
                kind="raw_loss",
                seeds=(0,),
                bootstrap_rollup_enabled=bootstrap_rollup_enabled,
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
    # Write a run_manifest.json so the wrapper can load it.
    from benchmarks.run_manifest import build_run_manifest, write_run_manifest

    manifest = build_run_manifest(
        config=config,
        run_id="test-run",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)
    return output_root


def test_run_bootstrap_rollup_writes_shard_on_happy_path(tmp_path: Path) -> None:
    """Default config (`bootstrap_rollup_enabled=True`) writes
    `bootstrap_rollup.parquet` after the raw_loss step."""
    config = _build_config(tmp_path=tmp_path)
    output_root = _run_b5_and_manifest(tmp_path, config)
    env = build_run_environment(profile="smoke")
    _run_bootstrap_rollup(config, env=env, output_root=output_root)
    assert rollup_path(output_root).exists()


def test_skip_bootstrap_rollup_via_experiment_spec_does_not_write_rollup_file(
    tmp_path: Path,
) -> None:
    """qa-C4: ExperimentSpec(bootstrap_rollup_enabled=False)
    opts out; the wrapper skips entirely and NO rollup file
    lands."""
    config = _build_config(tmp_path=tmp_path, bootstrap_rollup_enabled=False)
    output_root = _run_b5_and_manifest(tmp_path, config)
    env = build_run_environment(profile="smoke")
    _run_bootstrap_rollup(config, env=env, output_root=output_root)
    assert not rollup_path(output_root).exists()


def test_run_bootstrap_rollup_cli_wrapper_catches_raw_rollup_error_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R5 arch-C1 fix: monkey-patch the aggregator to raise
    `RawRollupError`; assert the wrapper catches it, deletes
    any partial output, and does NOT propagate the exception
    (so the CLI continues with exit 0 and the leaderboard
    falls back to std variant)."""
    config = _build_config(tmp_path=tmp_path)
    output_root = _run_b5_and_manifest(tmp_path, config)
    env = build_run_environment(profile="smoke")

    # Pre-create a stale rollup file so we can verify the
    # wrapper deletes it on RawRollupError.
    output_root.mkdir(parents=True, exist_ok=True)
    rollup_path(output_root).write_bytes(b"stale partial content")
    assert rollup_path(output_root).exists()

    from benchmarks.report.bootstrap_rollup import RawRollupError as _Err

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise _Err("simulated aggregator failure")

    # Monkey-patch the binding in `benchmarks.run` (where
    # `_run_bootstrap_rollup` looks up `aggregate_bootstrap_rollup`),
    # not the source module, because the import happens at module
    # load time and the local name is what the wrapper calls.
    import benchmarks.run as _run_module

    monkeypatch.setattr(_run_module, "aggregate_bootstrap_rollup", _boom)
    # Must NOT raise.
    _run_bootstrap_rollup(config, env=env, output_root=output_root)
    # Partial output deleted.
    assert not rollup_path(output_root).exists()


def test_run_bootstrap_rollup_skips_silently_when_run_manifest_absent(
    tmp_path: Path,
) -> None:
    """When run_manifest.json is missing (e.g. caller forgot to
    write it before calling _run_bootstrap_rollup), the wrapper
    logs a warning and returns; no rollup is written."""
    register_all_fakes_and_get_panels()
    config = _build_config(tmp_path=tmp_path)
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_raw_loss(config, output_root=output_root, env=env)
    # NO run_manifest.json written.
    _run_bootstrap_rollup(config, env=env, output_root=output_root)
    assert not rollup_path(output_root).exists()


def test_run_bootstrap_rollup_skips_silently_when_run_manifest_load_fails(
    tmp_path: Path,
) -> None:
    """qa-I2: when `load_run_manifest` raises (e.g., corrupt JSON),
    the wrapper logs + returns; no rollup is written and no
    exception propagates."""
    register_all_fakes_and_get_panels()
    config = _build_config(tmp_path=tmp_path)
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_raw_loss(config, output_root=output_root, env=env)
    # Write a corrupt run_manifest.json so the load step raises.
    from benchmarks.run_manifest import run_manifest_path

    run_manifest_path(output_root).write_bytes(b"{not valid json")
    _run_bootstrap_rollup(config, env=env, output_root=output_root)
    assert not rollup_path(output_root).exists()


def test_run_bootstrap_rollup_writes_failure_sentinel_on_raw_rollup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1 code-I1: when the aggregator raises RawRollupError, the
    wrapper drops `bootstrap_aggregator_failed.txt` so
    `render_from_dir` surfaces the "Bootstrap aggregator failed"
    footnote in the std fallback (wire-up the renderer's
    `aggregator_error_class` path that was dead in CLI before)."""
    config = _build_config(tmp_path=tmp_path)
    output_root = _run_b5_and_manifest(tmp_path, config)
    env = build_run_environment(profile="smoke")

    from benchmarks.report.bootstrap_rollup import RawRollupError as _Err

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise _Err("simulated aggregator failure")

    import benchmarks.run as _run_module

    monkeypatch.setattr(_run_module, "aggregate_bootstrap_rollup", _boom)
    _run_bootstrap_rollup(config, env=env, output_root=output_root)
    sentinel = output_root / "bootstrap_aggregator_failed.txt"
    assert sentinel.exists()
    assert sentinel.read_text(encoding="utf-8").strip() == "RawRollupError"


def test_run_bootstrap_rollup_unlinks_stale_failure_sentinel_on_success(
    tmp_path: Path,
) -> None:
    """R2 arch-C1 close: a stale `bootstrap_aggregator_failed.txt`
    from a prior failed run is unlinked when the next aggregate
    call succeeds; otherwise the renderer would surface the std +
    failure footnote indefinitely."""
    config = _build_config(tmp_path=tmp_path)
    output_root = _run_b5_and_manifest(tmp_path, config)
    env = build_run_environment(profile="smoke")
    # Pre-create the stale sentinel.
    sentinel = output_root / "bootstrap_aggregator_failed.txt"
    sentinel.write_text("RawRollupError", encoding="utf-8")
    assert sentinel.exists()
    _run_bootstrap_rollup(config, env=env, output_root=output_root)
    # Sentinel removed; rollup written and non-empty (a zero-byte
    # parquet would be a regression on the write contract).
    assert not sentinel.exists()
    assert rollup_path(output_root).exists()
    assert rollup_path(output_root).stat().st_size > 0
