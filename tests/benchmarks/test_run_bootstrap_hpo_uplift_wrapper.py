"""Phase B15 D-B13.3 CLI-wrapper tests for `_run_bootstrap_hpo_uplift_rollup`.

Pins the wrapper's full contract:
- Gate A: opt-out flag
- Gate B: run_manifest absent
- Gate C: run_manifest corrupt
- Gate D: manifest exists but variant=tuned absent
- success path writes shard
- RawRollupError caught + sentinel written
- stale sentinel unlinked on success
- cross-report sentinel isolation (B5, B6, B7 pre-planted)
- no_paired_cells sentinel propagation through the wrapper
"""

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.bootstrap_manifest import (
    aggregator_failed_sentinel_path,
    hpo_uplift_aggregator_failed_sentinel_path,
    hpo_uplift_rollup_path,
    load_hpo_uplift_rollup,
    pairwise_aggregator_failed_sentinel_path,
    training_time_aggregator_failed_sentinel_path,
)
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments import build_run_environment
from benchmarks.run import _run_bootstrap_hpo_uplift_rollup
from benchmarks.run_manifest import (
    build_run_manifest,
    run_manifest_path,
    write_run_manifest,
)

from tests.benchmarks._fakes import register_all_fakes_and_get_panels


def _build_config(
    *,
    tmp_path: Path,
    bootstrap_hpo_uplift_enabled: bool = True,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(
            ExperimentSpec(kind="raw_loss", seeds=(0,)),
            ExperimentSpec(
                kind="hpo_uplift",
                seeds=(0,),
                bootstrap_hpo_uplift_enabled=bootstrap_hpo_uplift_enabled,
            ),
        ),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )


def _make_run_manifest(config: BenchmarkConfig, output_root: Path) -> None:
    register_all_fakes_and_get_panels()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        config=config,
        run_id="b15-wrapper-test",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)


def _manifest_row(
    *,
    seed: int = 0,
    fold_index: int = 0,
    variant: str = "default",
    log_loss: float | None = 0.50,
    skipped_reason: str | None = None,
) -> dict[str, object]:
    return {
        "dataset_name": "fake_binary",
        "model_name": "fake_constant_binary",
        "task_type": "binary",
        "seed": seed,
        "fold_index": fold_index,
        "variant": variant,
        "log_loss": log_loss,
        "skipped_reason": skipped_reason,
    }


def _stub_load_run(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]) -> None:
    import benchmarks.report.bootstrap_hpo_uplift as _module

    df = pd.DataFrame(rows)
    monkeypatch.setattr(_module, "load_run", lambda _root: df)


def _happy_rows() -> list[dict[str, object]]:
    """Minimal happy-path B5 manifest: 2 seeds x 1 fold paired."""
    rows: list[dict[str, object]] = []
    for s in (0, 1):
        rows.append(_manifest_row(seed=s, fold_index=0, variant="default", log_loss=0.50))
        rows.append(_manifest_row(seed=s, fold_index=0, variant="tuned", log_loss=0.30))
    return rows


# --- 1. Happy path ---------------------------------------------------------


def test_run_bootstrap_hpo_uplift_rollup_writes_shard_on_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    _stub_load_run(monkeypatch, _happy_rows())

    env = build_run_environment(profile="smoke")
    _run_bootstrap_hpo_uplift_rollup(config, env=env, output_root=output_root)
    assert hpo_uplift_rollup_path(output_root).exists()


# --- 2. Gate A: opt-out -----------------------------------------------------


def test_run_bootstrap_hpo_uplift_rollup_skips_via_opt_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path=tmp_path, bootstrap_hpo_uplift_enabled=False)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    _stub_load_run(monkeypatch, _happy_rows())

    env = build_run_environment(profile="smoke")
    _run_bootstrap_hpo_uplift_rollup(config, env=env, output_root=output_root)
    assert not hpo_uplift_rollup_path(output_root).exists()
    assert not hpo_uplift_aggregator_failed_sentinel_path(output_root).exists()


# --- 3. B5 manifest absent (file-level) ------------------------------------


def test_run_bootstrap_hpo_uplift_rollup_skips_when_b5_manifest_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the B5 manifest is empty (no rows), the aggregator
    returns [] and the wrapper writes a clean empty shard. No
    failure sentinel."""
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    _stub_load_run(monkeypatch, [])

    env = build_run_environment(profile="smoke")
    _run_bootstrap_hpo_uplift_rollup(config, env=env, output_root=output_root)
    assert not hpo_uplift_aggregator_failed_sentinel_path(output_root).exists()


# --- 4. Gate D: manifest exists but variant=tuned absent ------------------


def test_run_bootstrap_hpo_uplift_rollup_skips_when_no_tuned_variant_rows_in_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manifest has only variant=default rows (B8 was never run)
    -> wrapper skips cleanly. NO rollup file written. NO failure
    sentinel written."""
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    rows = [
        _manifest_row(seed=0, fold_index=0, variant="default", log_loss=0.50),
        _manifest_row(seed=1, fold_index=0, variant="default", log_loss=0.51),
    ]
    _stub_load_run(monkeypatch, rows)

    env = build_run_environment(profile="smoke")
    _run_bootstrap_hpo_uplift_rollup(config, env=env, output_root=output_root)
    assert not hpo_uplift_rollup_path(output_root).exists()
    assert not hpo_uplift_aggregator_failed_sentinel_path(output_root).exists()


# --- 5. run_manifest absent ------------------------------------------------


def test_run_bootstrap_hpo_uplift_rollup_skips_when_run_manifest_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    # No run_manifest written.
    assert not run_manifest_path(output_root).exists()
    _stub_load_run(monkeypatch, _happy_rows())

    env = build_run_environment(profile="smoke")
    _run_bootstrap_hpo_uplift_rollup(config, env=env, output_root=output_root)
    assert not hpo_uplift_rollup_path(output_root).exists()


# --- 6. run_manifest load fails --------------------------------------------


def test_run_bootstrap_hpo_uplift_rollup_skips_when_run_manifest_load_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_manifest_path(output_root).write_bytes(b"{not valid json")
    _stub_load_run(monkeypatch, _happy_rows())

    env = build_run_environment(profile="smoke")
    _run_bootstrap_hpo_uplift_rollup(config, env=env, output_root=output_root)
    assert not hpo_uplift_rollup_path(output_root).exists()


# --- 7. RawRollupError caught + continues ----------------------------------


def test_run_bootstrap_hpo_uplift_rollup_catches_raw_rollup_error_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Monkeypatch the aggregator to raise RawRollupError; assert
    the wrapper catches it, deletes any partial output, does NOT
    propagate."""
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)

    # Pre-create a stale partial rollup.
    hpo_uplift_rollup_path(output_root).write_bytes(b"stale partial")

    from benchmarks.report.bootstrap_rollup import RawRollupError as _Err

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise _Err("simulated hpo_uplift aggregator failure")

    import benchmarks.run as _run_module

    monkeypatch.setattr(_run_module, "aggregate_bootstrap_hpo_uplift_rollup", _boom)
    _run_bootstrap_hpo_uplift_rollup(
        config, env=build_run_environment(profile="smoke"), output_root=output_root
    )
    assert not hpo_uplift_rollup_path(output_root).exists()


# --- 8. Failure sentinel written (dedicated pin) ---------------------------


def test_run_bootstrap_hpo_uplift_rollup_writes_failure_sentinel_on_raw_rollup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)

    from benchmarks.report.bootstrap_rollup import RawRollupError as _Err

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise _Err("simulated failure")

    import benchmarks.run as _run_module

    monkeypatch.setattr(_run_module, "aggregate_bootstrap_hpo_uplift_rollup", _boom)
    _run_bootstrap_hpo_uplift_rollup(
        config, env=build_run_environment(profile="smoke"), output_root=output_root
    )
    sentinel = hpo_uplift_aggregator_failed_sentinel_path(output_root)
    assert sentinel.exists()
    assert sentinel.read_text(encoding="utf-8").strip() == "RawRollupError"


# --- 9. Stale sentinel unlinked on success --------------------------------


def test_run_bootstrap_hpo_uplift_rollup_unlinks_stale_failure_sentinel_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    _stub_load_run(monkeypatch, _happy_rows())

    sentinel = hpo_uplift_aggregator_failed_sentinel_path(output_root)
    sentinel.write_text("RawRollupError", encoding="utf-8")
    assert sentinel.exists()

    env = build_run_environment(profile="smoke")
    _run_bootstrap_hpo_uplift_rollup(config, env=env, output_root=output_root)
    assert not sentinel.exists()
    assert hpo_uplift_rollup_path(output_root).exists()


# --- 10. Cross-report sentinel isolation (qa-C3 B8 direction) -------------


def test_run_bootstrap_hpo_uplift_rollup_failure_does_not_touch_b5_b6_b7_sentinels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-plant ALL THREE other sentinels (B5, B6, B7) with
    distinct content strings; assert all three files exist with
    UNCHANGED content after the B15 failure (B14 Stage-3 qa-I3
    closure pattern: read_text comparison, not just exists)."""
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)

    b5_sentinel = aggregator_failed_sentinel_path(output_root)
    b6_sentinel = pairwise_aggregator_failed_sentinel_path(output_root)
    b7_sentinel = training_time_aggregator_failed_sentinel_path(output_root)
    b5_sentinel.write_text("PrevB5Failure", encoding="utf-8")
    b6_sentinel.write_text("PrevB6Failure", encoding="utf-8")
    b7_sentinel.write_text("PrevB7Failure", encoding="utf-8")

    from benchmarks.report.bootstrap_rollup import RawRollupError as _Err

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise _Err("simulated failure")

    import benchmarks.run as _run_module

    monkeypatch.setattr(_run_module, "aggregate_bootstrap_hpo_uplift_rollup", _boom)
    _run_bootstrap_hpo_uplift_rollup(
        config, env=build_run_environment(profile="smoke"), output_root=output_root
    )

    assert hpo_uplift_aggregator_failed_sentinel_path(output_root).exists()
    # B5/B6/B7 sentinels untouched.
    assert b5_sentinel.exists()
    assert b5_sentinel.read_text(encoding="utf-8").strip() == "PrevB5Failure"
    assert b6_sentinel.exists()
    assert b6_sentinel.read_text(encoding="utf-8").strip() == "PrevB6Failure"
    assert b7_sentinel.exists()
    assert b7_sentinel.read_text(encoding="utf-8").strip() == "PrevB7Failure"


# --- 11. no_paired_cells sentinel propagation through the wrapper ---------


def test_run_bootstrap_hpo_uplift_rollup_writes_no_paired_cells_sentinel_row_in_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: aggregator emits no_paired_cells sentinel ->
    wrapper writes shard containing that row -> NO failure
    sentinel (R3 qa-I2 closure)."""
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    # Default at fold=0, tuned at fold=1 -> inner-join yields 0.
    rows = [
        _manifest_row(seed=0, fold_index=0, variant="default", log_loss=0.50),
        _manifest_row(seed=0, fold_index=1, variant="tuned", log_loss=0.30),
    ]
    _stub_load_run(monkeypatch, rows)

    env = build_run_environment(profile="smoke")
    _run_bootstrap_hpo_uplift_rollup(config, env=env, output_root=output_root)

    assert hpo_uplift_rollup_path(output_root).exists()
    assert not hpo_uplift_aggregator_failed_sentinel_path(output_root).exists()
    loaded = load_hpo_uplift_rollup(output_root)
    assert any(r.bootstrap_skipped_reason == "no_paired_cells" for r in loaded)
