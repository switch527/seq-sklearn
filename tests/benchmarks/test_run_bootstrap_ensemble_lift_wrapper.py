"""Phase B16 D-B13.4 CLI-wrapper tests for `_run_bootstrap_ensemble_lift_rollup`.

Pins the wrapper's full contract:
- Gate A: opt-out flag
- Gate B: run_manifest absent
- Gate C: run_manifest corrupt
- Gate D: manifest exists but seq OR baseline family absent
- success path writes shard
- RawRollupError caught + sentinel written
- stale sentinel unlinked on success
- cross-report sentinel isolation (B5, B6, B7, B8 pre-planted)
- no_gbm_predictions / no_seq_predictions sentinel propagation
  through the wrapper (both directions)
"""

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.bootstrap_manifest import (
    aggregator_failed_sentinel_path,
    ensemble_lift_aggregator_failed_sentinel_path,
    ensemble_lift_rollup_path,
    hpo_uplift_aggregator_failed_sentinel_path,
    load_ensemble_lift_rollup,
    pairwise_aggregator_failed_sentinel_path,
    training_time_aggregator_failed_sentinel_path,
)
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments import build_run_environment
from benchmarks.experiments.ensemble_lift import (
    ComputePerCellLiftDeltasResult,
    PerCellLiftDelta,
)
from benchmarks.run import _run_bootstrap_ensemble_lift_rollup
from benchmarks.run_manifest import (
    build_run_manifest,
    run_manifest_path,
    write_run_manifest,
)

from tests.benchmarks._fakes import register_all_fakes_and_get_panels

_GBM_MODEL = "gbm_constant"
_SEQ_MODEL = "seq_constant"


def _build_config(
    *,
    tmp_path: Path,
    bootstrap_ensemble_lift_enabled: bool = True,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(
            ExperimentSpec(kind="raw_loss", seeds=(0,)),
            ExperimentSpec(
                kind="ensemble_lift",
                seeds=(0,),
                bootstrap_ensemble_lift_enabled=bootstrap_ensemble_lift_enabled,
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
        run_id="b16-wrapper-test",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)


def _manifest_row(
    *,
    dataset_name: str = "fake_binary",
    model_name: str = _GBM_MODEL,
    seed: int = 0,
    fold_index: int = 0,
    skipped_reason: str | None = None,
) -> dict[str, object]:
    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "task_type": "binary",
        "seed": seed,
        "fold_index": fold_index,
        "variant": "default",
        "skipped_reason": skipped_reason,
    }


def _stub_load_run(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]) -> None:
    import benchmarks.report.bootstrap_ensemble_lift as _module

    df = pd.DataFrame(rows)
    monkeypatch.setattr(_module, "load_run", lambda _root: df)  # type: ignore[misc]


def _stub_families(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmarks.report.bootstrap_ensemble_lift as _module

    def _families(_df: pd.DataFrame) -> dict[str, str]:
        return {_GBM_MODEL: "gbm", _SEQ_MODEL: "seq_sklearn"}

    monkeypatch.setattr(_module, "model_families", _families)  # type: ignore[misc]


def _stub_per_cell_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub `compute_per_cell_lift_deltas` to return a 2-cell
    finite-loss result so the bootstrap path runs to completion."""
    import benchmarks.report.bootstrap_ensemble_lift as _module

    result = ComputePerCellLiftDeltasResult(
        cells=(
            PerCellLiftDelta(
                seed=0,
                fold_index=0,
                loss_gbm=0.60,
                loss_gbm_plus_seq=0.40,
                delta_loss=0.20,
                oracle_loss=None,
            ),
            PerCellLiftDelta(
                seed=1,
                fold_index=0,
                loss_gbm=0.60,
                loss_gbm_plus_seq=0.40,
                delta_loss=0.20,
                oracle_loss=None,
            ),
        ),
        selector="log_loss",
    )
    monkeypatch.setattr(_module, "compute_per_cell_lift_deltas", lambda **_kwargs: result)  # type: ignore[misc]


def _stub_per_cell_no_gbm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub returning empty cells with seen_no_gbm=True so the
    aggregator emits the `no_gbm_predictions` sentinel."""
    import benchmarks.report.bootstrap_ensemble_lift as _module

    result = ComputePerCellLiftDeltasResult(
        cells=(), seen_no_gbm=True, seen_no_seq=False, selector="log_loss"
    )
    monkeypatch.setattr(_module, "compute_per_cell_lift_deltas", lambda **_kwargs: result)  # type: ignore[misc]


def _stub_per_cell_no_seq(monkeypatch: pytest.MonkeyPatch) -> None:
    """Symmetric mirror: seen_no_seq=True."""
    import benchmarks.report.bootstrap_ensemble_lift as _module

    result = ComputePerCellLiftDeltasResult(
        cells=(), seen_no_gbm=False, seen_no_seq=True, selector="log_loss"
    )
    monkeypatch.setattr(_module, "compute_per_cell_lift_deltas", lambda **_kwargs: result)  # type: ignore[misc]


def _happy_rows() -> list[dict[str, object]]:
    """Minimal happy-path B5 manifest: 2 seeds by both families."""
    rows: list[dict[str, object]] = []
    for s in (0, 1):
        rows.append(_manifest_row(model_name=_GBM_MODEL, seed=s, fold_index=0))
        rows.append(_manifest_row(model_name=_SEQ_MODEL, seed=s, fold_index=0))
    return rows


# --- 1. Happy path ---------------------------------------------------------


def test_run_bootstrap_ensemble_lift_rollup_writes_shard_on_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    _stub_load_run(monkeypatch, _happy_rows())
    _stub_families(monkeypatch)
    _stub_per_cell_happy(monkeypatch)

    env = build_run_environment(profile="smoke")
    _run_bootstrap_ensemble_lift_rollup(config, env=env, output_root=output_root)
    assert ensemble_lift_rollup_path(output_root).exists()
    assert not ensemble_lift_aggregator_failed_sentinel_path(output_root).exists()


# --- 2. Gate A: opt-out ----------------------------------------------------


def test_run_bootstrap_ensemble_lift_rollup_skips_via_opt_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path=tmp_path, bootstrap_ensemble_lift_enabled=False)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    _stub_load_run(monkeypatch, _happy_rows())
    _stub_families(monkeypatch)
    _stub_per_cell_happy(monkeypatch)

    env = build_run_environment(profile="smoke")
    _run_bootstrap_ensemble_lift_rollup(config, env=env, output_root=output_root)
    assert not ensemble_lift_rollup_path(output_root).exists()
    assert not ensemble_lift_aggregator_failed_sentinel_path(output_root).exists()


# --- 3. B5 manifest empty -> empty shard ----------------------------------


def test_run_bootstrap_ensemble_lift_rollup_skips_when_b5_manifest_absent(
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
    _stub_families(monkeypatch)

    env = build_run_environment(profile="smoke")
    _run_bootstrap_ensemble_lift_rollup(config, env=env, output_root=output_root)
    assert not ensemble_lift_aggregator_failed_sentinel_path(output_root).exists()


# --- 4. Gate D: seq family absent from manifest ---------------------------


def test_run_bootstrap_ensemble_lift_rollup_skips_when_seq_family_absent_from_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manifest has only baseline-family rows -> Gate D no-op. NO
    rollup file written. NO failure sentinel."""
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    rows = [_manifest_row(model_name=_GBM_MODEL, seed=0, fold_index=0)]
    _stub_load_run(monkeypatch, rows)
    _stub_families(monkeypatch)

    env = build_run_environment(profile="smoke")
    _run_bootstrap_ensemble_lift_rollup(config, env=env, output_root=output_root)
    assert not ensemble_lift_rollup_path(output_root).exists()
    assert not ensemble_lift_aggregator_failed_sentinel_path(output_root).exists()


# --- 5. Gate D: baseline family absent from manifest (symmetric) ----------


def test_run_bootstrap_ensemble_lift_rollup_skips_when_baseline_family_absent_from_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Symmetric mirror of test 4: only seq-family rows present.
    Closes qa-C4 Gate D SYMMETRIC direction."""
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    rows = [_manifest_row(model_name=_SEQ_MODEL, seed=0, fold_index=0)]
    _stub_load_run(monkeypatch, rows)
    _stub_families(monkeypatch)

    env = build_run_environment(profile="smoke")
    _run_bootstrap_ensemble_lift_rollup(config, env=env, output_root=output_root)
    assert not ensemble_lift_rollup_path(output_root).exists()
    assert not ensemble_lift_aggregator_failed_sentinel_path(output_root).exists()


# --- 6. run_manifest absent (file-level) ----------------------------------


def test_run_bootstrap_ensemble_lift_rollup_skips_when_run_manifest_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    assert not run_manifest_path(output_root).exists()
    _stub_load_run(monkeypatch, _happy_rows())
    _stub_families(monkeypatch)

    env = build_run_environment(profile="smoke")
    _run_bootstrap_ensemble_lift_rollup(config, env=env, output_root=output_root)
    assert not ensemble_lift_rollup_path(output_root).exists()
    # qa-R1-I1: Gate B must NOT write a failure sentinel.
    assert not ensemble_lift_aggregator_failed_sentinel_path(output_root).exists()


# --- 7. run_manifest load fails (corrupt JSON) ----------------------------


def test_run_bootstrap_ensemble_lift_rollup_skips_when_run_manifest_load_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_manifest_path(output_root).write_bytes(b"{not valid json")
    _stub_load_run(monkeypatch, _happy_rows())
    _stub_families(monkeypatch)

    env = build_run_environment(profile="smoke")
    _run_bootstrap_ensemble_lift_rollup(config, env=env, output_root=output_root)
    assert not ensemble_lift_rollup_path(output_root).exists()
    # qa-R1-I1: Gate C must NOT write a failure sentinel.
    assert not ensemble_lift_aggregator_failed_sentinel_path(output_root).exists()


# --- 8. RawRollupError caught + continues ---------------------------------


def test_run_bootstrap_ensemble_lift_rollup_catches_raw_rollup_error_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aggregator raises RawRollupError; wrapper catches, deletes
    any partial output, does not propagate."""
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)

    ensemble_lift_rollup_path(output_root).write_bytes(b"stale partial")

    from benchmarks.report.bootstrap_rollup import RawRollupError as _Err

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise _Err("simulated ensemble_lift aggregator failure")

    import benchmarks.run as _run_module

    monkeypatch.setattr(_run_module, "aggregate_bootstrap_ensemble_lift_rollup", _boom)
    _run_bootstrap_ensemble_lift_rollup(
        config,
        env=build_run_environment(profile="smoke"),
        output_root=output_root,
    )
    assert not ensemble_lift_rollup_path(output_root).exists()


# --- 9. Failure sentinel written -------------------------------------------


def test_run_bootstrap_ensemble_lift_rollup_writes_failure_sentinel_on_raw_rollup_error(
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

    monkeypatch.setattr(_run_module, "aggregate_bootstrap_ensemble_lift_rollup", _boom)
    _run_bootstrap_ensemble_lift_rollup(
        config,
        env=build_run_environment(profile="smoke"),
        output_root=output_root,
    )
    sentinel = ensemble_lift_aggregator_failed_sentinel_path(output_root)
    assert sentinel.exists()
    assert sentinel.read_text(encoding="utf-8").strip() == "RawRollupError"


# --- 10. Stale sentinel unlinked on success -------------------------------


def test_run_bootstrap_ensemble_lift_rollup_unlinks_stale_failure_sentinel_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    _stub_load_run(monkeypatch, _happy_rows())
    _stub_families(monkeypatch)
    _stub_per_cell_happy(monkeypatch)

    sentinel = ensemble_lift_aggregator_failed_sentinel_path(output_root)
    sentinel.write_text("RawRollupError", encoding="utf-8")
    assert sentinel.exists()

    env = build_run_environment(profile="smoke")
    _run_bootstrap_ensemble_lift_rollup(config, env=env, output_root=output_root)
    assert not sentinel.exists()
    assert ensemble_lift_rollup_path(output_root).exists()


# --- 11. Cross-report sentinel isolation (B5/B6/B7/B8) --------------------


def test_run_bootstrap_ensemble_lift_rollup_failure_does_not_touch_other_sentinels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-plant ALL FOUR other sentinels (B5, B6, B7, B8) with
    distinct content strings; assert all four files exist with
    UNCHANGED content after the B16 failure (B14 Stage-3 qa-I3
    closure pattern). arch-N3 name shortening so a future B17
    doesn't need a rename."""
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)

    b5_sentinel = aggregator_failed_sentinel_path(output_root)
    b6_sentinel = pairwise_aggregator_failed_sentinel_path(output_root)
    b7_sentinel = training_time_aggregator_failed_sentinel_path(output_root)
    b8_sentinel = hpo_uplift_aggregator_failed_sentinel_path(output_root)
    b5_sentinel.write_text("PrevB5Failure", encoding="utf-8")
    b6_sentinel.write_text("PrevB6Failure", encoding="utf-8")
    b7_sentinel.write_text("PrevB7Failure", encoding="utf-8")
    b8_sentinel.write_text("PrevB8Failure", encoding="utf-8")

    from benchmarks.report.bootstrap_rollup import RawRollupError as _Err

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise _Err("simulated failure")

    import benchmarks.run as _run_module

    monkeypatch.setattr(_run_module, "aggregate_bootstrap_ensemble_lift_rollup", _boom)
    _run_bootstrap_ensemble_lift_rollup(
        config,
        env=build_run_environment(profile="smoke"),
        output_root=output_root,
    )

    assert ensemble_lift_aggregator_failed_sentinel_path(output_root).exists()
    assert b5_sentinel.read_text(encoding="utf-8").strip() == "PrevB5Failure"
    assert b6_sentinel.read_text(encoding="utf-8").strip() == "PrevB6Failure"
    assert b7_sentinel.read_text(encoding="utf-8").strip() == "PrevB7Failure"
    assert b8_sentinel.read_text(encoding="utf-8").strip() == "PrevB8Failure"


# --- 12. no_gbm_predictions sentinel propagation through wrapper ----------


def test_run_bootstrap_ensemble_lift_rollup_writes_no_gbm_predictions_sentinel_row_in_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: aggregator emits `no_gbm_predictions` sentinel
    -> wrapper writes shard containing that row -> NO failure
    sentinel."""
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    _stub_load_run(monkeypatch, _happy_rows())
    _stub_families(monkeypatch)
    _stub_per_cell_no_gbm(monkeypatch)

    env = build_run_environment(profile="smoke")
    _run_bootstrap_ensemble_lift_rollup(config, env=env, output_root=output_root)

    assert ensemble_lift_rollup_path(output_root).exists()
    assert not ensemble_lift_aggregator_failed_sentinel_path(output_root).exists()
    loaded = load_ensemble_lift_rollup(output_root)
    assert any(r.bootstrap_skipped_reason == "no_gbm_predictions" for r in loaded)


# --- 13. no_seq_predictions sentinel propagation (symmetric mirror) -------


def test_run_bootstrap_ensemble_lift_rollup_writes_no_seq_predictions_sentinel_row_in_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Symmetric mirror: aggregator emits `no_seq_predictions`
    sentinel -> shard contains that row. Closes R2 qa-I2."""
    config = _build_config(tmp_path=tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    _stub_load_run(monkeypatch, _happy_rows())
    _stub_families(monkeypatch)
    _stub_per_cell_no_seq(monkeypatch)

    env = build_run_environment(profile="smoke")
    _run_bootstrap_ensemble_lift_rollup(config, env=env, output_root=output_root)

    assert ensemble_lift_rollup_path(output_root).exists()
    assert not ensemble_lift_aggregator_failed_sentinel_path(output_root).exists()
    loaded = load_ensemble_lift_rollup(output_root)
    assert any(r.bootstrap_skipped_reason == "no_seq_predictions" for r in loaded)
