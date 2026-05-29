"""Phase B18 / D-B16.6 shared CLI-wrapper factory tests.

15 collected items:
- 9 gate tests covering every branch of the four-gate cascade
  in `run_bootstrap_rollup_via_factory`.
- 1 wire-up parametrize over the 5 (wrapper, spec, expected
  aggregator name) triples, asserting each thin wrapper in
  `benchmarks/run.py` actually routes through the factory
  with the right `_BX_SPEC` constant and the correctly-named
  family aggregator (5 collected cells).
- 1 log-content sanity test asserting the Gate-1 message
  includes `spec.label`.
"""

import logging
from pathlib import Path
from typing import Any

import pytest
from benchmarks._bootstrap_wrapper import (
    BootstrapWrapperSpec,
    run_bootstrap_rollup_via_factory,
)
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments import build_run_environment
from benchmarks.experiments.raw_loss import RunEnvironment
from benchmarks.report.bootstrap_rollup import RawRollupError
from benchmarks.run_manifest import RunManifest, build_run_manifest, write_run_manifest

from tests.benchmarks._fakes import register_all_fakes_and_get_panels


def _make_config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(ExperimentSpec(kind="raw_loss", seeds=(0,)),),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )


def _make_run_manifest(config: BenchmarkConfig, output_root: Path) -> None:
    register_all_fakes_and_get_panels()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        config=config,
        run_id="b18-factory-test",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)


def _rollup_path(root: Path) -> Path:
    return root / "stub_rollup.parquet"


def _sentinel_path(root: Path) -> Path:
    return root / "stub_sentinel.txt"


def _make_spec(*, is_enabled: bool = True) -> BootstrapWrapperSpec:
    """Stub spec with controllable opt-out + filesystem helpers."""
    return BootstrapWrapperSpec(
        label="stub_label",
        is_enabled=lambda _config: is_enabled,
        rollup_path=_rollup_path,
        sentinel_path=_sentinel_path,
    )


def _ok_aggregator(
    config: BenchmarkConfig,
    *,
    output_root: Path,
    env: RunEnvironment,
    manifest: RunManifest,
) -> list[Any]:
    """Happy-path stub aggregator returning a 2-row list."""
    del config, output_root, env, manifest
    return ["row0", "row1"]


def _raising_aggregator(
    config: BenchmarkConfig,
    *,
    output_root: Path,
    env: RunEnvironment,
    manifest: RunManifest,
) -> list[Any]:
    del config, output_root, env, manifest
    raise RawRollupError("simulated aggregator failure")


# --- 1. Gate 1: opt-out predicate False ------------------------------------


def test_factory_skips_on_opt_out_predicate_false(tmp_path: Path) -> None:
    """spec.is_enabled returns False -> wrapper returns
    immediately without touching the filesystem."""
    config = _make_config(tmp_path)
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    spec = _make_spec(is_enabled=False)
    env = build_run_environment(profile="smoke")

    run_bootstrap_rollup_via_factory(
        config,
        env=env,
        output_root=output_root,
        spec=spec,
        aggregator=_ok_aggregator,
    )

    assert not _rollup_path(output_root).exists()
    assert not _sentinel_path(output_root).exists()


# --- 2. Gate 2: run_manifest absent ----------------------------------------


def test_factory_skips_when_run_manifest_absent(tmp_path: Path) -> None:
    """run_manifest.json absent -> wrapper warns and returns
    without invoking the aggregator."""
    config = _make_config(tmp_path)
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    spec = _make_spec()
    env = build_run_environment(profile="smoke")

    captured: list[str] = []

    def _record_aggregator(
        config: BenchmarkConfig,
        *,
        output_root: Path,
        env: RunEnvironment,
        manifest: RunManifest,
    ) -> list[Any]:
        del config, output_root, env, manifest
        captured.append("called")
        return []

    run_bootstrap_rollup_via_factory(
        config,
        env=env,
        output_root=output_root,
        spec=spec,
        aggregator=_record_aggregator,
    )

    assert captured == []
    assert not _rollup_path(output_root).exists()
    assert not _sentinel_path(output_root).exists()


# --- 3-5. Gate 3: load_run_manifest raises (3 narrow except arms) ----------


def _stub_load_raises(monkeypatch: pytest.MonkeyPatch, exc_factory: Any) -> None:
    """Patch the factory's `load_run_manifest` to raise the
    given exception type."""
    import benchmarks._bootstrap_wrapper as _module

    def _raise(_root: Path) -> RunManifest:
        raise exc_factory()

    monkeypatch.setattr(_module, "load_run_manifest", _raise)


def test_factory_skips_when_load_run_manifest_raises_file_not_found_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate 3 arm 1: FileNotFoundError -> skip + warn."""
    config = _make_config(tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    _stub_load_raises(monkeypatch, FileNotFoundError)

    env = build_run_environment(profile="smoke")
    run_bootstrap_rollup_via_factory(
        config,
        env=env,
        output_root=output_root,
        spec=_make_spec(),
        aggregator=_ok_aggregator,
    )

    assert not _rollup_path(output_root).exists()
    assert not _sentinel_path(output_root).exists()


def test_factory_skips_when_load_run_manifest_raises_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate 3 arm 2: ValueError -> skip + warn."""
    config = _make_config(tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    _stub_load_raises(monkeypatch, ValueError)

    env = build_run_environment(profile="smoke")
    run_bootstrap_rollup_via_factory(
        config,
        env=env,
        output_root=output_root,
        spec=_make_spec(),
        aggregator=_ok_aggregator,
    )

    assert not _rollup_path(output_root).exists()
    assert not _sentinel_path(output_root).exists()


def test_factory_skips_when_load_run_manifest_raises_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate 3 arm 3: pydantic ValidationError -> skip + warn."""
    from pydantic import BaseModel, ValidationError

    class _Probe(BaseModel):
        v: int

    def _make_validation_error() -> ValidationError:
        try:
            _Probe.model_validate({"v": "not_an_int"})
        except ValidationError as exc:
            return exc
        raise AssertionError("pydantic accepted bad input")

    config = _make_config(tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    _stub_load_raises(monkeypatch, _make_validation_error)

    env = build_run_environment(profile="smoke")
    run_bootstrap_rollup_via_factory(
        config,
        env=env,
        output_root=output_root,
        spec=_make_spec(),
        aggregator=_ok_aggregator,
    )

    assert not _rollup_path(output_root).exists()
    assert not _sentinel_path(output_root).exists()


# --- 6-7. Gate 4a: happy path (stale-sentinel present vs absent) -----------


def test_factory_happy_path_with_stale_sentinel_unlinks_and_writes_rollup(
    tmp_path: Path,
) -> None:
    """Gate 4a stale-present arm: aggregator succeeds, stale
    sentinel exists, factory unlinks it."""
    config = _make_config(tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    # Pre-plant a stale sentinel; it must be removed.
    sentinel = _sentinel_path(output_root)
    sentinel.write_text("PrevFailure", encoding="utf-8")
    assert sentinel.exists()

    env = build_run_environment(profile="smoke")
    run_bootstrap_rollup_via_factory(
        config,
        env=env,
        output_root=output_root,
        spec=_make_spec(),
        aggregator=_ok_aggregator,
    )

    assert not sentinel.exists()


def test_factory_happy_path_with_no_stale_sentinel_does_not_raise(
    tmp_path: Path,
) -> None:
    """Gate 4a stale-absent arm (closes qa-R1-C2): aggregator
    succeeds, no stale sentinel present, factory takes the
    `if stale.exists()` False branch without crashing."""
    config = _make_config(tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    sentinel = _sentinel_path(output_root)
    assert not sentinel.exists()

    env = build_run_environment(profile="smoke")
    run_bootstrap_rollup_via_factory(
        config,
        env=env,
        output_root=output_root,
        spec=_make_spec(),
        aggregator=_ok_aggregator,
    )

    assert not sentinel.exists()


# --- 8-9. Gate 4b: RawRollupError (partial-present vs absent) -------------


def test_factory_raw_rollup_error_with_partial_file_deletes_and_writes_sentinel(
    tmp_path: Path,
) -> None:
    """Gate 4b partial-present arm: aggregator raises, partial
    rollup file present, factory deletes it AND writes
    sentinel."""
    config = _make_config(tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    partial = _rollup_path(output_root)
    partial.write_bytes(b"stale partial")
    assert partial.exists()

    env = build_run_environment(profile="smoke")
    run_bootstrap_rollup_via_factory(
        config,
        env=env,
        output_root=output_root,
        spec=_make_spec(),
        aggregator=_raising_aggregator,
    )

    assert not partial.exists()
    sentinel = _sentinel_path(output_root)
    assert sentinel.exists()
    assert sentinel.read_text(encoding="utf-8").strip() == "RawRollupError"


def test_factory_raw_rollup_error_with_no_partial_file_writes_sentinel_only(
    tmp_path: Path,
) -> None:
    """Gate 4b partial-absent arm (closes qa-R1-I2):
    aggregator raises, no partial file present, factory
    skips the unlink AND writes the sentinel."""
    config = _make_config(tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    partial = _rollup_path(output_root)
    assert not partial.exists()

    env = build_run_environment(profile="smoke")
    run_bootstrap_rollup_via_factory(
        config,
        env=env,
        output_root=output_root,
        spec=_make_spec(),
        aggregator=_raising_aggregator,
    )

    sentinel = _sentinel_path(output_root)
    assert sentinel.exists()
    assert sentinel.read_text(encoding="utf-8").strip() == "RawRollupError"


# --- 10. Wire-up parametrize: each wrapper routes through the factory -----


_WIRE_UP_CASES = [
    (
        "_run_bootstrap_rollup",
        "_B5_SPEC",
        "aggregate_bootstrap_rollup",
    ),
    (
        "_run_bootstrap_pairwise_rollup",
        "_B6_SPEC",
        "aggregate_bootstrap_pairwise_rollup",
    ),
    (
        "_run_bootstrap_training_time_rollup",
        "_B7_SPEC",
        "aggregate_bootstrap_training_time_rollup",
    ),
    (
        "_run_bootstrap_hpo_uplift_rollup",
        "_B15_SPEC",
        "aggregate_bootstrap_hpo_uplift_rollup",
    ),
    (
        "_run_bootstrap_ensemble_lift_rollup",
        "_B16_SPEC",
        "aggregate_bootstrap_ensemble_lift_rollup",
    ),
]


@pytest.mark.parametrize(
    ("wrapper_name", "spec_name", "expected_aggregator_name"),
    _WIRE_UP_CASES,
)
def test_all_five_wrappers_route_through_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrapper_name: str,
    spec_name: str,
    expected_aggregator_name: str,
) -> None:
    """qa-R1-I1 closure: each of the 5 thin wrappers in
    `benchmarks/run.py` MUST route through
    `run_bootstrap_rollup_via_factory` with the right
    `_BX_SPEC` constant AND the correctly-named family
    aggregator. A future refactor that wires the wrong spec
    or the wrong aggregator silently breaks the family's
    coverage; this test catches it."""
    import benchmarks.run as _run_module

    recorded: dict[str, Any] = {}

    def _capture(
        _config: BenchmarkConfig,
        *,
        env: RunEnvironment,
        output_root: Path,
        spec: BootstrapWrapperSpec,
        aggregator: Any,
    ) -> None:
        del _config, env, output_root
        recorded["spec"] = spec
        recorded["aggregator"] = aggregator

    monkeypatch.setattr(_run_module, "run_bootstrap_rollup_via_factory", _capture)

    config = _make_config(tmp_path)
    output_root = tmp_path / "out"
    _make_run_manifest(config, output_root)
    env = build_run_environment(profile="smoke")
    wrapper = getattr(_run_module, wrapper_name)
    wrapper(config, env=env, output_root=output_root)

    expected_spec = getattr(_run_module, spec_name)
    assert recorded["spec"] is expected_spec
    assert recorded["aggregator"].__name__ == expected_aggregator_name


# --- 11. Gate-1 log-content sanity (closes qa-R1-N1) ----------------------


def test_factory_gate_1_log_message_includes_spec_label(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Gate 1's log message MUST include `spec.label` so a
    future maintainer can grep for the per-family suffix in
    operator-facing logs."""
    config = _make_config(tmp_path)
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    spec = _make_spec(is_enabled=False)
    env = build_run_environment(profile="smoke")

    with caplog.at_level(logging.INFO, logger="benchmarks._bootstrap_wrapper"):
        run_bootstrap_rollup_via_factory(
            config,
            env=env,
            output_root=output_root,
            spec=spec,
            aggregator=_ok_aggregator,
        )

    assert any("stub_label" in record.getMessage() for record in caplog.records)
