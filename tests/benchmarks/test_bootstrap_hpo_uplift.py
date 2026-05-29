"""Phase B15 D-B13.3 HPO-uplift bootstrap-CI aggregator tests.

Covers the 17 named aggregator tests from the B15.5 design.
Uses monkeypatch on `load_run` to inject synthetic manifests so
the tests don't depend on a real B8 driver run.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.bootstrap_manifest import (
    HPOUpliftRollupRow,
    hpo_uplift_rollup_path,
    load_hpo_uplift_rollup,
)
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments import build_run_environment
from benchmarks.report.bootstrap_hpo_uplift import (
    aggregate_bootstrap_hpo_uplift_rollup,
)
from benchmarks.report.bootstrap_rollup import RawRollupError
from benchmarks.run_manifest import (
    RunManifest,
    build_run_manifest,
    write_run_manifest,
)

from tests.benchmarks._fakes import register_all_fakes_and_get_panels


def _make_config(
    tmp_path: Path,
    *,
    bootstrap_n_resamples: int | None = None,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(
            ExperimentSpec(kind="raw_loss", seeds=(0,)),
            ExperimentSpec(
                kind="hpo_uplift",
                seeds=(0,),
                bootstrap_n_resamples=bootstrap_n_resamples,
            ),
        ),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )


def _make_run_manifest(config: BenchmarkConfig, output_root: Path) -> RunManifest:
    register_all_fakes_and_get_panels()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        config=config,
        run_id="b15-test",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)
    return manifest


def _manifest_row(
    *,
    dataset_name: str = "fake_binary",
    model_name: str = "fake_constant_binary",
    task_type: str = "binary",
    seed: int = 0,
    fold_index: int = 0,
    variant: str = "default",
    log_loss: float | None = 0.50,
    skipped_reason: str | None = None,
) -> dict[str, object]:
    """Minimal B5 manifest row shape: only the columns the
    aggregator reads (`variant`, `task_type`, `seed`, `fold_index`,
    `skipped_reason`, the per-task primary loss column)."""
    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "task_type": task_type,
        "seed": seed,
        "fold_index": fold_index,
        "variant": variant,
        "log_loss": log_loss,
        "skipped_reason": skipped_reason,
    }


def _setup(tmp_path: Path) -> tuple[BenchmarkConfig, Path, RunManifest]:
    config = _make_config(tmp_path)
    output_root = tmp_path / "out"
    manifest = _make_run_manifest(config, output_root)
    return config, output_root, manifest


def _stub_load_run(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]) -> None:
    """Stub `load_run` inside `bootstrap_hpo_uplift` to return a
    synthetic manifest DataFrame."""
    import benchmarks.report.bootstrap_hpo_uplift as _module

    df = pd.DataFrame(rows)
    monkeypatch.setattr(_module, "load_run", lambda _root: df)


# --- 1. Paired cells emit CI ------------------------------------------------


def test_aggregate_bootstrap_hpo_uplift_rollup_paired_cells_emit_ci(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2 seeds x 2 folds x default+tuned; per-cell default=0.50,
    tuned=0.30; assert primary_loss_mean ≈ 0.20 (positive Δ,
    closes qa-C1 sign) and ci_lo <= mean <= ci_hi."""
    config, output_root, manifest = _setup(tmp_path)
    rows = []
    for s in (0, 1):
        for f in (0, 1):
            rows.append(_manifest_row(seed=s, fold_index=f, variant="default", log_loss=0.50))
            rows.append(_manifest_row(seed=s, fold_index=f, variant="tuned", log_loss=0.30))
    _stub_load_run(monkeypatch, rows)

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_hpo_uplift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert len(rollup) == 1
    row = rollup[0]
    assert row.primary_loss_mean == pytest.approx(0.20, abs=1e-9)
    assert row.n_cells_paired == 4
    assert row.primary_loss_ci_lo is not None
    assert row.primary_loss_ci_hi is not None
    assert row.primary_loss_ci_lo <= row.primary_loss_mean <= row.primary_loss_ci_hi
    assert row.primary_metric == "delta"
    assert row.primary_loss_column == "log_loss"
    assert row.bootstrap_skipped_reason is None


# --- 2. Sign convention -----------------------------------------------------


def test_aggregate_bootstrap_hpo_uplift_rollup_sign_convention_default_minus_tuned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redundant pin for qa-C1: a sign-flip implementation
    returning -0.20 would fail this bare-equality assertion."""
    config, output_root, manifest = _setup(tmp_path)
    rows = [
        _manifest_row(seed=0, fold_index=0, variant="default", log_loss=0.50),
        _manifest_row(seed=0, fold_index=0, variant="tuned", log_loss=0.30),
        _manifest_row(seed=1, fold_index=0, variant="default", log_loss=0.50),
        _manifest_row(seed=1, fold_index=0, variant="tuned", log_loss=0.30),
    ]
    _stub_load_run(monkeypatch, rows)

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_hpo_uplift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert rollup[0].primary_loss_mean == pytest.approx(0.20, abs=1e-9)


# --- 3. Anti-degeneracy oracle ---------------------------------------------


def test_aggregate_bootstrap_hpo_uplift_rollup_ci_width_nonzero_with_multiple_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4+ paired cells with non-identical deltas; assert strict
    ci_hi - ci_lo > 0 (mirrors B14 qa-C1)."""
    config, output_root, manifest = _setup(tmp_path)
    rows = []
    # Per-cell deltas span 0.05..0.20.
    for i in range(4):
        rows.append(_manifest_row(seed=i, fold_index=0, variant="default", log_loss=0.50))
        rows.append(_manifest_row(seed=i, fold_index=0, variant="tuned", log_loss=0.30 + 0.05 * i))
    _stub_load_run(monkeypatch, rows)

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_hpo_uplift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    row = rollup[0]
    assert row.primary_loss_ci_hi is not None
    assert row.primary_loss_ci_lo is not None
    assert row.primary_loss_ci_hi - row.primary_loss_ci_lo > 0.0


# --- 4. Single-entity-degenerate primitive path ----------------------------


def test_aggregate_bootstrap_hpo_uplift_rollup_single_paired_cell_degenerate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1 paired cell -> primitive's n_entities==1 path fires.
    mean == ci_lo == ci_hi == per-cell Δ."""
    config, output_root, manifest = _setup(tmp_path)
    rows = [
        _manifest_row(seed=0, fold_index=0, variant="default", log_loss=0.45),
        _manifest_row(seed=0, fold_index=0, variant="tuned", log_loss=0.30),
    ]
    _stub_load_run(monkeypatch, rows)

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_hpo_uplift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    row = rollup[0]
    assert row.n_cells_paired == 1
    assert row.primary_loss_mean == pytest.approx(0.15, abs=1e-9)
    assert row.primary_loss_ci_lo == row.primary_loss_mean
    assert row.primary_loss_ci_hi == row.primary_loss_mean


# --- 5/6/7. Sentinel rows: default_only, tuned_only, paired_but_no_valid_loss ---


def test_aggregate_bootstrap_hpo_uplift_rollup_default_only_emits_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default arm present; tuned arm absent FOR THIS group ->
    sentinel with EXACT reason 'default_only' (qa-C5). Add a
    separate (model) group with a tuned row so Gate D
    (manifest-level "no tuned rows") does NOT short-circuit
    the aggregator."""
    config, output_root, manifest = _setup(tmp_path)
    rows = [
        # Group under test: default-only.
        _manifest_row(seed=0, fold_index=0, model_name="m_default_only", variant="default", log_loss=0.40),
        _manifest_row(seed=0, fold_index=1, model_name="m_default_only", variant="default", log_loss=0.42),
        # Different (model) group with a tuned row so Gate D is bypassed.
        _manifest_row(seed=0, fold_index=0, model_name="m_other", variant="default", log_loss=0.50),
        _manifest_row(seed=0, fold_index=0, model_name="m_other", variant="tuned", log_loss=0.30),
    ]
    _stub_load_run(monkeypatch, rows)

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_hpo_uplift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    by_model = {r.model_name: r for r in rollup}
    assert by_model["m_default_only"].bootstrap_skipped_reason == "default_only"
    assert by_model["m_default_only"].primary_loss_mean is None


def test_aggregate_bootstrap_hpo_uplift_rollup_tuned_only_emits_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tuned arm present; default arm absent -> sentinel with
    EXACT reason 'tuned_only' (qa-C5)."""
    config, output_root, manifest = _setup(tmp_path)
    rows = [
        _manifest_row(seed=0, fold_index=0, variant="tuned", log_loss=0.30),
        _manifest_row(seed=0, fold_index=1, variant="tuned", log_loss=0.32),
    ]
    _stub_load_run(monkeypatch, rows)

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_hpo_uplift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert rollup[0].bootstrap_skipped_reason == "tuned_only"


def test_aggregate_bootstrap_hpo_uplift_rollup_paired_but_no_valid_loss_emits_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both arms present, all paired cells have NaN losses ->
    sentinel with EXACT reason 'paired_but_no_valid_loss' (qa-C5)."""
    config, output_root, manifest = _setup(tmp_path)
    rows = [
        _manifest_row(seed=0, fold_index=0, variant="default", log_loss=float("nan")),
        _manifest_row(seed=0, fold_index=0, variant="tuned", log_loss=float("nan")),
        _manifest_row(seed=0, fold_index=1, variant="default", log_loss=float("nan")),
        _manifest_row(seed=0, fold_index=1, variant="tuned", log_loss=float("nan")),
    ]
    _stub_load_run(monkeypatch, rows)

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_hpo_uplift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert rollup[0].bootstrap_skipped_reason == "paired_but_no_valid_loss"


# --- 8. no_paired_cells sentinel -------------------------------------------


def test_aggregate_bootstrap_hpo_uplift_rollup_unpaired_cell_emits_no_paired_cells_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default at fold=0; tuned at fold=1 -> inner-join yields
    zero matches -> sentinel with EXACT reason 'no_paired_cells'."""
    config, output_root, manifest = _setup(tmp_path)
    rows = [
        _manifest_row(seed=0, fold_index=0, variant="default", log_loss=0.50),
        _manifest_row(seed=0, fold_index=1, variant="tuned", log_loss=0.30),
    ]
    _stub_load_run(monkeypatch, rows)

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_hpo_uplift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert rollup[0].bootstrap_skipped_reason == "no_paired_cells"
    assert rollup[0].n_cells_paired == 0


# --- 9. Empty manifest -----------------------------------------------------


def test_aggregate_bootstrap_hpo_uplift_rollup_empty_manifest_returns_empty_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty manifest -> []; no rollup file written."""
    config, output_root, manifest = _setup(tmp_path)
    _stub_load_run(monkeypatch, [])

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_hpo_uplift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert rollup == []
    assert not hpo_uplift_rollup_path(output_root).exists()


# --- 10. No tuned-variant rows (Gate D aggregator side) --------------------


def test_aggregate_bootstrap_hpo_uplift_rollup_no_tuned_variant_rows_returns_empty_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manifest has only variant=default rows (B8 was never run)
    -> aggregator returns []."""
    config, output_root, manifest = _setup(tmp_path)
    rows = [
        _manifest_row(seed=0, fold_index=0, variant="default", log_loss=0.50),
        _manifest_row(seed=0, fold_index=1, variant="default", log_loss=0.51),
    ]
    _stub_load_run(monkeypatch, rows)

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_hpo_uplift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert rollup == []


# --- 11. Fingerprint round-trip --------------------------------------------


def test_aggregate_bootstrap_hpo_uplift_rollup_records_manifest_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every emitted row's `manifest_fingerprint` equals the live
    RunManifest.fingerprint()."""
    config, output_root, manifest = _setup(tmp_path)
    rows = [
        _manifest_row(seed=0, fold_index=0, variant="default", log_loss=0.50),
        _manifest_row(seed=0, fold_index=0, variant="tuned", log_loss=0.30),
    ]
    _stub_load_run(monkeypatch, rows)

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_hpo_uplift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    fp = manifest.fingerprint()
    assert all(r.manifest_fingerprint == fp for r in rollup)


# --- 12. primary_loss_column audit -----------------------------------------


def test_aggregate_bootstrap_hpo_uplift_rollup_records_primary_loss_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`primary_loss_column` is 'log_loss' for binary cells AND
    round-trips through parquet."""
    config, output_root, manifest = _setup(tmp_path)
    rows = [
        _manifest_row(seed=0, fold_index=0, variant="default", log_loss=0.50),
        _manifest_row(seed=0, fold_index=0, variant="tuned", log_loss=0.30),
    ]
    _stub_load_run(monkeypatch, rows)

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_hpo_uplift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert rollup[0].primary_loss_column == "log_loss"
    loaded = load_hpo_uplift_rollup(output_root)
    assert loaded[0].primary_loss_column == "log_loss"


# --- 13. Malformed paired cell raises (group not flagged) ------------------


def test_aggregate_bootstrap_hpo_uplift_rollup_malformed_paired_cell_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A paired cell where one side has NaN loss AND the group is
    NOT (all-NaN-paired) raises RawRollupError."""
    config, output_root, manifest = _setup(tmp_path)
    rows = [
        _manifest_row(seed=0, fold_index=0, variant="default", log_loss=0.50),
        _manifest_row(seed=0, fold_index=0, variant="tuned", log_loss=float("nan")),
        # A second valid pair so the group is not entirely NaN.
        _manifest_row(seed=0, fold_index=1, variant="default", log_loss=0.51),
        _manifest_row(seed=0, fold_index=1, variant="tuned", log_loss=0.30),
    ]
    _stub_load_run(monkeypatch, rows)

    env = build_run_environment(profile="smoke")
    with pytest.raises(RawRollupError, match="NaN primary loss"):
        aggregate_bootstrap_hpo_uplift_rollup(
            config, output_root=output_root, env=env, manifest=manifest
        )


# --- 14. NaN cell with all-NaN paired group emits sentinel (R2 qa-I1) ------


def test_aggregate_bootstrap_hpo_uplift_rollup_nan_cell_with_skipped_reason_emits_sentinel_not_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary case: all paired cells have NaN. The aggregator
    routes via the paired_but_no_valid_loss sentinel (NOT a
    raise) AND `row.bootstrap_skipped_reason ==
    "paired_but_no_valid_loss"` (R3 qa-I1 BARE equality)."""
    config, output_root, manifest = _setup(tmp_path)
    rows = [
        _manifest_row(seed=0, fold_index=0, variant="default", log_loss=float("nan")),
        _manifest_row(seed=0, fold_index=0, variant="tuned", log_loss=float("nan")),
    ]
    _stub_load_run(monkeypatch, rows)

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_hpo_uplift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert rollup[0].bootstrap_skipped_reason == "paired_but_no_valid_loss"


# --- 15. Duplicate (seed, fold) pair raises --------------------------------


def test_aggregate_bootstrap_hpo_uplift_rollup_duplicate_seed_fold_pair_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two rows sharing (seed=0, fold=0) in the same (dataset,
    model, variant) group -> RawRollupError."""
    config, output_root, manifest = _setup(tmp_path)
    rows = [
        _manifest_row(seed=0, fold_index=0, variant="default", log_loss=0.50),
        _manifest_row(seed=0, fold_index=0, variant="default", log_loss=0.51),  # dup
        _manifest_row(seed=0, fold_index=0, variant="tuned", log_loss=0.30),
    ]
    _stub_load_run(monkeypatch, rows)

    env = build_run_environment(profile="smoke")
    with pytest.raises(RawRollupError, match="duplicate"):
        aggregate_bootstrap_hpo_uplift_rollup(
            config, output_root=output_root, env=env, manifest=manifest
        )


# --- 16. OOM ceiling -------------------------------------------------------


def test_aggregate_bootstrap_hpo_uplift_rollup_oom_gate_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lower BOOTSTRAP_ROW_COUNT_CEILING to 1 -> RawRollupError."""
    config, output_root, manifest = _setup(tmp_path)
    rows = [
        _manifest_row(seed=0, fold_index=0, variant="default", log_loss=0.50),
        _manifest_row(seed=0, fold_index=0, variant="tuned", log_loss=0.30),
    ]
    _stub_load_run(monkeypatch, rows)

    import benchmarks.report.bootstrap_hpo_uplift as _module

    monkeypatch.setattr(_module, "BOOTSTRAP_ROW_COUNT_CEILING", 1)
    env = build_run_environment(profile="smoke")
    with pytest.raises(RawRollupError, match="ceiling"):
        aggregate_bootstrap_hpo_uplift_rollup(
            config, output_root=output_root, env=env, manifest=manifest
        )


# --- 17. Per-spec n_resamples override -------------------------------------


def test_aggregate_bootstrap_hpo_uplift_rollup_respects_per_spec_n_resamples_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ExperimentSpec(kind="hpo_uplift", bootstrap_n_resamples=N)
    overrides the profile default."""
    config = _make_config(tmp_path, bootstrap_n_resamples=137)
    output_root = tmp_path / "out"
    manifest = _make_run_manifest(config, output_root)
    rows = [
        _manifest_row(seed=0, fold_index=0, variant="default", log_loss=0.50),
        _manifest_row(seed=0, fold_index=0, variant="tuned", log_loss=0.30),
    ]
    _stub_load_run(monkeypatch, rows)

    captured: dict[str, int] = {}

    def _capturing_primitive(
        losses: np.ndarray,
        entity_ids: np.ndarray,
        *,
        n_resamples: int,
        **_kwargs: object,
    ) -> tuple[float, float, float]:
        captured["n_resamples"] = n_resamples
        return (0.20, 0.18, 0.22)

    import benchmarks.report.bootstrap_hpo_uplift as _module

    monkeypatch.setattr(_module, "entity_block_bootstrap_ci", _capturing_primitive)

    env = build_run_environment(profile="smoke")
    aggregate_bootstrap_hpo_uplift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert captured["n_resamples"] == 137


# silence unused HPOUpliftRollupRow alias warning (used implicitly by load).
_ = HPOUpliftRollupRow
