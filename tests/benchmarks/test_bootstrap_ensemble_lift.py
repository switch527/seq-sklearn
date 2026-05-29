"""Phase B16 D-B13.4 ensemble-lift bootstrap-CI aggregator tests.

Covers the 15 named aggregator tests from the B16.6 design.
Uses monkeypatch on `load_run`, `_model_families`, and
`compute_per_cell_lift_deltas` to inject synthetic manifests +
per-cell deltas so the tests don't depend on a real B11 driver
run with cached predictions shards.
"""

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments import build_run_environment
from benchmarks.experiments.ensemble_lift import (
    PerCellLiftDelta,
    _ComputePerCellResult,
)
from benchmarks.report.bootstrap_ensemble_lift import (
    aggregate_bootstrap_ensemble_lift_rollup,
)
from benchmarks.report.bootstrap_rollup import RawRollupError
from benchmarks.run_manifest import RunManifest, build_run_manifest, write_run_manifest

from tests.benchmarks._fakes import register_all_fakes_and_get_panels

_GBM_MODEL = "gbm_constant"
_SEQ_MODEL = "seq_constant"


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
                kind="ensemble_lift",
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
        run_id="b16-test",
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
    model_name: str = _GBM_MODEL,
    task_type: str = "binary",
    seed: int = 0,
    fold_index: int = 0,
    variant: str = "default",
    skipped_reason: str | None = None,
) -> dict[str, object]:
    """Minimal B5 manifest row shape for the B16 aggregator."""
    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "task_type": task_type,
        "seed": seed,
        "fold_index": fold_index,
        "variant": variant,
        "skipped_reason": skipped_reason,
    }


def _setup(tmp_path: Path) -> tuple[BenchmarkConfig, Path, RunManifest]:
    config = _make_config(tmp_path)
    output_root = tmp_path / "out"
    manifest = _make_run_manifest(config, output_root)
    return config, output_root, manifest


def _stub_load_run(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]) -> None:
    """Stub `load_run` inside `bootstrap_ensemble_lift` to return a
    synthetic manifest DataFrame."""
    import benchmarks.report.bootstrap_ensemble_lift as _module

    df = pd.DataFrame(rows)
    monkeypatch.setattr(_module, "load_run", lambda _root: df)  # type: ignore[misc]


def _stub_families(
    monkeypatch: pytest.MonkeyPatch,
    mapping: dict[str, str] | None = None,
) -> None:
    """Stub `_model_families` to bypass registry lookups."""
    import benchmarks.report.bootstrap_ensemble_lift as _module

    families = mapping if mapping is not None else {_GBM_MODEL: "gbm", _SEQ_MODEL: "seq_sklearn"}
    monkeypatch.setattr(_module, "_model_families", lambda _df: dict(families))  # type: ignore[misc]


def _stub_per_cell(
    monkeypatch: pytest.MonkeyPatch,
    result_by_dataset: dict[str, _ComputePerCellResult],
) -> None:
    """Stub `compute_per_cell_lift_deltas` to return a controlled
    result per dataset, bypassing the B11 predictions-shard
    machinery."""
    import benchmarks.report.bootstrap_ensemble_lift as _module

    def _fake(*, dataset_name: str, **_kwargs: object) -> _ComputePerCellResult:
        return result_by_dataset.get(
            dataset_name,
            _ComputePerCellResult(cells=(), selector="log_loss"),
        )

    monkeypatch.setattr(_module, "compute_per_cell_lift_deltas", _fake)  # type: ignore[misc]


def _ok_rows_both_families(dataset_name: str = "fake_binary") -> list[dict[str, object]]:
    """Two seeds by two folds by both families OK manifest rows."""
    rows = []
    for s in (0, 1):
        for f in (0, 1):
            rows.append(
                _manifest_row(
                    dataset_name=dataset_name, model_name=_GBM_MODEL, seed=s, fold_index=f
                )
            )
            rows.append(
                _manifest_row(
                    dataset_name=dataset_name, model_name=_SEQ_MODEL, seed=s, fold_index=f
                )
            )
    return rows


def _per_cell_pairs(values: list[tuple[int, int, float, float]]) -> _ComputePerCellResult:
    """Build a `_ComputePerCellResult` from (seed, fold, loss_gbm, loss_gbm_plus_seq) tuples."""
    cells = tuple(
        PerCellLiftDelta(
            seed=s,
            fold_index=f,
            loss_gbm=lg,
            loss_gbm_plus_seq=lgs,
            delta_loss=lg - lgs,
            oracle_loss=None,
        )
        for s, f, lg, lgs in values
    )
    return _ComputePerCellResult(cells=cells, selector="log_loss")


# --- 1. Paired cells emit CI ------------------------------------------------


def test_aggregate_bootstrap_ensemble_lift_rollup_paired_cells_emit_ci(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2 seeds x 2 folds with `loss(GBM)=0.60, loss(GBM+seq)=0.40`
    per cell; assert `primary_loss_mean == pytest.approx(0.20)` and
    `ci_lo <= mean <= ci_hi`. Closes qa-C1 sign-convention pin."""
    config, output_root, manifest = _setup(tmp_path)
    _stub_load_run(monkeypatch, _ok_rows_both_families())
    _stub_families(monkeypatch)
    _stub_per_cell(
        monkeypatch,
        {"fake_binary": _per_cell_pairs([(s, f, 0.60, 0.40) for s in (0, 1) for f in (0, 1)])},
    )

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert len(rollup) == 1
    row = rollup[0]
    assert row.primary_loss_mean == pytest.approx(0.20, abs=1e-9)
    assert row.n_cells_paired == 4
    assert row.primary_loss_ci_lo is not None
    assert row.primary_loss_ci_hi is not None
    assert row.primary_loss_mean is not None
    assert row.primary_loss_ci_lo <= row.primary_loss_mean <= row.primary_loss_ci_hi
    assert row.primary_metric == "delta_loss"
    assert row.primary_loss_column == "log_loss"
    assert row.bootstrap_skipped_reason is None


# --- 2. Sign convention -----------------------------------------------------


def test_aggregate_bootstrap_ensemble_lift_rollup_sign_convention_baseline_minus_combined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asymmetric fixture (`0.60`/`0.40`) pinned to bare-equality
    `+0.20`. A sign-flip implementation returning `-0.20` would fail.
    Closes qa-C2."""
    config, output_root, manifest = _setup(tmp_path)
    _stub_load_run(monkeypatch, _ok_rows_both_families())
    _stub_families(monkeypatch)
    _stub_per_cell(
        monkeypatch,
        {"fake_binary": _per_cell_pairs([(0, 0, 0.60, 0.40), (1, 0, 0.60, 0.40)])},
    )

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert rollup[0].primary_loss_mean == pytest.approx(0.20, abs=1e-9)


# --- 3. Anti-degeneracy oracle ---------------------------------------------


def test_aggregate_bootstrap_ensemble_lift_rollup_ci_width_nonzero_with_multiple_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4+ paired cells with non-identical deltas; strict
    `ci_hi - ci_lo > 0`."""
    config, output_root, manifest = _setup(tmp_path)
    _stub_load_run(monkeypatch, _ok_rows_both_families())
    _stub_families(monkeypatch)
    _stub_per_cell(
        monkeypatch,
        {"fake_binary": _per_cell_pairs([(i, 0, 0.60, 0.40 + 0.05 * i) for i in range(4)])},
    )

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    row = rollup[0]
    assert row.primary_loss_ci_hi is not None
    assert row.primary_loss_ci_lo is not None
    assert row.primary_loss_ci_hi - row.primary_loss_ci_lo > 0.0


# --- 4. Seed-disjoint -> deterministic lexically-first sentinel ------------


def test_aggregate_bootstrap_ensemble_lift_rollup_seed_disjoint_yields_no_gbm_predictions_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GBM at seeds {0,1} fold 0, seq at seeds {2,3} fold 0
    (completely disjoint seed sets). The B11 driver flags BOTH
    `no_gbm_predictions` and `no_seq_predictions`; the aggregator
    emits the lexically-first sentinel `"no_gbm_predictions"`
    deterministically. Closes qa-C1 + qa-R3-I1 (bare equality)."""
    config, output_root, manifest = _setup(tmp_path)
    rows = []
    for s in (0, 1):
        rows.append(_manifest_row(model_name=_GBM_MODEL, seed=s, fold_index=0))
    for s in (2, 3):
        rows.append(_manifest_row(model_name=_SEQ_MODEL, seed=s, fold_index=0))
    _stub_load_run(monkeypatch, rows)
    _stub_families(monkeypatch)
    _stub_per_cell(
        monkeypatch,
        {
            "fake_binary": _ComputePerCellResult(
                cells=(),
                seen_no_gbm=True,
                seen_no_seq=True,
                selector="log_loss",
            )
        },
    )

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert len(rollup) == 1
    row = rollup[0]
    assert row.n_cells_paired == 0
    assert row.bootstrap_skipped_reason == "no_gbm_predictions"


# --- 5. Single-paired-cell degenerate -------------------------------------


def test_aggregate_bootstrap_ensemble_lift_rollup_single_paired_cell_degenerate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1 paired cell -> primitive's single-entity degenerate path
    fires; mean == ci_lo == ci_hi == per-cell Δ."""
    config, output_root, manifest = _setup(tmp_path)
    _stub_load_run(monkeypatch, _ok_rows_both_families())
    _stub_families(monkeypatch)
    _stub_per_cell(
        monkeypatch,
        {"fake_binary": _per_cell_pairs([(0, 0, 0.50, 0.35)])},
    )

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    row = rollup[0]
    assert row.n_cells_paired == 1
    assert row.primary_loss_mean == pytest.approx(0.15, abs=1e-9)


# --- 6. no_gbm_predictions sentinel (Gate D bypassed by 2nd dataset) ------


def test_aggregate_bootstrap_ensemble_lift_rollup_no_gbm_predictions_emits_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset under test has only seq rows. Add a 2nd dataset with
    both families so Gate D passes. Bare equality on the sentinel."""
    config, output_root, manifest = _setup(tmp_path)
    rows = [
        _manifest_row(dataset_name="fake_binary", model_name=_SEQ_MODEL, seed=0, fold_index=0),
        _manifest_row(dataset_name="bypass", model_name=_GBM_MODEL, seed=0, fold_index=0),
        _manifest_row(dataset_name="bypass", model_name=_SEQ_MODEL, seed=0, fold_index=0),
    ]
    _stub_load_run(monkeypatch, rows)
    _stub_families(monkeypatch)
    _stub_per_cell(
        monkeypatch,
        {
            "fake_binary": _ComputePerCellResult(
                cells=(), seen_no_gbm=True, seen_no_seq=False, selector="log_loss"
            ),
            "bypass": _per_cell_pairs([(0, 0, 0.60, 0.40)]),
        },
    )

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    by_dataset = {r.dataset_name: r for r in rollup}
    assert by_dataset["fake_binary"].bootstrap_skipped_reason == "no_gbm_predictions"
    assert by_dataset["bypass"].bootstrap_skipped_reason is None


# --- 7. no_seq_predictions sentinel (symmetric) ----------------------------


def test_aggregate_bootstrap_ensemble_lift_rollup_no_seq_predictions_emits_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Symmetric mirror of test 6. Dataset under test has only GBM
    rows; 2nd dataset bypasses Gate D."""
    config, output_root, manifest = _setup(tmp_path)
    rows = [
        _manifest_row(dataset_name="fake_binary", model_name=_GBM_MODEL, seed=0, fold_index=0),
        _manifest_row(dataset_name="bypass", model_name=_GBM_MODEL, seed=0, fold_index=0),
        _manifest_row(dataset_name="bypass", model_name=_SEQ_MODEL, seed=0, fold_index=0),
    ]
    _stub_load_run(monkeypatch, rows)
    _stub_families(monkeypatch)
    _stub_per_cell(
        monkeypatch,
        {
            "fake_binary": _ComputePerCellResult(
                cells=(), seen_no_gbm=False, seen_no_seq=True, selector="log_loss"
            ),
            "bypass": _per_cell_pairs([(0, 0, 0.60, 0.40)]),
        },
    )

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    by_dataset = {r.dataset_name: r for r in rollup}
    assert by_dataset["fake_binary"].bootstrap_skipped_reason == "no_seq_predictions"


# --- 8. all_cells_skipped_in_manifest sentinel -----------------------------


def test_aggregate_bootstrap_ensemble_lift_rollup_all_cells_skipped_emits_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Target dataset has both families but all cells flagged
    `skipped_reason`; 2nd dataset has both families OK. The B11
    helper returns empty cells with neither `seen_no_gbm` nor
    `seen_no_seq` set -> aggregator routes to
    `all_cells_skipped_in_manifest`. Closes qa-I4 Gate D bypass."""
    config, output_root, manifest = _setup(tmp_path)
    rows = [
        _manifest_row(
            dataset_name="fake_binary",
            model_name=_GBM_MODEL,
            seed=0,
            fold_index=0,
            skipped_reason="something",
        ),
        _manifest_row(
            dataset_name="fake_binary",
            model_name=_SEQ_MODEL,
            seed=0,
            fold_index=0,
            skipped_reason="something",
        ),
        _manifest_row(dataset_name="bypass", model_name=_GBM_MODEL, seed=0, fold_index=0),
        _manifest_row(dataset_name="bypass", model_name=_SEQ_MODEL, seed=0, fold_index=0),
    ]
    _stub_load_run(monkeypatch, rows)
    _stub_families(monkeypatch)
    _stub_per_cell(
        monkeypatch,
        {
            "fake_binary": _ComputePerCellResult(
                cells=(), seen_no_gbm=False, seen_no_seq=False, selector="log_loss"
            ),
            "bypass": _per_cell_pairs([(0, 0, 0.60, 0.40)]),
        },
    )

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    by_dataset = {r.dataset_name: r for r in rollup}
    assert by_dataset["fake_binary"].bootstrap_skipped_reason == "all_cells_skipped_in_manifest"


# --- 9. Empty manifest -> empty list --------------------------------------


def test_aggregate_bootstrap_ensemble_lift_rollup_empty_manifest_returns_empty_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty manifest -> aggregator returns `[]` without writing
    any rollup or sentinel."""
    config, output_root, manifest = _setup(tmp_path)
    _stub_load_run(monkeypatch, [])

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert rollup == []


# --- 10. Records manifest_fingerprint --------------------------------------


def test_aggregate_bootstrap_ensemble_lift_rollup_records_manifest_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rollup row's `manifest_fingerprint` equals
    `manifest.fingerprint()` so the renderer's freshness check
    can compare directly."""
    config, output_root, manifest = _setup(tmp_path)
    _stub_load_run(monkeypatch, _ok_rows_both_families())
    _stub_families(monkeypatch)
    _stub_per_cell(
        monkeypatch,
        {"fake_binary": _per_cell_pairs([(0, 0, 0.60, 0.40), (0, 1, 0.60, 0.40)])},
    )

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert rollup[0].manifest_fingerprint == manifest.fingerprint()


# --- 11. Records primary_loss_column for binary AND regression ------------


@pytest.mark.parametrize(
    ("task_type", "expected_column"),
    [("binary", "log_loss"), ("regression_point", "rmse")],
)
def test_aggregate_bootstrap_ensemble_lift_rollup_records_primary_loss_column_for_binary_and_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_type: str,
    expected_column: str,
) -> None:
    """The audit field tracks `log_loss` for classification and
    `rmse` for regression_point."""
    config, output_root, manifest = _setup(tmp_path)
    rows = []
    for s in (0, 1):
        for f in (0,):
            rows.append(
                _manifest_row(
                    model_name=_GBM_MODEL,
                    seed=s,
                    fold_index=f,
                    task_type=task_type,
                )
            )
            rows.append(
                _manifest_row(
                    model_name=_SEQ_MODEL,
                    seed=s,
                    fold_index=f,
                    task_type=task_type,
                )
            )
    _stub_load_run(monkeypatch, rows)
    _stub_families(monkeypatch)
    selector = expected_column
    _stub_per_cell(
        monkeypatch,
        {
            "fake_binary": _ComputePerCellResult(
                cells=tuple(
                    PerCellLiftDelta(
                        seed=s,
                        fold_index=0,
                        loss_gbm=0.60,
                        loss_gbm_plus_seq=0.40,
                        delta_loss=0.20,
                        oracle_loss=None,
                    )
                    for s in (0, 1)
                ),
                selector=selector,
            )
        },
    )

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert rollup[0].primary_loss_column == expected_column


# --- 12. Malformed cell raises --------------------------------------------


def test_aggregate_bootstrap_ensemble_lift_rollup_malformed_cell_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-finite `delta_loss` in the per-cell records raises
    `RawRollupError` (the upstream B11 helper only emits finite-
    loss cells; non-finite at this layer indicates predictions-
    shard corruption)."""
    config, output_root, manifest = _setup(tmp_path)
    _stub_load_run(monkeypatch, _ok_rows_both_families())
    _stub_families(monkeypatch)
    _stub_per_cell(
        monkeypatch,
        {
            "fake_binary": _ComputePerCellResult(
                cells=(
                    PerCellLiftDelta(
                        seed=0,
                        fold_index=0,
                        loss_gbm=0.60,
                        loss_gbm_plus_seq=0.40,
                        delta_loss=float("nan"),
                        oracle_loss=None,
                    ),
                ),
                selector="log_loss",
            )
        },
    )

    env = build_run_environment(profile="smoke")
    with pytest.raises(RawRollupError, match="non-finite"):
        aggregate_bootstrap_ensemble_lift_rollup(
            config, output_root=output_root, env=env, manifest=manifest
        )


# --- 13. Duplicate (seed, fold) pair raises -------------------------------


def test_aggregate_bootstrap_ensemble_lift_rollup_duplicate_seed_fold_pair_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same `(seed=0, fold=0)` row appears TWICE in the GBM
    family arm. The aggregator raises `RawRollupError` rather than
    silently double-counting (parity with B15
    `_duplicate_seed_fold_pair_raises`). Closes R2 qa-C1."""
    config, output_root, manifest = _setup(tmp_path)
    rows = [
        _manifest_row(model_name=_GBM_MODEL, seed=0, fold_index=0),
        _manifest_row(model_name=_GBM_MODEL, seed=0, fold_index=0),  # duplicate
        _manifest_row(model_name=_SEQ_MODEL, seed=0, fold_index=0),
    ]
    _stub_load_run(monkeypatch, rows)
    _stub_families(monkeypatch)

    env = build_run_environment(profile="smoke")
    with pytest.raises(RawRollupError, match="duplicate"):
        aggregate_bootstrap_ensemble_lift_rollup(
            config, output_root=output_root, env=env, manifest=manifest
        )


# --- 14. OOM ceiling raises ------------------------------------------------


def test_aggregate_bootstrap_ensemble_lift_rollup_oom_gate_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `n_cells * n_resamples > BOOTSTRAP_ROW_COUNT_CEILING`,
    the aggregator raises `RawRollupError` (defensive OOM guard
    inherited from B13/B14/B15)."""
    config, output_root, manifest = _setup(tmp_path)
    _stub_load_run(monkeypatch, _ok_rows_both_families())
    _stub_families(monkeypatch)
    _stub_per_cell(
        monkeypatch,
        {"fake_binary": _per_cell_pairs([(0, 0, 0.60, 0.40)])},
    )
    import benchmarks.report.bootstrap_ensemble_lift as _module

    monkeypatch.setattr(_module, "BOOTSTRAP_ROW_COUNT_CEILING", 1)  # type: ignore[misc]

    env = build_run_environment(profile="smoke")
    with pytest.raises(RawRollupError, match="ceiling"):
        aggregate_bootstrap_ensemble_lift_rollup(
            config, output_root=output_root, env=env, manifest=manifest
        )


# --- 15. Per-spec n_resamples override --------------------------------------


def test_aggregate_bootstrap_ensemble_lift_rollup_respects_per_spec_n_resamples_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ExperimentSpec override of `bootstrap_n_resamples=137`
    (an unusual, detectable value) is honored end-to-end; the
    monkeypatched primitive captures the value the aggregator
    passes through."""
    config = _make_config(tmp_path, bootstrap_n_resamples=137)
    output_root = tmp_path / "out"
    manifest = _make_run_manifest(config, output_root)
    _stub_load_run(monkeypatch, _ok_rows_both_families())
    _stub_families(monkeypatch)
    _stub_per_cell(
        monkeypatch,
        {"fake_binary": _per_cell_pairs([(0, 0, 0.60, 0.40), (0, 1, 0.60, 0.40)])},
    )

    captured: dict[str, int] = {}

    def _capture_primitive(
        deltas: object,
        entity_ids: object,
        *,
        n_resamples: int,
        confidence: float,
        seed: int,
    ) -> tuple[float, float, float]:
        del deltas, entity_ids, confidence, seed
        captured["n_resamples"] = n_resamples
        return (0.20, 0.15, 0.25)

    import benchmarks.report.bootstrap_ensemble_lift as _module

    monkeypatch.setattr(_module, "entity_block_bootstrap_ci", _capture_primitive)  # type: ignore[misc]

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert captured["n_resamples"] == 137
    assert rollup[0].bootstrap_n_resamples == 137


# --- 16. Byte-string regression pin on B11 _per_dataset_lift --------------


def test_compute_per_cell_lift_deltas_per_dataset_lift_byte_string_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-B16-1: pinning that the refactor extraction of
    `compute_per_cell_lift_deltas` from `_per_dataset_lift` did
    not change the per-dataset summary. A synthetic monkeypatched
    helper feeds known per-cell deltas; we then assert the
    resulting `PerDatasetLift.model_dump()` matches an
    explicit `EXPECTED_DICT` covering every field on the
    schema (closes qa-C3 byte-pin scope mandate)."""
    del tmp_path  # silence unused
    import benchmarks.experiments.ensemble_lift as _b11

    # Stub helper output: 4 cells, asymmetric losses so means
    # and std are non-trivial.
    records = (
        PerCellLiftDelta(
            seed=0,
            fold_index=0,
            loss_gbm=0.60,
            loss_gbm_plus_seq=0.40,
            delta_loss=0.20,
            oracle_loss=0.30,
        ),
        PerCellLiftDelta(
            seed=0,
            fold_index=1,
            loss_gbm=0.70,
            loss_gbm_plus_seq=0.50,
            delta_loss=0.20,
            oracle_loss=0.40,
        ),
        PerCellLiftDelta(
            seed=1,
            fold_index=0,
            loss_gbm=0.65,
            loss_gbm_plus_seq=0.45,
            delta_loss=0.20,
            oracle_loss=0.35,
        ),
        PerCellLiftDelta(
            seed=1,
            fold_index=1,
            loss_gbm=0.55,
            loss_gbm_plus_seq=0.35,
            delta_loss=0.20,
            oracle_loss=0.25,
        ),
    )

    def _fake(
        *,
        dataset_name: str,
        task_type: str,
        seed_fold_pairs: list[tuple[int, int]],
        gbm_cells: pd.DataFrame,
        seq_cells: pd.DataFrame,
        output_root: Path,
    ) -> _ComputePerCellResult:
        del dataset_name, task_type, seed_fold_pairs, gbm_cells, seq_cells, output_root
        return _ComputePerCellResult(cells=records, selector="log_loss")

    monkeypatch.setattr(_b11, "compute_per_cell_lift_deltas", _fake)

    result = _b11._per_dataset_lift(
        dataset_name="byte_pin_dataset",
        task_type="binary",
        seed_fold_pairs=[(s, f) for s in (0, 1) for f in (0, 1)],
        gbm_cells=pd.DataFrame(),
        seq_cells=pd.DataFrame(),
        output_root=Path("/tmp"),
    )
    dump = result.model_dump()
    assert set(dump.keys()) == {
        "dataset_name",
        "task_type",
        "primary_loss_column",
        "n_cells_paired",
        "loss_gbm_only_mean",
        "loss_gbm_plus_seq_mean",
        "delta_loss_mean",
        "delta_loss_std",
        "oracle_loss_mean",
        "oracle_delta_loss_mean",
        "no_gbm_predictions",
        "no_seq_predictions",
    }
    assert dump["dataset_name"] == "byte_pin_dataset"
    assert dump["task_type"] == "binary"
    assert dump["primary_loss_column"] == "log_loss"
    assert dump["n_cells_paired"] == 4
    assert dump["no_gbm_predictions"] is False
    assert dump["no_seq_predictions"] is False
    assert dump["loss_gbm_only_mean"] == pytest.approx(0.625, abs=1e-9)
    assert dump["loss_gbm_plus_seq_mean"] == pytest.approx(0.425, abs=1e-9)
    assert dump["delta_loss_mean"] == pytest.approx(0.20, abs=1e-9)
    assert dump["delta_loss_std"] == pytest.approx(0.0, abs=1e-9)
    assert dump["oracle_loss_mean"] == pytest.approx(0.325, abs=1e-9)
    assert dump["oracle_delta_loss_mean"] == pytest.approx(0.30, abs=1e-9)
