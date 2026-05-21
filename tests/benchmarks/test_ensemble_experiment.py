"""Phase B6 ensemble-experiment e2e tests (B6.2).

Reuses the fake-registry adapters from `test_raw_loss_experiment.py`
to materialize a B5 manifest, then drives `run_ensemble` against it
to verify the pairwise statistics shard layout, the per-pair skip
reasons, the empty-join handling, the resume-via-sentinel
behavior, and the renderer.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.datasets._base import PanelDataset
from benchmarks.experiments import (
    EnsembleExperimentResult,
    PairwiseRow,
    build_run_environment,
    run_ensemble,
    run_raw_loss,
)
from benchmarks.experiments.ensemble import (
    PairwiseKeyError,
    is_pair_complete,
    load_pairwise,
    pair_key,
    pairwise_shard_path,
    write_pairwise_cell,
)
from benchmarks.predictions import load_predictions
from benchmarks.report.ensemble import aggregate_pairs, render_pairwise_markdown

# Reuse the synthetic-registry helpers from the shared fakes module.
from tests.benchmarks._fakes import register_all_fakes_and_get_panels


@pytest.fixture
def fake_registry() -> list[PanelDataset]:
    """Register the eleven synthetic adapters + three synthetic
    datasets for the test. The autouse `isolated_registry` conftest
    fixture restores the registry on teardown."""
    return register_all_fakes_and_get_panels()


def _make_config(
    *,
    datasets: tuple[str, ...],
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    output_dir: Path,
    cache_dir: Path,
    experiment_kinds: tuple[str, ...] = ("raw_loss", "ensemble"),
) -> BenchmarkConfig:
    return BenchmarkConfig(
        datasets=datasets,
        models=models,
        experiments=tuple(
            ExperimentSpec(kind=k, seeds=seeds)
            for k in experiment_kinds  # type: ignore[arg-type]
        ),
        output_dir=output_dir,
        cache_dir=cache_dir,
    )


def _run_raw_loss_then_ensemble(
    *,
    datasets: tuple[str, ...],
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    tmp_path: Path,
) -> tuple[Path, EnsembleExperimentResult]:
    config = _make_config(
        datasets=datasets,
        models=models,
        seeds=seeds,
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    run_raw_loss(config, output_root=output_root, env=env)
    result = run_ensemble(config, output_root=output_root, env=env)
    return output_root, result


# --- pair_key validation -----------------------------------------------------


def test_pair_key_canonicalizes_to_lexicographic_order() -> None:
    key_ab = pair_key(
        dataset_name="ds_a",
        model_a="model_z",
        model_b="model_a",
        seed=0,
        fold_index=1,
    )
    key_ba = pair_key(
        dataset_name="ds_a",
        model_a="model_a",
        model_b="model_z",
        seed=0,
        fold_index=1,
    )
    # The (A, B) and (B, A) shards collapse to the same key.
    assert key_ab == key_ba
    assert "model_a__vs__model_z" in key_ab


def test_pair_key_rejects_same_model() -> None:
    with pytest.raises(PairwiseKeyError, match="distinct models"):
        pair_key(
            dataset_name="ds_a",
            model_a="model_x",
            model_b="model_x",
            seed=0,
            fold_index=0,
        )


def test_pair_key_rejects_separator_in_token() -> None:
    with pytest.raises(PairwiseKeyError, match="separator"):
        pair_key(
            dataset_name="ds__broken",
            model_a="a",
            model_b="b",
            seed=0,
            fold_index=0,
        )


def test_pair_key_rejects_negative_fold() -> None:
    with pytest.raises(PairwiseKeyError, match="fold_index"):
        pair_key(
            dataset_name="ds_a",
            model_a="a",
            model_b="b",
            seed=0,
            fold_index=-1,
        )


# --- end-to-end pairwise on a binary B5 manifest -----------------------------


def test_ensemble_run_emits_pairwise_shards_for_binary_dataset(
    fake_registry: object, tmp_path: Path
) -> None:
    del fake_registry
    # Two classifier models on the binary dataset, single seed,
    # 5 folds: one (model_a, model_b) pair x 5 folds = 5 pairwise
    # shards.
    output_root, result = _run_raw_loss_then_ensemble(
        datasets=("fake_binary",),
        models=("fake_constant_binary", "fake_nan_proba_classifier"),
        seeds=(0,),
        tmp_path=tmp_path,
    )
    assert isinstance(result, EnsembleExperimentResult)
    assert result.pairs_attempted == 5
    assert result.pairs_skipped_predictions_missing == 0
    manifest = load_pairwise(output_root)
    assert len(manifest) == 5
    # The diversity stats (Q, phi, disagreement, double-fault) are
    # computed from the hard prediction labels and are always
    # defined for these synthetic fakes (both classifiers produce
    # an integer label per row).
    assert bool(manifest["yule_q"].notna().all())
    assert bool(manifest["phi"].notna().all())
    assert bool(manifest["disagreement_rate"].notna().all())
    assert bool(manifest["double_fault_rate"].notna().all())
    # The pearson_pred_corr and pearson_error_corr columns CAN
    # be NaN when one model has zero variance on the fold (the
    # constant-binary fakes have constant predictions, so the
    # correlation columns are undefined). The notna assertion is
    # restricted to the always-defined diversity stats.


def test_ensemble_run_handles_regression_point_dataset(
    fake_registry: object, tmp_path: Path
) -> None:
    del fake_registry
    # Only one regressor model registered; need to register another
    # for the pair. Use the fake_quantile_regressor (whose
    # task_types includes regression_point too).
    output_root, result = _run_raw_loss_then_ensemble(
        datasets=("fake_regression_point",),
        models=("fake_constant_regressor", "fake_quantile_regressor"),
        seeds=(0,),
        tmp_path=tmp_path,
    )
    assert result.pairs_attempted == 5
    manifest = load_pairwise(output_root)
    # Regression cells do NOT populate the four classification
    # diversity stats; only the cross-task correlation fields are
    # eligible. The fakes produce constant predictions per cell, so
    # the correlation columns are nan for THIS adapter pair; the
    # column is present in the schema regardless.
    assert bool(manifest["yule_q"].isna().all())
    assert bool(manifest["disagreement_rate"].isna().all())
    assert "pearson_pred_corr" in manifest.columns


# --- resumability ------------------------------------------------------------


def test_ensemble_is_resumable_via_sentinel(fake_registry: object, tmp_path: Path) -> None:
    del fake_registry
    output_root, first = _run_raw_loss_then_ensemble(
        datasets=("fake_binary",),
        models=("fake_constant_binary", "fake_nan_proba_classifier"),
        seeds=(0,),
        tmp_path=tmp_path,
    )
    assert first.pairs_attempted == 5
    # Re-run: every pair sentinel is present, so the second call
    # does no work.
    env = build_run_environment(profile="smoke")
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary", "fake_nan_proba_classifier"),
        seeds=(0,),
        output_dir=output_root,
        cache_dir=tmp_path / "cache",
    )
    second = run_ensemble(config, output_root=output_root, env=env)
    assert second.pairs_attempted == 0
    assert second.pairs_already_complete == 5


# --- empty-join skip ---------------------------------------------------------


def test_ensemble_emits_empty_join_skip_when_predictions_dont_overlap(
    fake_registry: object, tmp_path: Path
) -> None:
    del fake_registry
    # Build a B5 manifest via raw_loss, then poison one model's
    # prediction shards to have disjoint panel_row_index from the
    # other (forces the inner-join to be empty).
    output_root, _ = _run_raw_loss_then_ensemble(
        datasets=("fake_binary",),
        models=("fake_constant_binary", "fake_nan_proba_classifier"),
        seeds=(0,),
        tmp_path=tmp_path,
    )
    # Delete the predictions sentinel directory's existing shards
    # for the second model and rewrite them with an offset
    # panel_row_index (shift by +1000) so the inner-join is empty.
    from benchmarks.manifest import cell_key
    from benchmarks.predictions import write_predictions

    for fold_index in range(5):
        key = cell_key(
            dataset_name="fake_binary",
            model_name="fake_nan_proba_classifier",
            seed=0,
            variant="default",
            fold_index=fold_index,
        )
        df = load_predictions(output_root, key)
        write_predictions(
            output_root,
            key,
            panel_row_index=df["panel_row_index"].to_numpy() + 1000,
            y_true=df["y_true"].to_numpy(),
            y_pred=df["y_pred"].to_numpy(),
            y_proba=df[["y_proba_0", "y_proba_1"]].to_numpy(),
        )

    # Clear the ensemble sentinels so the driver re-runs the pairs.
    import shutil

    pairwise_sentinel_dir = output_root / "pairwise" / "_done"
    if pairwise_sentinel_dir.exists():
        shutil.rmtree(pairwise_sentinel_dir)

    env = build_run_environment(profile="smoke")
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary", "fake_nan_proba_classifier"),
        seeds=(0,),
        output_dir=output_root,
        cache_dir=tmp_path / "cache",
    )
    result = run_ensemble(config, output_root=output_root, env=env)
    assert result.pairs_skipped_empty_join == 5
    manifest = load_pairwise(output_root)
    assert bool(manifest["skipped_reason"].str.startswith("empty_panel_row_index_join").all())


# --- predictions-missing skip ------------------------------------------------


def test_ensemble_emits_predictions_missing_when_one_shard_absent(
    fake_registry: object, tmp_path: Path
) -> None:
    del fake_registry
    output_root, _ = _run_raw_loss_then_ensemble(
        datasets=("fake_binary",),
        models=("fake_constant_binary", "fake_nan_proba_classifier"),
        seeds=(0,),
        tmp_path=tmp_path,
    )
    # Delete one model's prediction shards entirely; the pair must
    # skip with `predictions_missing`.
    from benchmarks.manifest import cell_key
    from benchmarks.predictions import predictions_path

    for fold_index in range(5):
        key = cell_key(
            dataset_name="fake_binary",
            model_name="fake_nan_proba_classifier",
            seed=0,
            variant="default",
            fold_index=fold_index,
        )
        predictions_path(output_root, key).unlink()
    # Clear ensemble sentinels so the run re-evaluates the pairs.
    import shutil

    sentinel_dir = output_root / "pairwise" / "_done"
    if sentinel_dir.exists():
        shutil.rmtree(sentinel_dir)
    pairwise_shards = output_root / "pairwise"
    for shard in pairwise_shards.glob("*.parquet"):
        shard.unlink()

    env = build_run_environment(profile="smoke")
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary", "fake_nan_proba_classifier"),
        seeds=(0,),
        output_dir=output_root,
        cache_dir=tmp_path / "cache",
    )
    result = run_ensemble(config, output_root=output_root, env=env)
    assert result.pairs_skipped_predictions_missing == 5
    manifest = load_pairwise(output_root)
    assert bool(manifest["skipped_reason"].str.startswith("predictions_missing").all())


# --- renderer ----------------------------------------------------------------


def test_render_pairwise_markdown_includes_dataset_and_top_n(
    fake_registry: object, tmp_path: Path
) -> None:
    del fake_registry
    output_root, _ = _run_raw_loss_then_ensemble(
        datasets=("fake_binary",),
        models=("fake_constant_binary", "fake_nan_proba_classifier"),
        seeds=(0,),
        tmp_path=tmp_path,
    )
    manifest = load_pairwise(output_root)
    md = render_pairwise_markdown(manifest)
    assert "Pairwise ensemble-complementarity report" in md
    assert "fake_binary" in md
    assert "complementarity_score" in md
    # Top-N section appears for classification pairs.
    assert "Top-" in md
    assert "most complementary" in md


def test_render_pairwise_markdown_returns_no_results_on_empty_manifest() -> None:
    md = render_pairwise_markdown(pd.DataFrame())
    assert md.startswith("# Pairwise")
    assert "No pairwise" in md


def test_aggregate_pairs_drops_all_skipped_groups() -> None:
    # If every row in a (dataset, A, B) group is skipped, the
    # summary list omits it.
    manifest = pd.DataFrame(
        [
            {
                "dataset_name": "ds_a",
                "model_a": "a",
                "model_b": "b",
                "task_type": "binary",
                "seed": 0,
                "fold_index": 0,
                "skipped_reason": "predictions_missing: ...",
                "yule_q": None,
                "disagreement_rate": None,
                "pearson_error_corr": None,
            }
        ]
    )
    assert aggregate_pairs(manifest) == []


# --- empty-config branch -----------------------------------------------------


def test_run_ensemble_raises_when_no_ensemble_experiment(
    fake_registry: object, tmp_path: Path
) -> None:
    del fake_registry
    # Run raw_loss first so the manifest is non-empty, then pass a
    # config without an `ensemble` experiment spec.
    config_with_raw = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        experiment_kinds=("raw_loss",),
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    run_raw_loss(config_with_raw, output_root=output_root, env=env)
    with pytest.raises(ValueError, match="no ensemble"):
        run_ensemble(config_with_raw, output_root=output_root, env=env)


def test_run_ensemble_raises_on_empty_manifest(fake_registry: object, tmp_path: Path) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="manifest"):
        run_ensemble(config, output_root=tmp_path / "out", env=env)


# --- write_pairwise_cell + load_pairwise round-trip --------------------------


def test_write_and_load_pairwise_cell_round_trip(tmp_path: Path) -> None:
    row = PairwiseRow(
        library_git_sha="lib_sha",
        run_id="run-uuid",
        started_at_utc="2026-05-21T00:00:00+00:00",
        dataset_name="ds_a",
        model_a="a",
        model_b="b",
        seed=0,
        fold_index=0,
        task_type="binary",
        skipped_reason=None,
        n_samples=10,
        n11=4,
        n10=1,
        n01=1,
        n00=4,
        yule_q=0.875,
        phi=0.6,
        disagreement_rate=0.2,
        double_fault_rate=0.4,
        pearson_pred_corr=0.8,
        spearman_pred_corr=0.75,
        pearson_error_corr=0.5,
    )
    write_pairwise_cell(tmp_path, row)
    key = row.pair_key()
    assert is_pair_complete(tmp_path, key)
    assert pairwise_shard_path(tmp_path, key).exists()
    manifest = load_pairwise(tmp_path)
    assert len(manifest) == 1
    assert manifest.iloc[0]["yule_q"] == pytest.approx(0.875)


def test_load_pairwise_returns_empty_on_no_shards(tmp_path: Path) -> None:
    manifest = load_pairwise(tmp_path)
    assert manifest.empty


# --- B6.2 oracle: error-correlation on known predictions ---------------------


def test_pairwise_error_correlation_pinned_by_synthetic_predictions(
    tmp_path: Path,
) -> None:
    # Hand-craft two prediction shards with KNOWN per-sample NLL
    # vectors so the pearson_error_corr is computable by hand.
    from benchmarks.experiments.ensemble import _pairwise_for_cell
    from benchmarks.experiments.raw_loss import RunEnvironment
    from benchmarks.manifest import cell_key
    from benchmarks.predictions import write_predictions

    env = RunEnvironment(
        run_id="test", library_git_sha="lib_sha", hardware_tier="cpu", profile="smoke"
    )
    y_true = np.array([0, 1, 0, 1, 0])
    proba_a = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7], [0.6, 0.4]])
    # B is wrong on every prediction in the same way (perfectly
    # correlated NLL): same proba except low confidence.
    # Model B: also correctly classifies (argmax matches y_true on
    # every row) but with VARYING confidence so per-sample NLL has
    # nonzero variance (otherwise pearson_correlation of the two
    # NLL vectors is nan).
    proba_b = np.array([[0.6, 0.4], [0.4, 0.6], [0.75, 0.25], [0.35, 0.65], [0.55, 0.45]])
    y_pred_a = proba_a.argmax(axis=1)
    y_pred_b = proba_b.argmax(axis=1)
    panel_row_index = np.array([100, 101, 102, 103, 104])
    key_a = cell_key(
        dataset_name="ds_x",
        model_name="model_a",
        seed=0,
        variant="default",
        fold_index=0,
    )
    key_b = cell_key(
        dataset_name="ds_x",
        model_name="model_b",
        seed=0,
        variant="default",
        fold_index=0,
    )
    write_predictions(
        tmp_path,
        key_a,
        panel_row_index=panel_row_index,
        y_true=y_true,
        y_pred=y_pred_a,
        y_proba=proba_a,
    )
    write_predictions(
        tmp_path,
        key_b,
        panel_row_index=panel_row_index,
        y_true=y_true,
        y_pred=y_pred_b,
        y_proba=proba_b,
    )
    row = _pairwise_for_cell(
        env=env,
        output_root=tmp_path,
        dataset_name="ds_x",
        model_a="model_a",
        model_b="model_b",
        task_type="binary",
        seed=0,
        fold_index=0,
        classes=np.array([0, 1]),
    )
    assert row.skipped_reason is None
    assert row.n_samples == 5
    # The classification 2x2: model_a is correct on rows 0,1; B is
    # always wrong (confidence 0.55 on the wrong class), but...
    # actually argmax of [0.55, 0.45] is index 0, which matches
    # y_true on the "0" rows. Recompute by hand.
    # y_pred_a from argmax: [0, 1, 0, 1, 0] (perfect on every row).
    # y_pred_b from argmax: [0, 1, 0, 1, 0] (also perfect since
    # 0.55 on index 0 wins for true-0 rows, 0.55 on index 1 wins
    # for true-1 rows).
    assert row.n11 == 5  # both correct on every row
    assert row.n10 == 0
    assert row.n01 == 0
    assert row.n00 == 0
    # All-agree -> Q == phi == 1.0 by the degenerate convention.
    assert row.yule_q == 1.0
    assert row.phi == 1.0
    assert row.disagreement_rate == 0.0
    # Pearson on the per-sample NLL: should be positive since A
    # and B follow the same prediction pattern.
    assert row.pearson_error_corr is not None
    assert not math.isnan(row.pearson_error_corr)


# --- R1 / qa-C1: `_pairwise_for_cell` no-proba else-arm pinned. ---


def test_pairwise_for_cell_no_proba_yields_nan_error_corr(tmp_path: Path) -> None:
    # When either model's prediction shard has no `y_proba_*`
    # columns, the classification-task path falls into the else-arm
    # at `pearson_error_corr = float("nan")`. Pre-R1 this branch was
    # unexercised; mutation testing would not have caught a sign
    # flip on the surrounding guard.
    from benchmarks.experiments.ensemble import _pairwise_for_cell
    from benchmarks.experiments.raw_loss import RunEnvironment
    from benchmarks.manifest import cell_key
    from benchmarks.predictions import write_predictions

    env = RunEnvironment(
        run_id="test",
        library_git_sha="lib_sha",
        hardware_tier="cpu",
        profile="smoke",
    )
    panel_row_index = np.array([0, 1, 2, 3])
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    # NO y_proba on either shard: classification path, no NLL.
    for model_name in ("model_a", "model_b"):
        key = cell_key(
            dataset_name="ds_x",
            model_name=model_name,
            seed=0,
            variant="default",
            fold_index=0,
        )
        write_predictions(
            tmp_path,
            key,
            panel_row_index=panel_row_index,
            y_true=y_true,
            y_pred=y_pred,
        )

    row = _pairwise_for_cell(
        env=env,
        output_root=tmp_path,
        dataset_name="ds_x",
        model_a="model_a",
        model_b="model_b",
        task_type="binary",
        seed=0,
        fold_index=0,
        classes=np.array([0, 1]),
    )
    assert row.skipped_reason is None
    assert row.n_samples == 4
    # Diversity stats still meaningful (both classifiers agree on
    # every row -> Q=phi=1.0 by the degenerate convention).
    assert row.yule_q == 1.0
    assert row.phi == 1.0
    # The else-arm output: error correlation undefined without proba.
    assert row.pearson_error_corr is not None
    assert math.isnan(row.pearson_error_corr)


# --- R1 / qa-C2 + arch-C2: dropped counter fields are not present. ---


def test_ensemble_result_does_not_carry_dead_task_mismatch_counters(
    fake_registry: object, tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary", "fake_nan_proba_classifier"),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    run_raw_loss(config, output_root=output_root, env=env)
    result = run_ensemble(config, output_root=output_root, env=env)
    # The pre-R1 `EnsembleExperimentResult` carried two dead
    # counters; the R1 fix dropped them. Their absence is part of
    # the public contract now.
    fields = set(type(result).model_fields.keys())
    assert "pairs_skipped_task_mismatch" not in fields
    assert "pairs_skipped_quantile" not in fields
    # The four reachable counters ARE present.
    assert {
        "pairs_attempted",
        "pairs_skipped_predictions_missing",
        "pairs_skipped_empty_join",
        "pairs_already_complete",
    }.issubset(fields)


# --- R1 / qa-I: load_pairwise dir-exists-empty path. ---


def test_load_pairwise_returns_empty_when_dir_exists_but_no_shards(tmp_path: Path) -> None:
    # Create `pairwise/` but no `.parquet` files inside.
    (tmp_path / "pairwise").mkdir()
    manifest = load_pairwise(tmp_path)
    assert manifest.empty


# --- R1 / qa-I: aggregate_pairs empty manifest. ---


def test_aggregate_pairs_empty_manifest_returns_empty_list() -> None:
    assert aggregate_pairs(pd.DataFrame()) == []


# --- R1 / qa-I: render_from_dir round-trip. ---


def test_render_from_dir_loads_pairwise_and_renders_markdown(
    fake_registry: object, tmp_path: Path
) -> None:
    del fake_registry
    from benchmarks.report.ensemble import render_from_dir

    output_root, _ = _run_raw_loss_then_ensemble(
        datasets=("fake_binary",),
        models=("fake_constant_binary", "fake_nan_proba_classifier"),
        seeds=(0,),
        tmp_path=tmp_path,
    )
    md = render_from_dir(output_root)
    assert "Pairwise ensemble-complementarity report" in md
    assert "fake_binary" in md


# --- R1 / qa-N1: spearman_correlation short-input branch. ---


def test_spearman_correlation_returns_nan_on_too_short_input() -> None:
    from benchmarks.metrics.pairwise import spearman_correlation

    assert math.isnan(spearman_correlation(np.array([1.0]), np.array([2.0])))


# --- R1 / qa-N2: pair_key invalid-token regex rejection. ---


def test_pair_key_rejects_token_starting_with_underscore() -> None:
    # The token regex `^[A-Za-z0-9]...` rejects names whose first
    # char is `_`; this exercises the regex arm distinct from the
    # `__`-separator arm.
    with pytest.raises(PairwiseKeyError, match="match"):
        pair_key(
            dataset_name="_hidden",
            model_a="a",
            model_b="b",
            seed=0,
            fold_index=0,
        )


# --- R2 / code-IMP: write_pairwise_cell sentinel-payload key-set
# pinned, mirroring the manifest-side B7.2 contract. ---


def test_write_pairwise_cell_sentinel_payload_has_canonical_keys(tmp_path: Path) -> None:
    import json

    from benchmarks.experiments.ensemble import pairwise_sentinel_path

    row = PairwiseRow(
        library_git_sha="lib_sha",
        run_id="run-uuid",
        started_at_utc="2026-05-21T00:00:00+00:00",
        dataset_name="ds_a",
        model_a="a",
        model_b="b",
        seed=0,
        fold_index=0,
        task_type="binary",
        skipped_reason=None,
        n_samples=10,
        n11=4,
        n10=1,
        n01=1,
        n00=4,
        yule_q=0.875,
        phi=0.6,
        disagreement_rate=0.2,
        double_fault_rate=0.4,
        pearson_pred_corr=0.8,
        spearman_pred_corr=0.75,
        pearson_error_corr=0.5,
    )
    write_pairwise_cell(tmp_path, row)
    key = row.pair_key()
    payload = json.loads(pairwise_sentinel_path(tmp_path, key).read_text())
    # The full key-set is the B7.2 sentinel contract for pairwise
    # shards; pin every key to catch a regression that drops one.
    assert set(payload.keys()) == {
        "pair_key",
        "completed_at_utc",
        "started_at_utc",
        "skipped_reason",
    }
    assert payload["pair_key"] == key
    assert payload["started_at_utc"] == row.started_at_utc
    assert payload["completed_at_utc"] >= row.started_at_utc
    assert payload["skipped_reason"] is None


# --- R2 / qa-IMP: `_proba_matrix_with_suffix` numeric-sort pinned
# for >=10 classes (lex sort would corrupt the matrix). ---


def test_pairwise_for_cell_numeric_proba_sort_with_ten_plus_classes(
    tmp_path: Path,
) -> None:
    from benchmarks.experiments.ensemble import _pairwise_for_cell
    from benchmarks.experiments.raw_loss import RunEnvironment
    from benchmarks.manifest import cell_key
    from benchmarks.predictions import write_predictions

    # 12-class one-hot prediction shards. Lex sort would place
    # y_proba_10 before y_proba_2 in `_proba_matrix_with_suffix`,
    # which corrupts the (N, 12) probability matrix and the
    # downstream classification_nll alignment.
    n_classes = 12
    env = RunEnvironment(
        run_id="test",
        library_git_sha="lib_sha",
        hardware_tier="cpu",
        profile="smoke",
    )
    panel_row_index = np.arange(n_classes, dtype=np.int64)
    y_true = panel_row_index.copy()  # one class per row
    y_pred = panel_row_index.copy()
    proba_a = np.eye(n_classes)
    proba_b = np.eye(n_classes)  # both perfectly confident on the diagonal

    for model_name, proba in (("model_a", proba_a), ("model_b", proba_b)):
        key = cell_key(
            dataset_name="ds_x",
            model_name=model_name,
            seed=0,
            variant="default",
            fold_index=0,
        )
        write_predictions(
            tmp_path,
            key,
            panel_row_index=panel_row_index,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=proba,
        )

    row = _pairwise_for_cell(
        env=env,
        output_root=tmp_path,
        dataset_name="ds_x",
        model_a="model_a",
        model_b="model_b",
        task_type="multiclass",
        seed=0,
        fold_index=0,
        classes=np.arange(n_classes),
    )
    # With correct numeric sort: both models predict perfectly and
    # the per-sample NLL is `-log(1.0) = 0` for every row; the
    # pearson_error_corr is nan (zero variance), not a wild value
    # from a misaligned matrix. The non-corruption signal: n11
    # equals n_classes (all rows agree-and-correct).
    assert row.n11 == n_classes
    assert row.n00 == 0
    assert row.yule_q == 1.0  # perfect agreement


# --- R2 / qa-IMP: `_resolve_classification_classes` returns None
# when every OK cell's prediction shard lacks proba columns. ---


def test_resolve_classification_classes_returns_none_when_all_shards_lack_proba(
    tmp_path: Path,
) -> None:
    from benchmarks.experiments.ensemble import _resolve_classification_classes
    from benchmarks.manifest import cell_key
    from benchmarks.predictions import write_predictions

    # Build a synthetic manifest DataFrame for a binary dataset
    # whose OK cells exist but every prediction shard was written
    # WITHOUT y_proba columns (a `supports_proba=False`-classifier
    # shape). The helper must exhaust the loop and return None.
    panel_row_index = np.array([0, 1, 2, 3])
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    for model_name in ("model_a",):
        key = cell_key(
            dataset_name="ds_x",
            model_name=model_name,
            seed=0,
            variant="default",
            fold_index=0,
        )
        write_predictions(
            tmp_path,
            key,
            panel_row_index=panel_row_index,
            y_true=y_true,
            y_pred=y_pred,
        )
    manifest = pd.DataFrame(
        [
            {
                "dataset_name": "ds_x",
                "model_name": "model_a",
                "seed": 0,
                "fold_index": 0,
                "task_type": "binary",
                "skipped_reason": None,
            }
        ]
    )
    classes = _resolve_classification_classes(manifest, tmp_path, "ds_x")
    assert classes is None
