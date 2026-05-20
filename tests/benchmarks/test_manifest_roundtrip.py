"""Phase B5 manifest tests (B7.2 atomic shard-then-sentinel).

Walks the three contracts the manifest module guarantees:

- `cell_key` rejects FS-unsafe tokens (`__` in a dataset name, etc.).
- `write_cell` lands the shard atomically and only writes the
  sentinel AFTER the shard is durably renamed (a crash between the
  two leaves the cell re-runnable on the next attempt; the second
  attempt overwrites the orphan shard atomically).
- `load_run` round-trips a manifest of write_cell-produced shards
  into a single DataFrame and equals what was written.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.manifest import (
    CellKeyError,
    ResultRow,
    cell_key,
    is_cell_complete,
    list_completed_keys,
    load_run,
    metrics_to_row_fields,
    pinball_from_json,
    pinball_to_json,
    results_dir,
    sentinel_dir,
    sentinel_path,
    shard_path,
    write_cell,
)


def _minimal_row(
    *,
    dataset_name: str = "ds_a",
    model_name: str = "tft_classifier",
    seed: int = 0,
    fold_index: int = 0,
    skipped_reason: str | None = None,
    log_loss: float | None = 0.5,
) -> ResultRow:
    """Build a `ResultRow` with the smallest valid field set."""
    return ResultRow(
        library_git_sha="lib_sha_00000000",
        benchmarks_git_sha="bench_sha_00000000",
        run_id="run-uuid-test",
        started_at_utc="2026-05-20T00:00:00+00:00",
        dataset_name=dataset_name,
        model_name=model_name,
        seed=seed,
        task_type="binary",
        fold_index=fold_index,
        n_folds=5,
        split_fingerprint="0" * 64,
        l_resolved=8,
        hardware_tier="cpu",
        profile="standard",
        skipped_reason=skipped_reason,
        log_loss=log_loss,
    )


def test_cell_key_round_trip_format() -> None:
    key = cell_key(
        dataset_name="ds_a",
        model_name="tft_classifier",
        seed=7,
        variant="default",
        fold_index=3,
    )
    assert key == "ds_a__tft_classifier__seed_7__default__fold_3"


def test_cell_key_rejects_double_underscore_in_token() -> None:
    with pytest.raises(CellKeyError, match="separator"):
        cell_key(
            dataset_name="ds__broken",
            model_name="tft_classifier",
            seed=0,
            variant="default",
            fold_index=0,
        )


def test_cell_key_rejects_leading_underscore_in_token() -> None:
    # The regex requires the first char to be `[A-Za-z0-9]`; a
    # leading underscore aliases the sentinel-directory prefix.
    with pytest.raises(CellKeyError, match="match"):
        cell_key(
            dataset_name="_hidden",
            model_name="tft_classifier",
            seed=0,
            variant="default",
            fold_index=0,
        )


def test_cell_key_rejects_negative_fold_index() -> None:
    with pytest.raises(CellKeyError, match="fold_index"):
        cell_key(
            dataset_name="ds_a",
            model_name="tft_classifier",
            seed=0,
            variant="default",
            fold_index=-1,
        )


def test_write_cell_produces_shard_then_sentinel(tmp_path: Path) -> None:
    row = _minimal_row()
    key = row.cell_key()
    write_cell(tmp_path, row)
    assert shard_path(tmp_path, key).exists()
    assert sentinel_path(tmp_path, key).exists()
    sentinel_payload = json.loads(sentinel_path(tmp_path, key).read_text())
    assert sentinel_payload["cell_key"] == key
    assert sentinel_payload["completed_at_utc"] == row.started_at_utc
    assert sentinel_payload["skipped_reason"] is None


def test_is_cell_complete_returns_false_before_write_and_true_after(
    tmp_path: Path,
) -> None:
    row = _minimal_row()
    key = row.cell_key()
    assert not is_cell_complete(tmp_path, key)
    write_cell(tmp_path, row)
    assert is_cell_complete(tmp_path, key)


def test_load_run_round_trips_one_cell(tmp_path: Path) -> None:
    row = _minimal_row(log_loss=0.42)
    write_cell(tmp_path, row)
    manifest = load_run(tmp_path)
    assert len(manifest) == 1
    assert manifest.iloc[0]["dataset_name"] == row.dataset_name
    assert manifest.iloc[0]["log_loss"] == pytest.approx(0.42)
    # The skipped_reason None round-trips as a null (pandas treats
    # the value as NaN under the parquet's nullable-string schema;
    # the assertion is on the manifest's "is null" semantics).
    assert pd.isna(manifest.iloc[0]["skipped_reason"])


def test_load_run_concatenates_multiple_shards_sorted_by_key(
    tmp_path: Path,
) -> None:
    rows = [
        _minimal_row(dataset_name="ds_b", model_name="model_x", log_loss=0.30),
        _minimal_row(dataset_name="ds_a", model_name="model_x", log_loss=0.50),
        _minimal_row(dataset_name="ds_a", model_name="model_y", log_loss=0.40),
    ]
    for row in rows:
        write_cell(tmp_path, row)
    manifest = load_run(tmp_path)
    assert len(manifest) == 3
    # `load_run` sorts shards by filename so the manifest is
    # byte-stable across runs. cell_key starts with the dataset
    # name, so dataset 'ds_a' rows come first.
    names = manifest["dataset_name"].tolist()
    assert names == sorted(names)


def test_load_run_returns_empty_dataframe_when_no_shards(tmp_path: Path) -> None:
    manifest = load_run(tmp_path)
    assert manifest.empty
    # The columns are still the ResultRow schema so a `pd.concat`
    # with a non-empty frame would not lose columns.
    assert set(manifest.columns) >= {
        "dataset_name",
        "model_name",
        "log_loss",
        "rmse",
        "skipped_reason",
    }


def test_load_run_raises_when_root_does_not_exist(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_run(missing)


def test_list_completed_keys_reflects_sentinel_writes(tmp_path: Path) -> None:
    assert list_completed_keys(tmp_path) == ()
    row = _minimal_row(dataset_name="ds_a", seed=0)
    write_cell(tmp_path, row)
    keys = list_completed_keys(tmp_path)
    assert keys == (row.cell_key(),)


def test_second_write_is_idempotent_on_same_cell(tmp_path: Path) -> None:
    # The B7.2 contract: a crash between shard rename and sentinel
    # write re-runs the cell; the SECOND write_cell call overwrites
    # the orphan shard atomically. Simulated by calling write_cell
    # twice and asserting the second call's row contents win.
    write_cell(tmp_path, _minimal_row(log_loss=0.50))
    write_cell(tmp_path, _minimal_row(log_loss=0.10))
    manifest = load_run(tmp_path)
    assert len(manifest) == 1
    assert manifest.iloc[0]["log_loss"] == pytest.approx(0.10)


def test_results_dir_and_sentinel_dir_are_distinct(tmp_path: Path) -> None:
    # The two directories are intentionally distinct so a glob over
    # one does not pick up files from the other (the sentinel files
    # live under `results/_done/`, never beside the shards).
    assert sentinel_dir(tmp_path) == results_dir(tmp_path) / "_done"


def test_skipped_cell_round_trips_reason(tmp_path: Path) -> None:
    row = _minimal_row(skipped_reason="task_type_mismatch: ...", log_loss=None)
    write_cell(tmp_path, row)
    manifest = load_run(tmp_path)
    assert manifest.iloc[0]["skipped_reason"] == row.skipped_reason
    # Skipped rows carry None for the metric columns.
    assert pd.isna(manifest.iloc[0]["log_loss"])
    # Sentinel payload reflects the skip too.
    sentinel = json.loads(sentinel_path(tmp_path, row.cell_key()).read_text())
    assert sentinel["skipped_reason"] == row.skipped_reason


def test_pinball_json_round_trip() -> None:
    pinball = {"pinball_q0.10": 0.05, "pinball_q0.50": 0.10, "pinball_q0.90": 0.07}
    encoded = pinball_to_json(pinball)
    assert encoded is not None
    decoded = pinball_from_json(encoded)
    assert decoded == pinball


def test_pinball_json_handles_none() -> None:
    assert pinball_to_json(None) is None
    assert pinball_from_json(None) is None


def test_metrics_to_row_fields_flattens_pinball() -> None:
    # `compute_all`'s regression_quantile dump carries `pinball` as a
    # nested dict; the row schema flattens to `pinball_json` so a
    # parquet shard has a uniform shape across task types.
    metrics_dump = {
        "rmse": 0.5,
        "mae": 0.4,
        "r2": 0.9,
        "mape": 0.1,
        "mape_skip_reason": None,
        "pinball": {"pinball_q0.50": 0.25},
    }
    fields = metrics_to_row_fields(metrics_dump)
    assert "pinball" not in fields
    assert fields["pinball_json"] == '{"pinball_q0.50":0.25}'
    assert fields["rmse"] == 0.5


def test_metrics_to_row_fields_passes_through_when_no_pinball() -> None:
    metrics_dump = {"log_loss": 0.7, "accuracy": 0.8}
    fields = metrics_to_row_fields(metrics_dump)
    assert fields == metrics_dump


def test_result_row_forbids_extra_fields() -> None:
    # The B4 records and the manifest record share the same
    # `extra="forbid"` contract; a typo'd metric name at write time
    # surfaces as a `ValidationError`, not a silently-dropped column.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ResultRow(
            library_git_sha="lib_sha",
            benchmarks_git_sha="bench_sha",
            run_id="run",
            started_at_utc="2026-05-20T00:00:00+00:00",
            dataset_name="ds_a",
            model_name="m",
            seed=0,
            task_type="binary",
            fold_index=0,
            n_folds=1,
            split_fingerprint="0" * 64,
            l_resolved=1,
            hardware_tier="cpu",
            profile="standard",
            not_a_real_field=1.0,  # pyright: ignore[reportCallIssue]
        )


def test_result_row_is_frozen() -> None:
    from pydantic import ValidationError

    row = _minimal_row()
    with pytest.raises(ValidationError):
        row.log_loss = 0.99  # pyright: ignore[reportAttributeAccessIssue]
