"""Loader determinism conformance tests (Gemini-C1).

The B13 bootstrap-rollup aggregator re-loads each dataset's
panel at aggregation time and indexes by `panel_row_index`. The
contract is: every registered loader produces ROW-IDENTICAL
output across calls (same length, same row order, same column
values). These tests pin the contract for every registered
dataset AND assert the conformance check itself is NOT
vacuously passing.
"""

from pathlib import Path
from typing import Any

import benchmarks.adapters  # type: ignore[reportUnusedImport]
import benchmarks.datasets  # type: ignore[reportUnusedImport]  # noqa: F401
import numpy as np
import pandas as pd
from benchmarks.config import DatasetSpec
from benchmarks.datasets._base import PanelDataset
from benchmarks.registry import get_loader, list_datasets, register_dataset


def _panels_equal(a: PanelDataset, b: PanelDataset) -> bool:
    if len(a.panel) != len(b.panel):
        return False
    if list(a.panel.columns) != list(b.panel.columns):
        return False
    if not a.panel.equals(b.panel):
        return False
    return bool(np.array_equal(a.y, b.y))


def test_every_registered_loader_row_order_is_deterministic_across_calls(
    tmp_path: Path,
) -> None:
    """For every registered dataset, two calls to its loader
    must produce row-identical output (length, order, values).
    B13 aggregator's entity_id re-resolution depends on this."""
    from tests.benchmarks._fakes import register_all_fakes_and_get_panels

    register_all_fakes_and_get_panels()
    cache_root = tmp_path / "loader_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    nondeterministic: list[str] = []
    skipped: list[tuple[str, str]] = []
    for name in list_datasets():
        loader = get_loader(name)
        try:
            first = loader(cache_root)
            second = loader(cache_root)
        except Exception as exc:
            skipped.append((name, f"{type(exc).__name__}: {exc}"))
            continue
        if not _panels_equal(first, second):
            nondeterministic.append(name)
    assert not nondeterministic, (
        f"loaders produced non-row-identical output across two calls "
        f"(B13 entity_id re-resolution depends on loader determinism): "
        f"{nondeterministic}; skipped (gated): {[s[0] for s in skipped]}"
    )


def test_conformance_check_fails_on_deliberately_nondeterministic_loader() -> None:
    """The conformance machinery (the `_panels_equal` predicate)
    must DETECT a non-deterministic loader. Register a synthetic
    loader that shuffles rows on every call; assert the
    equality check returns False. Proves the conformance test
    is not vacuously passing."""
    spec = DatasetSpec(
        name="nondeterministic_test_loader",
        task_type="binary",
        access_tier="OPEN",
        size_tier="small",
        balance="balanced",
        modality="numeric",
        source_uri="https://example.test/x.csv",
        integrity_sha256="0" * 64,
        archive_basename="x.csv",
        entity_col="entity_id",
        time_col="period",
        target_col="y",
        feature_real_cols=("x1",),
        feature_categorical_cols=(),
        lookback=2,
        positive_label=1,
        citation="conformance negative-path test",
    )

    panel = pd.DataFrame(
        [{"entity_id": f"e_{i}", "period": 0, "x1": float(i), "y": i % 2} for i in range(8)]
    )
    y = panel["y"].to_numpy()

    call_count: list[int] = [0]

    def _shuffling_loader(_root: Path, *_: Any, **__: Any) -> PanelDataset:
        call_count[0] += 1
        # Shuffle on every call EXCEPT the first so two
        # successive calls return non-identical panels.
        if call_count[0] == 1:
            return PanelDataset(spec=spec, panel=panel.copy(), y=y.copy())
        permuted = panel.iloc[::-1].reset_index(drop=True)
        return PanelDataset(spec=spec, panel=permuted, y=y[::-1].copy())

    register_dataset(spec, _shuffling_loader)
    first = _shuffling_loader(Path())
    second = _shuffling_loader(Path())
    assert not _panels_equal(first, second), (
        "the conformance check FAILED to detect a non-deterministic "
        "loader; the conformance test is vacuously passing"
    )
