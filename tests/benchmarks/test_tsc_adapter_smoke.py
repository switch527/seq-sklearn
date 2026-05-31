"""Phase B12 TSC adapter smoke tests (aeon-installed path).

Gated on `pytest.importorskip("aeon")` so the suite stays green
on the default aeon-missing CI. Smoke tests pass a minimum-cost
hyperparameter override (`num_kernels=100`) to keep wall-clock
under 5s and avoid the aeon numba JIT compile path dominating
test time.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

aeon = pytest.importorskip("aeon")


from benchmarks.adapters.tsc import (  # noqa: E402
    _Catch22ClassifierAdapter,
    _MultiRocketClassifierAdapter,
    _RocketClassifierAdapter,
)
from benchmarks.config import DatasetSpec  # noqa: E402


def _binary_spec() -> DatasetSpec:
    return DatasetSpec(
        name="ds_tsc_smoke",
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
        lookback=10,
        positive_label=1,
        citation="tsc smoke",
    )


def _binary_panel() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for entity in range(6):
        # Each entity gets 12 rows so the trailing-window clip
        # exercises both the kept and dropped sides, and the
        # final 10-row window is above aeon MultiRocket's
        # n_timepoints >= 9 floor.
        for t in range(12):
            x = float(rng.normal())
            rows.append(
                {
                    "entity_id": f"e_{entity}",
                    "period": t,
                    "x1": x,
                    "y": int(entity % 2),
                }
            )
    panel = pd.DataFrame(rows)
    return panel, panel["y"].to_numpy()


# --- ROCKET -----------------------------------------------------------------


@pytest.mark.slow
def test_rocket_classifier_fits_predicts_and_proba_sums_to_one() -> None:
    spec = _binary_spec()
    panel, y = _binary_panel()
    adapter = _RocketClassifierAdapter(
        spec=spec,
        hyperparameters={"num_kernels": 100},
    )
    adapter.fit(panel, y)
    y_pred = adapter.predict(panel)
    assert y_pred.shape == (len(panel),)
    y_proba = adapter.predict_proba(panel)
    assert y_proba.shape == (len(panel), 2)
    # Sum-to-1 on rows that are NOT NaN (below-floor rows broadcast
    # NaN; below-floor entities are excluded from the tensor and
    # carry NaN per the A9.1 convention).
    non_nan = ~np.isnan(y_proba).any(axis=1)
    np.testing.assert_allclose(y_proba[non_nan].sum(axis=1), 1.0, atol=1e-6)


# --- MultiRocket ------------------------------------------------------------


@pytest.mark.slow
def test_multirocket_classifier_fits_predicts_and_proba_sums_to_one() -> None:
    spec = _binary_spec()
    panel, y = _binary_panel()
    adapter = _MultiRocketClassifierAdapter(
        spec=spec,
        hyperparameters={"num_kernels": 100, "max_dilations_per_kernel": 8},
    )
    adapter.fit(panel, y)
    y_pred = adapter.predict(panel)
    assert y_pred.shape == (len(panel),)
    y_proba = adapter.predict_proba(panel)
    assert y_proba.shape == (len(panel), 2)
    non_nan = ~np.isnan(y_proba).any(axis=1)
    np.testing.assert_allclose(y_proba[non_nan].sum(axis=1), 1.0, atol=1e-6)


# --- Catch22 ----------------------------------------------------------------


@pytest.mark.slow
def test_catch22_classifier_fits_predicts_and_proba_sums_to_one() -> None:
    spec = _binary_spec()
    panel, y = _binary_panel()
    adapter = _Catch22ClassifierAdapter(spec=spec, hyperparameters={})
    adapter.fit(panel, y)
    y_pred = adapter.predict(panel)
    assert y_pred.shape == (len(panel),)
    y_proba = adapter.predict_proba(panel)
    assert y_proba.shape == (len(panel), 2)
    non_nan = ~np.isnan(y_proba).any(axis=1)
    np.testing.assert_allclose(y_proba[non_nan].sum(axis=1), 1.0, atol=1e-6)


# --- entity-homogeneity (R-B12-2 oracle) ------------------------------------


@pytest.mark.slow
def test_rocket_predict_proba_is_entity_homogeneous() -> None:
    """The R-B12-2 oracle: every row belonging to the same entity
    carries an IDENTICAL probability vector. A buggy broadcast
    that permuted predictions across entities would still satisfy
    shape + sum-to-1 but violate this."""
    spec = _binary_spec()
    panel, y = _binary_panel()
    adapter = _RocketClassifierAdapter(
        spec=spec,
        hyperparameters={"num_kernels": 100},
    )
    adapter.fit(panel, y)
    y_proba = adapter.predict_proba(panel)
    # Group by entity_id; within each group, all rows must carry
    # the same probability vector (or all NaN for a below-floor
    # entity).
    for _entity_id, group in panel.groupby("entity_id"):
        positions = panel.index.get_indexer(group.index)
        rows = y_proba[positions]
        # NaN-skip and identity check on the kept rows.
        kept = ~np.isnan(rows).any(axis=1)
        if kept.sum() <= 1:
            continue
        kept_rows = rows[kept]
        for row in kept_rows[1:]:
            np.testing.assert_array_equal(row, kept_rows[0])
