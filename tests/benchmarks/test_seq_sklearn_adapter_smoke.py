"""End-to-end fit/predict smoke for the seq-sklearn TFT adapters.

Proves the adapter actually works against the library's v1.0.0
façade by fitting on a tiny synthetic panel and asserting shape +
type of the output. Marked ``slow`` per the repo convention
(`pytest-mark` config), so it skips on dev-loop runs but lands in
the default CI suite alongside the existing tests/e2e/ smoke.
"""

import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.seq_sklearn import (
    SeqSklearnTFTClassifierAdapter,
    SeqSklearnTFTRegressorAdapter,
)
from benchmarks.config import DatasetSpec


def _make_spec(*, name: str, task_type: str) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        task_type=task_type,  # pyright: ignore[reportArgumentType]
        access_tier="OPEN",
        size_tier="small",
        balance="balanced",
        modality="numeric",
        source_uri="https://example.com/data.csv",
        integrity_sha256="0" * 64,
        archive_basename="data.csv",
        entity_col="entity_id",
        time_col="cycle",
        target_col="y",
        feature_real_cols=("x0", "x1"),
        feature_categorical_cols=(),
        lookback=4,
        observation_cutoff_rule=None,
        densification_policy=None,
        positive_label=1 if task_type == "binary" else None,
        excluded=False,
        exclusion_reason=None,
        citation="Acme 2020",
    )


def _make_panel(*, n_entities: int = 4, n_periods: int = 6) -> pd.DataFrame:
    """Tiny synthetic panel: one row per (entity, cycle), two real
    features (a deterministic ramp + a stochastic noise channel)."""
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for e in range(n_entities):
        for t in range(n_periods):
            rows.append(
                {
                    "entity_id": e,
                    "cycle": t,
                    "x0": float(t),
                    "x1": float(rng.normal()),
                }
            )
    return pd.DataFrame(rows)


def _binary_y(panel: pd.DataFrame) -> np.ndarray:
    """Bernoulli targets correlated with `x0` so the model has signal."""
    rng = np.random.default_rng(0)
    p = 1.0 / (1.0 + np.exp(-(panel["x0"].to_numpy() - 2.5)))
    return (rng.random(len(panel)) < p).astype(np.int64)


def _regression_y(panel: pd.DataFrame) -> np.ndarray:
    """Linear-in-x0 target with mild noise."""
    rng = np.random.default_rng(0)
    return (
        panel["x0"].to_numpy()
        + 0.5 * panel["x1"].to_numpy()
        + rng.normal(scale=0.1, size=len(panel))
    ).astype(np.float64)


@pytest.mark.slow
def test_tft_classifier_adapter_fits_and_predicts_smoke() -> None:
    spec = _make_spec(name="ds_binary", task_type="binary")
    panel = _make_panel()
    y = _binary_y(panel)

    adapter = SeqSklearnTFTClassifierAdapter(
        spec=spec,
        hyperparameters={
            "hidden_size": 16,
            "attention_heads": 2,
            "max_epochs": 1,
            "batch_size": 8,
            "val_fraction": 0.25,
            "cal_fraction": 0.0,
            "precision": "32-true",
            "seed": 0,
        },
    )
    adapter.fit(panel, y)

    preds = adapter.predict(panel)
    assert preds.shape == (len(panel),)
    assert preds.dtype.kind in ("i", "u")

    proba = adapter.predict_proba(panel)
    assert proba.shape == (len(panel), 2)
    assert (proba >= 0).all()
    assert (proba <= 1).all()


@pytest.mark.slow
def test_tft_regressor_adapter_fits_and_predicts_smoke() -> None:
    spec = _make_spec(name="ds_reg", task_type="regression_point")
    panel = _make_panel()
    y = _regression_y(panel)

    adapter = SeqSklearnTFTRegressorAdapter(
        spec=spec,
        hyperparameters={
            "hidden_size": 16,
            "attention_heads": 2,
            "max_epochs": 1,
            "batch_size": 8,
            "val_fraction": 0.25,
            "cal_fraction": 0.0,
            "precision": "32-true",
            "seed": 0,
        },
    )
    adapter.fit(panel, y)

    preds = adapter.predict(panel)
    assert preds.shape == (len(panel),)
    assert preds.dtype.kind == "f"
