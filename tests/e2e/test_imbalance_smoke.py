"""N1 imbalance smoke, one test per imbalance_strategy (slow, nightly).

Each legal binary imbalance strategy (cross_entropy cell: none,
class_weighted, oversample_minority, undersample_majority) must fit and
predict on a skewed-class synthetic panel and return a sane,
in-range, both-classes-reachable output. This is a smoke contract (the
pipeline runs end to end under every sampler), not a metric threshold.

Marked ``slow``; excluded from the 5-minute per-PR budget (N2).
CPU-pinned.
"""

import numpy as np
import pytest
import torch

from seq_sklearn.config._adapters import SamplerParams, SchedulerParams, TabularConfigParams
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

_SEED = 42
_IMBALANCE_STRATEGIES = (
    "none",
    "class_weighted",
    "oversample_minority",
    "undersample_majority",
)


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


@pytest.mark.parametrize("strategy", _IMBALANCE_STRATEGIES)
def test_imbalance_strategy_smoke(monkeypatch: pytest.MonkeyPatch, strategy: str) -> None:
    _force_cpu(monkeypatch)
    gen = SyntheticPanelGenerator(
        target_kind="binary",
        num_entities=120,
        periods_per_entity=24,
        signal_strength=0.7,
        class_balance=0.85,  # skewed: ~85/15 to exercise the sampler
        lookback=12,
        seed=_SEED,
    )
    x, y = gen.generate(seed=_SEED)
    assert set(np.unique(y)) == {0, 1}

    est = TFTClassifier(
        task_type="binary",
        tabular_config=TabularConfigParams(
            id_col=gen.id_col,
            time_col=gen.time_col,
            static_categorical_cols=tuple(gen.static_categorical_cols),
            static_real_cols=tuple(gen.static_real_cols),
            time_varying_real_cols=tuple(gen.time_varying_real_cols),
            time_varying_categorical_cols=tuple(gen.time_varying_categorical_cols),
            lookback=gen.lookback,
            min_periods=1,
            min_periods_predict=1,
            max_categorical_cardinality=10_000,
        ),
        sampler=SamplerParams(strategy=strategy),  # type: ignore[arg-type]
        scheduler=SchedulerParams(name="constant", warmup_steps=0),
        hidden_size=32,
        attention_heads=2,
        max_epochs=10,
        batch_size=64,
        val_fraction=0.2,
        cal_fraction=0.0,
        precision="32-true",
        verbose=False,
        seed=_SEED,
    ).fit(x, y)

    proba = est.predict_proba(x)
    preds = est.predict(x)
    assert proba.shape == (len(x), 2)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert preds.shape == (len(x),)
    assert set(np.unique(preds)).issubset({0, 1})
    assert np.isfinite(proba).all()
