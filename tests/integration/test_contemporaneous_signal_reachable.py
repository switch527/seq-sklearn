"""#4: the contemporaneous default config can learn an F6 signal.

The Phase 9 root cause was ``prediction_step=1`` re-aligning the
already-contemporaneous F6 panel into a 1-step forecast, which
collapsed TFT accuracy (GBDT ~0.98 vs TFT ~0.68 at ps=1, ~0.94 at
ps=0). This fast integration test pins that a regression back to
``prediction_step=1`` is caught in the inner loop, not only the
nightly slow e2e: a default-config classifier on a small binary
panel clears a loose accuracy floor.
"""

import numpy as np
import pytest
import torch

from seq_sklearn.config._adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier

# NOT slow: ledger #4 requires this to run in the inner `pytest -m
# "not slow"` loop so a regression to prediction_step=1 is caught
# there, not only the nightly e2e. Tuned to a <30s footprint.
pytestmark = pytest.mark.integration


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def test_signal_reachable_default_config_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_cpu(monkeypatch)
    # Footprint tuned against a measured run (S4-pinned): ~8s on CPU,
    # acc ~0.80 at the contemporaneous default, ~0.68 if regressed to
    # prediction_step=1. Comfortably inside the <30s inner-loop budget.
    gen = SyntheticPanelGenerator(
        target_kind="binary",
        num_entities=36,
        periods_per_entity=14,
        signal_strength=0.97,
        noise_level=0.03,
        lookback=6,
        seed=42,
    )
    panel, y = gen.generate(seed=42)
    tab = TabularConfigParams(
        id_col=gen.id_col,
        time_col=gen.time_col,
        static_categorical_cols=tuple(gen.static_categorical_cols),
        static_real_cols=tuple(gen.static_real_cols),
        time_varying_real_cols=tuple(gen.time_varying_real_cols),
        time_varying_categorical_cols=tuple(gen.time_varying_categorical_cols),
        lookback=gen.lookback,
        max_categorical_cardinality=10_000,
    )
    # prediction_step is intentionally NOT set: this exercises the
    # contemporaneous default. A regression to =1 fails this floor.
    assert tab.prediction_step == 0

    est = TFTClassifier(
        task_type="binary",
        tabular_config=tab,
        scheduler=SchedulerParams(name="constant", warmup_steps=0),
        hidden_size=16,
        attention_heads=2,
        max_epochs=12,
        batch_size=64,
        val_fraction=0.2,
        cal_fraction=0.0,
        precision="32-true",
        verbose=False,
        seed=42,
    ).fit(panel, y)

    pred = est.predict(panel)
    finite = (
        np.isfinite(np.asarray(pred, dtype=float))
        if pred.dtype != object
        else np.ones(len(pred), bool)
    )
    acc = float(np.mean(pred[finite] == y[finite]))
    majority = max(float(np.mean(y)), 1.0 - float(np.mean(y)))
    # Provisional loose floor (S4-pinned): well above majority-class and
    # far above the ~0.68 ps=1 regression level.
    assert acc >= 0.70, f"train accuracy {acc:.3f} below floor (majority {majority:.3f})"
