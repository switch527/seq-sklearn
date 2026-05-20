"""N1 GPU/CPU determinism parity (Phase 7, nightly only).

CPU FP32-deterministic and CUDA FP32 must agree within
``atol=1e-5, rtol=1e-5``. Marked ``gpu`` + ``slow`` so the per-PR CI
(CPU-only) skips it; the nightly ``gpu`` job (A19) runs it. Skips
cleanly when no CUDA device is present.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from seq_sklearn.config.adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.determinism]


def _panel() -> tuple[SyntheticPanelGenerator, pd.DataFrame, np.ndarray]:
    gen = SyntheticPanelGenerator(
        target_kind="binary",
        num_entities=20,
        periods_per_entity=24,
        signal_strength=0.7,
        lookback=8,
        seed=42,
    )
    panel, y = gen.generate(seed=42)
    return gen, panel, y


def _estimator(gen: SyntheticPanelGenerator) -> TFTClassifier:
    return TFTClassifier(
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
        scheduler=SchedulerParams(name="constant", warmup_steps=0),
        hidden_size=16,
        attention_heads=2,
        max_epochs=3,
        batch_size=16,
        val_fraction=0.2,
        cal_fraction=0.0,
        precision="32-true",
        seed=42,
        verbose=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tft_classifier_gpu_cpu_predict_proba_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from seq_sklearn.hardware import HardwareTier

    gen, x, y = _panel()

    # CPU run: pin hardware detection to CPU.
    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    cpu_proba = _estimator(gen).fit(x, y).predict_proba(x)

    # GPU run: restore real detection so the Trainer selects CUDA.
    monkeypatch.undo()
    gpu_proba = _estimator(gen).fit(x, y).predict_proba(x)

    assert np.allclose(cpu_proba, gpu_proba, atol=1e-5, rtol=1e-5)
