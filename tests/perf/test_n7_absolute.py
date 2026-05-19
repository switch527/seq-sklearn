"""N7 absolute-budget validation (acceptance criterion 9).

Marked `gpu` AND `slow` so it never runs in PR or nightly CI (the N7
reference workload exceeds the CI envelope). It is the documented
evidence for `docs/requirements.md` criterion 9 and is run manually
on an A100/T4/4090 as the Phase 12 release-checklist step
(`pytest -m "gpu and slow" tests/perf/test_n7_absolute.py`). Phase
11's relative gate does NOT by itself discharge criterion 9.
"""

import time

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

# N7 reference (docs/requirements.md N7): TFT 128 hidden, 4 heads,
# lookback 12; 100k entities x 24 months x 30 features < 8 GB GPU;
# training < 30 min on A100/T4/4090; batch of 1024 windows < 10 ms GPU.
N7_GPU_MEM_BYTES = 8 * 1024**3
N7_TRAIN_SECONDS = 30 * 60
N7_GPU_INFER_MS = 10.0


def test_n7_absolute_budgets(perf_determinism: None) -> None:
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        pytest.skip("N7 absolute budgets require a CUDA device")

    from seq_sklearn.config._adapters import SchedulerParams, TabularConfigParams
    from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
    from seq_sklearn.models.transformer.tft.classifier import TFTClassifier

    gen = SyntheticPanelGenerator(
        target_kind="binary",
        num_entities=100_000,
        periods_per_entity=24,
        lookback=12,
        signal_strength=0.8,
        seed=11,
    )
    panel, y = gen.generate(seed=11)
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
        scheduler=SchedulerParams(name="constant", warmup_steps=0),
        hidden_size=128,
        attention_heads=4,
        max_epochs=1,
        batch_size=256,
        val_fraction=0.1,
        cal_fraction=0.0,
        precision="32-true",
        seed=11,
        verbose=False,
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    est.fit(panel, y)
    train_seconds = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated()

    batch = panel.head(1024)
    for _ in range(3):
        est.predict(batch)
    lat = []
    for _ in range(20):
        t = time.perf_counter()
        est.predict(batch)
        lat.append((time.perf_counter() - t) / len(batch) * 1000.0)
    infer_ms = float(np.median(lat))

    assert peak_mem < N7_GPU_MEM_BYTES, f"N7 memory: {peak_mem} >= {N7_GPU_MEM_BYTES}"
    assert train_seconds < N7_TRAIN_SECONDS, f"N7 train: {train_seconds}s"
    assert infer_ms < N7_GPU_INFER_MS, f"N7 inference: {infer_ms}ms/sample"
