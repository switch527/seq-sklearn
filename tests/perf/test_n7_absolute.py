"""N7 absolute-budget validation (acceptance criterion 9).

Two functions split by hardware tier:

- ``test_n7_absolute_budgets`` (marked ``gpu`` AND ``slow``): the three
  GPU-and-training N7 numbers (peak memory, training wall-clock, GPU
  inference latency) on an A100/T4/4090.
- ``test_n7_cpu_inference_latency`` (marked ``slow``, CPU-only): the
  CPU inference-latency number, run on the release reference CPU.

Neither runs in PR or nightly CI (the workloads exceed the CI
envelope). Together they discharge criterion 9; the release-checklist
runs both during the v1.0.0 cut and records all four numbers in the
``CHANGELOG.md`` v1.0.0 entry. Phase 11's relative gate does NOT by
itself discharge criterion 9.
"""

import time

import pytest

# N7 reference (docs/requirements.md N7): TFT 128 hidden, 4 heads,
# lookback 12; 100k entities x 24 months x 30 features < 8 GB GPU;
# training < 30 min on A100/T4/4090; whole 1024-window batch
# < 10 ms GPU and < 100 ms CPU. The constants are PER-BATCH budgets,
# matching the spec verbatim; the assertions below time wall-clock
# of a single .predict(batch) call without dividing by batch size.
N7_GPU_MEM_BYTES = 8 * 1024**3
N7_TRAIN_SECONDS = 30 * 60
N7_GPU_INFER_MS = 10.0
N7_CPU_INFER_MS = 100.0


@pytest.mark.gpu
@pytest.mark.slow
def test_n7_absolute_budgets(perf_determinism: None) -> None:
    import os

    if not os.environ.get("SEQ_SKLEARN_N7_GPU"):
        pytest.skip(
            "set SEQ_SKLEARN_N7_GPU=1 on a release-reference GPU "
            "(A100/T4/4090) to record the criterion-9 GPU budgets"
        )

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        pytest.skip("N7 absolute budgets require a CUDA device")

    from seq_sklearn.config.adapters import SchedulerParams, TabularConfigParams
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
        lat.append((time.perf_counter() - t) * 1000.0)
    infer_ms = float(np.median(lat))

    assert peak_mem < N7_GPU_MEM_BYTES, f"N7 memory: {peak_mem} >= {N7_GPU_MEM_BYTES}"
    assert train_seconds < N7_TRAIN_SECONDS, f"N7 train: {train_seconds}s"
    assert infer_ms < N7_GPU_INFER_MS, f"N7 inference (per 1024-batch): {infer_ms}ms"


@pytest.mark.slow
def test_n7_cpu_inference_latency(perf_determinism: None) -> None:
    """N7 CPU-inference budget: a 1024-window batch under 100 ms total.

    Smaller fit workload than the GPU function (the CPU N7 number is
    the latency budget, not the training-throughput budget), so we fit
    a shorter quickstart-shaped panel on CPU then time inference on a
    1024-row batch. Gated behind ``SEQ_SKLEARN_N7_CPU=1`` so it never
    runs incidentally on a developer laptop where the strict per-batch
    budget is unmeasurable; the release engineer sets the env var on
    the release-reference CPU and runs
    ``pytest -m slow tests/perf/test_n7_absolute.py::test_n7_cpu_inference_latency``.
    """
    import os

    if not os.environ.get("SEQ_SKLEARN_N7_CPU"):
        pytest.skip("set SEQ_SKLEARN_N7_CPU=1 on a release-reference CPU to record the criterion-9 CPU inference budget")

    import numpy as np

    from seq_sklearn.config.adapters import SchedulerParams, TabularConfigParams
    from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
    from seq_sklearn.models.transformer.tft.classifier import TFTClassifier

    gen = SyntheticPanelGenerator(
        target_kind="binary",
        num_entities=2_000,
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
    est.fit(panel, y)

    batch = panel.head(1024)
    for _ in range(3):
        est.predict(batch)
    lat = []
    for _ in range(20):
        t = time.perf_counter()
        est.predict(batch)
        lat.append((time.perf_counter() - t) * 1000.0)
    infer_ms = float(np.median(lat))

    assert infer_ms < N7_CPU_INFER_MS, f"N7 CPU inference (per 1024-batch): {infer_ms}ms"
