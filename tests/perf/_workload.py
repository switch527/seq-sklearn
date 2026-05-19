"""Phase 11 perf proxy workload (PA.1 / PA.3).

ONE fixed config shared by all three benchmarks so the three metrics
describe the SAME execution. Imports torch + the estimator, so this
module is imported only inside ``perf``-marked benchmark bodies and
the capture CLI, never at `_gate.py` import time (PC.1a).
"""

import numpy as np
import pandas as pd

from seq_sklearn.config._adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier

__all__ = [
    "BENCH_MIN_ROUNDS",
    "INFERENCE_BATCH",
    "INFERENCE_REPEATS",
    "INFERENCE_WARMUP",
    "PEAK_MEM_CHILD_TIMEOUT_S",
    "PROXY_ATTENTION_HEADS",
    "PROXY_BUILD_TIMEOUT_S",
    "PROXY_HIDDEN_SIZE",
    "PROXY_L",
    "PROXY_N",
    "PROXY_P",
    "PROXY_SEED",
    "build_proxy_estimator_and_panel",
]

# Proxy size (PA.1): L matches the N7 reference lookback; N/P scaled
# for a CPU nightly envelope. Changing any of these invalidates the
# checked-in baselines and is a PERF_BASELINE_REVIEWED: change.
PROXY_N = 256
PROXY_P = 24
PROXY_L = 12
PROXY_SEED = 11
PROXY_HIDDEN_SIZE = 128  # N7 reference architecture
PROXY_ATTENTION_HEADS = 4  # N7 reference architecture

# Named tuning constants (PD.3 / arch R3-N; PG.8 asserts they are
# the values actually passed through).
BENCH_MIN_ROUNDS = 5
INFERENCE_WARMUP = 3
INFERENCE_REPEATS = 20
INFERENCE_BATCH = 1024  # N7 latency reference size

# Subprocess timeouts (qa-I1 / R1): an oversized proxy or a
# crashed/OOM-killed child must fail loudly, never hang.
PROXY_BUILD_TIMEOUT_S = 180
PEAK_MEM_CHILD_TIMEOUT_S = 240


def build_proxy_estimator_and_panel() -> tuple[TFTClassifier, pd.DataFrame, np.ndarray]:
    """The single proxy used by all three benchmarks (PA.3).

    Returns an UNFITTED ``TFTClassifier`` plus the panel and target;
    each benchmark decides what to time (fit step, fit+predict peak,
    predict latency). Determinism is enabled by the perf fixture
    (PG.5) before any benchmark runs (PA.2).
    """
    gen = SyntheticPanelGenerator(
        target_kind="binary",
        num_entities=PROXY_N,
        periods_per_entity=PROXY_P,
        lookback=PROXY_L,
        signal_strength=0.8,
        seed=PROXY_SEED,
    )
    panel, y = gen.generate(seed=PROXY_SEED)
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
        hidden_size=PROXY_HIDDEN_SIZE,
        attention_heads=PROXY_ATTENTION_HEADS,
        max_epochs=2,
        batch_size=64,
        val_fraction=0.2,
        cal_fraction=0.0,
        precision="32-true",
        seed=PROXY_SEED,
        verbose=False,
    )
    return est, panel, y
