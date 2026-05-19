"""Phase 11 perf proxy workload (PA.1 / PA.3).

ONE fixed config shared by all three benchmarks so the three metrics
describe the SAME execution. Imports torch + the estimator, so this
module is imported only inside ``perf``-marked benchmark bodies and
the capture CLI, never at `_gate.py` import time (PC.1a).
"""

import numpy as np
import pandas as pd

from seq_sklearn.config.adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier

# Constants live in the torch-free `_constants` module (PC.1a); only
# the heavy builder lives here. Anything that needs only a constant
# imports `_constants`, never this module.
from tests.perf._constants import (
    PROXY_ATTENTION_HEADS,
    PROXY_HIDDEN_SIZE,
    PROXY_L,
    PROXY_N,
    PROXY_P,
    PROXY_SEED,
)

__all__ = ["build_proxy_estimator_and_panel"]


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
