"""
Extract attention weights
=========================

`predict_with_attention` returns the variable-selection and
temporal-attention surfaces alongside predictions as an immutable
dataclass. This example shows how to read them.
"""

# %%
# Fit a tiny classifier
# ---------------------
import numpy as np

from seq_sklearn import TFTClassifier
from seq_sklearn.config._adapters import TabularConfigParams
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator

gen = SyntheticPanelGenerator(
    target_kind="binary",
    num_entities=24,
    periods_per_entity=6,
    signal_strength=0.9,
    lookback=6,
    seed=0,
)
panel, y = gen.generate(seed=0)

clf = TFTClassifier(
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
    hidden_size=16,
    attention_heads=2,
    max_epochs=1,
    batch_size=32,
    val_fraction=0.2,
    cal_fraction=0.0,
    precision="32-true",
    verbose=False,
    seed=0,
).fit(panel, y)

# %%
# `predict_with_attention` returns a frozen dataclass
# ---------------------------------------------------
out = clf.predict_with_attention(panel)
print("predictions      :", out.predictions.shape)
print("probabilities    :", out.probabilities.shape)
print("VSN weights      :", out.var_selection_weights.shape)
print("Static VSN weights:", out.static_var_selection_weights.shape)
print("attention weights:", out.attention_weights.shape)

# %%
# Per-step importance is the mean over heads
# ------------------------------------------
per_step = out.attention_weights.mean(axis=-1)
print("per-step importance shape:", per_step.shape)
print("row 0 importance (most-recent is index 0):", np.round(per_step[0], 3))
