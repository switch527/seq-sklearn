"""
Binary classifier with attention
================================

Fit a tiny `TFTClassifier` on a synthetic binary panel and read out
the variable-selection and temporal-attention surfaces. CPU,
``max_epochs=1``: the gallery is a teaching surface, not a
benchmarking one.
"""

# %%
# Build a small synthetic panel
# -----------------------------
from seq_sklearn import TabularConfigParams, TFTClassifier
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator

gen = SyntheticPanelGenerator(
    target_kind="binary",
    num_entities=32,
    periods_per_entity=8,
    signal_strength=0.9,
    lookback=6,
    seed=0,
)
panel, y = gen.generate(seed=0)
print(f"panel shape: {panel.shape}, positive rate: {y.mean():.2f}")

# %%
# Fit and predict
# ---------------
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

proba = clf.predict_proba(panel)
print(f"predict_proba shape: {proba.shape}")

# %%
# Read the interpretability surfaces
# ----------------------------------
out = clf.predict_with_attention(panel)
print(f"variable-selection weights shape: {out.var_selection_weights.shape}")
print(f"static variable-selection weights shape: {out.static_var_selection_weights.shape}")
print(f"temporal attention shape: {out.attention_weights.shape}")
