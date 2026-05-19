"""
Quantile regression
===================

Fit a `TFTRegressor` with quantile heads on a synthetic panel and
read out the per-row, per-quantile predictions. Pinball loss is
the training objective; conformal calibration is available when
``cal_fraction > 0``.
"""

# %%
# Build a tiny regression panel
# -----------------------------
from seq_sklearn import TFTRegressor
from seq_sklearn.config._adapters import TabularConfigParams
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator

gen = SyntheticPanelGenerator(
    target_kind="regression_quantile",
    num_entities=24,
    periods_per_entity=8,
    signal_strength=0.8,
    lookback=6,
    seed=0,
)
panel, y = gen.generate(seed=0)
print(f"panel shape: {panel.shape}, target range: [{y.min():.2f}, {y.max():.2f}]")

# %%
# Fit the quantile regressor
# --------------------------
reg = TFTRegressor(
    task_type="regression_quantile",
    quantiles=(0.1, 0.5, 0.9),
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
).fit(panel, y.astype(float))

# %%
# Predict the quantiles
# ---------------------
quantiles = reg.predict_quantiles(panel)
print(f"predict_quantiles shape: {quantiles.shape}  # (n_rows, 3)")
print(
    f"first row quantiles: 10%={quantiles[0, 0]:.2f}, "
    f"50%={quantiles[0, 1]:.2f}, 90%={quantiles[0, 2]:.2f}"
)
