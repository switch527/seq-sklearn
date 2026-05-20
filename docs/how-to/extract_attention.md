# Extract attention and variable-selection weights

TFT's two interpretable surfaces (variable selection and temporal
attention) are returned as a typed dataclass, not computed by a
separate analysis step.

## Variable selection

For each timestep, the TFT learns a softmax weight per input
variable. Reading those weights tells you which features the model
relied on for that observation.

```python
from seq_sklearn import TFTClassifier

clf = TFTClassifier(...).fit(panel_train, y_train)
out = clf.predict_with_attention(panel_test)

# `out` is an immutable `AttentionOutput` dataclass.
out.predictions               # ndarray of class predictions
out.probabilities             # ndarray of class probabilities
out.var_selection_weights     # (n_rows, lookback, n_features) softmax weights
out.static_var_selection_weights  # (n_rows, n_static_features) softmax weights
out.attention_weights         # (n_rows, n_heads, lookback, lookback) per-head attention
```

The column ordering inside `var_selection_weights` matches
`tabular_config.time_varying_real_cols + time_varying_categorical_cols`
(in declaration order). For the static weights, it's
`static_real_cols + static_categorical_cols`.

## Temporal attention

`out.attention_weights` is the per-head self-attention map over the
lookback window, shape `(n_rows, n_heads, lookback, lookback)`. The
last query position is the most-recent timestep; reading its row out
of the attention map gives the per-head distribution of attention
the prediction placed onto past timesteps. A simple "which timesteps
mattered" summary is that row averaged over heads:

```python
per_step_importance = out.attention_weights[:, :, -1, :].mean(axis=1)
# shape (n_rows, lookback)
```

For a single observation `i`, `per_step_importance[i, t]` is how
much the final-timestep prediction attended to past position `t`,
averaged over heads. Index `t = lookback - 1` is the most-recent
timestep; `t = 0` is the oldest.

## A common confusion

Variable-selection weights are computed inside the
*variable-selection network* (VSN) and reflect how much each input is
amplified BEFORE the LSTM and attention layers see it. They are NOT
the same as feature-importance derived from gradients or permutation;
they are an internal mechanism of the architecture.

For a sklearn-style permutation importance, use
`sklearn.inspection.permutation_importance` on a fitted `TFTClassifier`
the same way you would on any sklearn classifier.

## Regression: a slightly different shape

For `TFTRegressor`, the attention output is `RegressionAttentionOutput`.
It has the same `var_selection_weights` / `static_var_selection_weights`
/ `attention_weights` fields, plus `predictions` (or `quantiles` if
the regressor is fit with quantile heads).

The output is BETA-tier in v1: fields may be added in MINOR releases
(but never removed), so prefer attribute access (`out.predictions`)
over tuple unpacking.

```{testcode}
from seq_sklearn import AttentionOutput, RegressionAttentionOutput

assert hasattr(AttentionOutput, "var_selection_weights")
assert hasattr(RegressionAttentionOutput, "var_selection_weights")
```
