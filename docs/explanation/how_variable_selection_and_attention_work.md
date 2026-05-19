# How variable selection and attention work

A conceptual walkthrough of TFT's two interpretable surfaces. This
page is the "why" for what [extract_attention](../how-to/extract_attention)
shows you how to do; the underlying math is in Lim et al. 2021.

## The variable-selection network (VSN)

A panel has many features per timestep. For any given observation,
some are informative and some are noise. The VSN learns a per-row,
per-timestep softmax weight over the features:

- Each input feature is first transformed by its own Gated Residual
  Network (GRN) into a fixed-dimensional embedding.
- A separate GRN produces a softmax over the features (one weight
  per feature, summing to 1 per timestep).
- The per-step input fed forward is the softmax-weighted sum of
  the per-feature embeddings.

The softmax weight is what `predict_with_attention` returns as
`var_selection_weights`. Reading it: "for this row at this
timestep, the model relied on feature X with weight 0.45".

Two important properties:

- **Per-timestep, per-row.** Two rows of the same entity can have
  very different VSN weights at the same timestep — the weight is
  computed from the row's actual feature values, not a global
  per-feature importance.
- **Operates BEFORE the LSTM.** VSN runs at the input stage; the
  LSTM sees the variable-selected combination, not the raw
  features. This is a structural property of the architecture, not
  a post-hoc explainer.

There's a parallel **static VSN** that selects over the
static-covariate inputs (one weight per static feature, no time
axis); same idea, different domain.

## Multi-head attention over the time axis

After the LSTM has consumed the variable-selected sequence, the
TFT applies **interpretable multi-head attention** (a variant of the
standard transformer attention) over the historical timesteps. The
key difference from vanilla MHA: the value projection is *shared*
across heads, so the per-head attention weights are directly
comparable and interpretable as "how much did each head rely on
each timestep".

`attention_weights` shape is `(n_rows, lookback, n_heads)`. A simple
"which timesteps mattered" summary is the mean over heads:

```python
per_step_importance = out.attention_weights.mean(axis=-1)
```

That gives shape `(n_rows, lookback)`. For row `i`,
`per_step_importance[i, t]` is the average attention placed on
`t` steps ago (smaller `t` = more recent).

## What the surfaces are good for

- **Debugging.** A model that places all attention at `t=0` (the
  most-recent timestep) is functionally a snapshot model that's
  ignoring history. Worth asking whether the lookback is actually
  helping.
- **Sanity checks.** Variable-selection weights should match domain
  intuition for a small number of features that obviously matter
  (e.g. recent transaction count for fraud). A trained model that
  selects "static categorical 3" with weight 0.9 for every row is
  probably overfit.
- **Regulatory disclosure.** For domains where you must justify
  individual predictions, the per-row, per-timestep weights are
  directly auditable.

## What the surfaces are NOT

- **Not gradient-based feature importance.** The VSN weight is what
  the model used internally, not what would-be-different-if-this-
  feature-changed.
- **Not a substitute for sklearn permutation importance.** For
  global feature importance with a counterfactual interpretation,
  use `sklearn.inspection.permutation_importance` on the fitted
  estimator.
- **Not a causal explanation.** Attention reflects what the model
  attended to, not what is causally responsible for the outcome.

## Stability

`AttentionOutput` / `RegressionAttentionOutput` are BETA-tier in v1:
fields may be added in MINOR releases but never removed. Use
attribute access (`out.predictions`), not tuple unpacking.

```{testcode}
from seq_sklearn import AttentionOutput

fields = AttentionOutput.__dataclass_fields__
assert "var_selection_weights" in fields
assert "attention_weights" in fields
```
