# Why a TFT, adapted to classification

## seq-sklearn is NOT a forecasting library

This is the first thing to read. The Temporal Fusion Transformer
(Lim et al., 2021) was designed for multi-horizon time-series
forecasting. seq-sklearn adapts the architecture to **supervised
classification and regression on panel data**, which is a
structurally different task:

- **Forecasting:** given a sequence up to time `t`, predict its
  continuation `[t+1, ..., t+h]`. The target is the same variable
  as the input, shifted into the future.
- **Classification/regression (what we do):** given a sequence of
  features up to time `t`, predict a separate label or value. The
  target is *not* the input continued.

These are different tasks. They share the same observation shape
(time-indexed multivariate data) but the modeling goal, the loss
function, the evaluation metrics, and the cross-validation strategy
all diverge.

If your problem is "predict the next month's value", reach for a
forecasting library (pytorch-forecasting, darts, neuralforecast,
sktime). If your problem is "given a customer's history, predict
churn / conversion / risk", seq-sklearn is the right tool.

## Why TFT, of all architectures

TFT has three properties that make it a strong default for panel
classification:

1. **Mixed input types out of the box.** The variable-selection
   network handles static-real, static-categorical, time-varying-
   real, and time-varying-categorical inputs in a single fitted
   path. Most sequence libraries require you to pre-engineer
   features into a single tensor; TFT consumes them mixed.
2. **Built-in interpretable surfaces.** Variable-selection weights
   and temporal attention are computed inside the forward pass,
   not bolted on as a post-hoc explainer. For regulated /
   high-stakes domains (credit, healthcare, churn) the
   interpretability is a first-class deliverable, not an
   afterthought.
3. **Stable training in modest data regimes.** TFT's gated
   residual networks and LayerNorm-everywhere design converge
   reliably with limited tuning, where pure transformer stacks
   often need heroics.

The cost is parameter count (a TFT is heavier than a small LSTM or
a Catch22+GBM baseline) and inference latency relative to gradient
boosting. For interpretability-required, mixed-input, modestly-
sized panel-classification problems, the trade is favorable.

## What changed from the forecasting paper

The forecasting TFT terminates in a per-horizon quantile head. The
seq-sklearn adaptation:

- **Head + loss swap.** Classification: a softmax/sigmoid head with
  cross-entropy or BCE. Regression: a point head with MSE, or a
  quantile head with pinball loss when `task_type="regression_quantile"`.
- **No multi-horizon machinery.** The model emits one prediction per
  window, not a horizon-indexed sequence.
- **Calibration layer (optional).** Temperature scaling on a held-out
  calibration fold; conformal quantile correction for the quantile
  regressor.

Everything else (variable selection, static covariate encoders, LSTM
backbone, interpretable multi-head attention, gated residual
networks) is unchanged from the original.

## What we do NOT do

- We do NOT support forecasting (predicting future values of an
  input channel). Use a forecasting library.
- We do NOT support multi-label classification in v1; that ships in
  v1.1.
- We do NOT support distributed / multi-GPU training. Single-GPU is
  the supported envelope.

The roadmap (`docs/requirements.md` "Roadmap") lays out what comes
next: PatchTST / TimesNet / TST (v2 transformer family), LSTM / GRU /
LSTM-FCN (v3 recurrent family).
