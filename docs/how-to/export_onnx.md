# Export to ONNX (and reload in onnxruntime)

A fitted `TFTClassifier` or `TFTRegressor` can be exported to ONNX
for deployment in any onnxruntime-supporting environment (CPU, GPU,
mobile, edge). The exported graph is **backbone + head producing raw
logits**; calibration, threshold tuning, and below-floor NaN-fill
remain sklearn-side numpy post-processing in `predict_proba` /
`predict_quantiles`.

## Install

ONNX is an optional extra:

```bash
pip install "seq-sklearn[onnx]"
```

This installs `onnx`, `onnxruntime`, and `onnxscript`. The extra is
not required for training or for `predict` against a fitted model in
Python; it's only for `export_onnx` and consuming the exported file.

## Export

```python
from seq_sklearn import TFTClassifier

clf = TFTClassifier(...).fit(panel_train, y_train)

# Pass a representative panel slice; one window per row is emitted.
# A short / left-padded entity in the slice trains the mask path.
clf.export_onnx("model.onnx", panel_train.head(50))
```

`export_onnx(path, X)` requires the fitted estimator and an example
panel `X` (the schema-valid input used to trace the graph). The trace
captures:

- the variable-selection network,
- the static-context encoder,
- the gated recurrent backbone (an `nn.LSTM` in the pack-free,
  gather-preserving export path),
- multi-head attention (forced through the math backend, no
  ScaledDotProductAttention fused op),
- the prediction head.

Dynamic batch is supported; the time dimension is fixed at the
fit-time `lookback`.

## Reload in onnxruntime

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

# Build the model's batch dict from your panel via the fitted transformer.
batch = clf.transformer_.transform(panel_test)
args = (
    batch["static_categorical"].numpy(),
    batch["static_real"].numpy(),
    batch["time_varying_real"].numpy(),
    batch["time_varying_categorical"].numpy(),
    batch["padding_mask"].numpy(),
)
names = [i.name for i in session.get_inputs()]
feed = dict(zip(names, args, strict=True))
logits = session.run(None, feed)[0]

# Sigmoid + calibration are sklearn-side, not in the graph.
proba = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
```

## What's NOT in the graph

Phase 10's design (the post-Gemini consensus) deliberately keeps
non-tensor operations out of the ONNX graph:

- **Calibration** (temperature / isotonic / conformal). Apply
  separately in the serving layer using the fitted calibrator.
- **Threshold tuning.** Pick the threshold in serving.
- **Below-floor NaN-fill.** Rows whose history is shorter than
  `min_periods_predict` get NaN probabilities; the graph does not
  emit them.
- **Caller-row-order restore** (F1). Sklearn-side post-processing.

If you need parity with `predict_proba`, the round-trip is documented
by the parity test (`tests/integration/test_onnx_parity.py`).

## Op-surface guarantee

The exported graph emits only the restricted op surface enumerated in
architecture A21 (a frozen allowlist verified by
`tests/deploy/test_restricted_op_surface.py`). No `Loop`, no `If`, no
`ScaledDotProductAttention`. This is what makes the graph portable.

## Common pitfalls

- **Passing the wrong X.** `X` must be a schema-valid panel matching
  the fitted `tabular_config`. Pass a slice with at least one
  short / left-padded entity so the mask path is traced.
- **Forgetting the post-processing.** The graph emits raw logits.
  Sigmoid + calibration + below-floor NaN are sklearn-side.
- **Trying to export the calibrator.** Calibration is intentionally
  outside; replicate it in your serving language.

```{testcode}
from seq_sklearn import TFTClassifier, TFTRegressor

assert hasattr(TFTClassifier, "export_onnx")
assert hasattr(TFTRegressor, "export_onnx")
```
