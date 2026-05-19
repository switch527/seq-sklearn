# Handle imbalanced classes

For binary classification with skewed prior (5% positives, 1%, the
typical churn/fraud/default setting), three things matter, in order
of impact:

1. **Use proper scoring rules at evaluation time.** Accuracy is
   misleading; report log-loss, ROC-AUC, PR-AUC, or Brier instead.
2. **Sample with awareness of the prior.** seq-sklearn's
   `SamplerConfig` supports class-balanced batches without resampling
   the whole panel.
3. **Calibrate the probabilities.** A raw model that's
   discrimination-good can still be miscalibrated; pass
   `cal_fraction > 0` to use temperature scaling.

## Class-balanced sampling

`SamplerConfig` is part of `TFTClassifier`'s training-side config.
The simplest balanced setting:

```python
from seq_sklearn import TFTClassifier
from seq_sklearn.config.sampler import SamplerConfig

clf = TFTClassifier(
    task_type="binary",
    tabular_config=...,             # the schema
    sampler=SamplerConfig(strategy="balanced_by_class"),
    cal_fraction=0.1,               # 10% held out for calibration
    max_epochs=80,                  # more epochs for tail classes
)
```

The sampler draws each batch with class-balanced expectation while
keeping the *underlying* distribution intact for validation/calibration
splits, so the held-out metrics are still calibrated against the real
prior.

## Threshold tuning vs. probability calibration

Two separate concerns:

- **Threshold tuning** picks the decision boundary that maximizes a
  target operating-point metric (recall at a precision floor, F1,
  Youden's J). seq-sklearn's `predict_proba` returns calibrated
  probabilities; you choose the threshold yourself by sweeping or by
  hitting a precision/recall target on a held-out set.
- **Probability calibration** corrects systematic miscalibration
  (e.g. a model that emits 0.4 for samples with true prior 0.1). With
  `cal_fraction > 0`, temperature scaling is fitted on the
  calibration fold.

Calibration is independent of threshold; for imbalanced problems you
typically want both.

## What about up-sampling / SMOTE?

seq-sklearn does not ship a per-row up-sampler because rows in a
panel are not independent samples; duplicating a row changes the
sequence, not the marginal. Class-balanced batch sampling is the
panel-correct analog. If you have very few positive entities (not
just positive rows), the right move is to weight the loss; future
versions may ship a `pos_weight` config field.

## Quick checklist

- [ ] Use ROC-AUC / PR-AUC / log-loss to evaluate, not accuracy.
- [ ] Set `sampler=SamplerConfig(strategy="balanced_by_class")`.
- [ ] Set `cal_fraction=0.1` or so to enable temperature calibration.
- [ ] Tune the decision threshold post-hoc on a held-out set.

```{testcode}
from seq_sklearn.config.adapters import SamplerParams

sampler = SamplerParams(strategy="balanced_by_class")
assert sampler.strategy == "balanced_by_class"
```
