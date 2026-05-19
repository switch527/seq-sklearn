# Tuning: the five fields that matter

The TFT config has many fields; for tuning, five of them dominate.
This page is the goal-framed guide for hand-tuning. For automated
search, see [Tune with Optuna](tune_with_optuna).

The rule: change one knob at a time, observe validation loss and
ROC-AUC together, and keep `seed` and `precision="32-true"` fixed
so you're measuring the knob, not run-to-run variance.

## To get better accuracy

1. **`hidden_size`** (default `64`). Doubling from `64` → `128` →
   `256` is the cheapest dial. Returns flatten quickly above `256`
   on most panels. **Trade-off:** model size grows ~quadratically;
   peak memory and step time grow with it.
2. **`max_epochs`** (default `60`). If validation loss is still
   falling at `max_epochs`, raise it. Internal early stopping will
   cut the run when the validation loss plateaus, so a high cap is
   safe. **Trade-off:** wall-clock time scales linearly.
3. **`dropout`** (default `0.1`). For very small panels (under ~500
   entities) or noisy targets, try `0.2`–`0.3`. For wide panels with
   strong signal, `0.05` or `0.0` can recover a fraction of accuracy.
   **Trade-off:** high dropout slows convergence; pair with more
   epochs.
4. **`scheduler`** (default
   `SchedulerParams(name="cosine_with_warmup", warmup_steps=50)`).
   Cosine with warmup converges noticeably better than `constant` in
   a bounded epoch budget. **Trade-off:** none, this is the recommended
   default; the option exists for sweep comparisons.
5. **`attention_heads`** (default `4`). Going to `8` helps modestly
   on panels with rich time-varying inputs; not much else. Must
   divide `hidden_size`. **Trade-off:** marginal step-time increase.

## To go faster

- **Lower `max_epochs`.** The single biggest lever for wall-clock.
- **Raise `batch_size`** until GPU memory caps out (typically 128 or
  256). Each batch is more parallel work per gradient step.
- **`precision="16-mixed"`** on Ampere or later GPUs gives a 1.5x-2x
  speedup. **Trade-off:** small loss of determinism / bit-exactness
  vs. `"32-true"`.
- **Lower `hidden_size`** if you can afford the accuracy.

## To reduce overfitting

- **Raise `dropout`** to `0.2` or `0.3`.
- **Lower `hidden_size`** (reduces capacity).
- **Add early stopping headroom** with a slightly larger
  `val_fraction` (e.g. `0.25`).
- **Use `cal_fraction>0`** if probabilities are your actual output;
  calibration corrects systematic miscalibration without retraining.
- **Use balanced sampling** on imbalanced panels (see
  [Handle imbalanced classes](imbalanced_classes)).

## What NOT to tune first

- The `tabular_config` internals (`max_categorical_cardinality`,
  scaler choice). Set once per dataset.
- The architecture-block knobs deeper than `hidden_size` and
  `attention_heads`. They exist for v3 cross-family experiments and
  rarely move the v1 needle.
- `learning_rate` (default `1e-3`). With cosine-warmup, the default
  rarely needs adjustment within an order of magnitude.

## When to stop

A sensible budget for first-pass tuning: 20-30 trials of an
Optuna search over `{hidden_size, max_epochs, dropout, batch_size}`.
After that, returns diminish quickly; you have likely found the
right architecture-class for your panel and further gain comes from
data, not hyperparameters.
