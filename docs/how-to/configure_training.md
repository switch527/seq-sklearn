# Configure training

seq-sklearn's training config is a pydantic v2 hierarchy. The full
field tables, types, defaults, and validators are the
[config reference](../reference/config). This page is the practical
"which knobs should I know about" tour for someone who has just done
the [tutorial](../tutorial/first_classifier).

## The shape of a TFT config

A `TFTClassifier` (or `TFTRegressor`) takes:

- a `tabular_config: TabularConfigParams` declaring the panel
  schema (see [the panel data contract](panel_data));
- architecture hyperparameters (`hidden_size`, `attention_heads`,
  `dropout`, etc.);
- training hyperparameters (`max_epochs`, `batch_size`,
  `val_fraction`, `cal_fraction`, `learning_rate`);
- precision/determinism settings (`precision`, `seed`);
- a `scheduler: SchedulerParams` block.

The classes are pydantic models with `extra="forbid"`: typos like
`hidden_dim=128` fail loudly at construction, not silently at
training time.

```python
from seq_sklearn import TFTClassifier, TFTConfig
from seq_sklearn.config._adapters import SchedulerParams, TabularConfigParams

clf = TFTClassifier(
    task_type="binary",
    tabular_config=TabularConfigParams(
        id_col="customer_id",
        time_col="month",
        time_varying_real_cols=("spend", "logins"),
        static_real_cols=("tenure_months",),
        static_categorical_cols=("segment",),
        time_varying_categorical_cols=("channel",),
        lookback=12,
        min_periods=3,
        min_periods_predict=1,
        max_categorical_cardinality=10_000,
    ),
    scheduler=SchedulerParams(name="cosine_with_warmup", warmup_steps=50),
    hidden_size=128,
    attention_heads=4,
    max_epochs=60,
    batch_size=64,
    val_fraction=0.2,
    cal_fraction=0.0,
    precision="32-true",
    seed=42,
)
```

## What each block does

- **`task_type`** — `"binary"`, `"multiclass"`, `"regression_point"`,
  or `"regression_quantile"`. Picks the head + loss. Wrong task type
  vs `y` raises at fit.
- **`tabular_config`** — the schema (see panel data page).
- **`hidden_size`** — width of the gated residual networks +
  attention. Defaults to 64; 128-256 is typical. Doubling roughly
  4x's the parameter count.
- **`attention_heads`** — number of multi-head-attention heads.
  Must divide `hidden_size`. Defaults to 4.
- **`dropout`** — applied inside the gated residual blocks. 0.1 is
  the default; 0.0 disables. Raise if you observe over-fitting in
  the validation loss.
- **`max_epochs`** — hard cap on training epochs. The internal
  trainer also early-stops on the validation loss; `max_epochs` is
  the ceiling.
- **`batch_size`** — windows per gradient step. Larger is faster
  per-epoch and more memory; 64-256 is typical for the proxy panel
  size.
- **`val_fraction`** — share of train rows held out for early
  stopping. 0.2 is the default. The split is *time-ordered*, not
  random, to avoid the F2 leak.
- **`cal_fraction`** — share for probability calibration. 0.0
  disables calibration (the cal fold folds back into train).
- **`learning_rate`** — Adam LR. Default 1e-3; 5e-4 helps with
  very small panels.
- **`scheduler`** — LR schedule. `cosine_with_warmup` is the
  recommended default (better convergence in a bounded epoch
  budget); `constant` for short experiments.
- **`precision`** — `"32-true"` (default, deterministic),
  `"16-mixed"` (faster on Ampere+ GPUs, slightly less reproducible).
- **`seed`** — full library determinism mode at this seed. Required
  for the N1 quickstart-in-CI reproducibility contract.

## What NOT to change first

- The architecture options below `hidden_size` (number of LSTM
  layers, GRN-block count) — they exist for the v3 family extension
  and rarely move the needle for a single TFT.
- The `tabular_config` internals (`max_categorical_cardinality`,
  scaling strategies) — set them once per dataset and forget.

Start by tuning `hidden_size`, `max_epochs`, `batch_size`, and
`dropout`; see [Tuning](tuning) and [Tune with Optuna](tune_with_optuna).

```{testcode}
from seq_sklearn import TFTClassifier
from seq_sklearn.config._adapters import SchedulerParams

sched = SchedulerParams(name="cosine_with_warmup", warmup_steps=50)
assert sched.name == "cosine_with_warmup"
assert TFTClassifier is not None
```
