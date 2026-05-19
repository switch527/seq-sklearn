# The panel data contract

This is the most important page to read before training. The panel
shape and the way seq-sklearn treats time are the single biggest
source of user confusion, especially for users coming from
forecasting libraries.

## The shape

A *panel* is a tidy DataFrame with one row per `(entity_id, period)`
pair. Columns split into four kinds, declared in
`TabularToSequenceConfig`:

- **`id_col`** — the entity identifier. A string, integer, or
  category. Identity-only; the model never learns "what an entity
  is", it sees one ordered sequence per entity.
- **`time_col`** — the period index. Any column whose values can be
  sorted to produce the within-entity ordering. The library does NOT
  interpret these values as wall-clock; see "Periods are not wall
  clock" below.
- **`static_categorical_cols` / `static_real_cols`** — columns whose
  value is constant over the entity's lifespan (segment, plan,
  region, account age at observation start). One value per entity,
  repeated in every row.
- **`time_varying_real_cols` / `time_varying_categorical_cols`** —
  columns whose value varies row to row within an entity.

The label `y` is one value per row (same length as the panel); the
estimator picks per-window labels internally based on `lookback`.

## Periods are not wall clock

**Consecutive rows are consecutive periods, regardless of the value
of `time_col`.** If you have monthly observations and one entity is
missing March, you have a one-period gap, which the model will treat
as "the next observed period is one step after the previous". The
library does not look at calendar arithmetic.

This matters because forecasting libraries usually densify, fill, or
resample by wall-clock; seq-sklearn does not. The choice is
deliberate: a panel of mixed cadences (some entities monthly, some
weekly) trains cleanly when periods are ordinal within each entity.

If you need calendar semantics (e.g. a hospital admission "30 days
later"), pre-process to a fixed grid before passing the panel in.
The library happily consumes the post-densified panel.

## Variable-length entities

Different entities can have different sequence lengths. The library
pads to `lookback` and masks the padding internally:

```python
config = TabularToSequenceConfig(
    id_col="customer_id",
    time_col="month",
    time_varying_real_cols=("spend", "logins"),
    time_varying_categorical_cols=("channel",),
    static_real_cols=("tenure_months",),
    static_categorical_cols=("segment",),
    lookback=12,         # the model sees the most-recent 12 periods
    min_periods=3,       # entities with < 3 observed periods are dropped
    min_periods_predict=1,  # at predict time, accept even 1-period entities
    max_categorical_cardinality=10_000,
)
```

`min_periods` is a fit-time filter. `min_periods_predict` is the
predict-time floor, deliberately laxer because at scoring time you
often have new entities with very few periods.

## Worked example with printed shapes

```python
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator

gen = SyntheticPanelGenerator(
    target_kind="binary",
    num_entities=4,           # tiny for readability
    periods_per_entity=(2, 5), # variable history per entity
    lookback=4,
    seed=0,
)
panel, y = gen.generate(seed=0)
print(panel.shape, panel.columns.tolist())
print(panel.head(8))
print("y first 8:", y[:8])
```

You will see something like:

```text
(14, 7) ['entity_id', 'period', 'static_cat_0', 'static_real_0',
         'tv_real_0', 'tv_real_1', 'tv_cat_0']
   entity_id  period static_cat_0  ...  tv_real_0  tv_real_1  tv_cat_0
0          0       0           A   ...      0.12      -0.45         x
1          0       1           A   ...      0.34       0.10         y
...
y first 8: [0 0 0 1 1 1 0 0]
```

The shape `(14, 7)` reflects the variable-length entities (some have
2 periods, some 5, summing to 14 rows).

## Common mistakes

- **Treating `time_col` as a forecast horizon.** It is not. The
  library is a *classifier* on the most-recent `lookback` periods,
  not a forecaster of the next period.
- **Mixing static and time-varying.** A column that's constant within
  entity but varies across entities is *static*. A column that
  varies row to row within entity is *time-varying*. Splitting
  wrong silently changes what the model sees.
- **Random-shuffling a multi-entity panel for cross-validation.** A
  shuffled split leaks future periods of one entity into the train
  set; use the [entity-time split](time_series_splitting.md)
  instead.

```{testcode}
from seq_sklearn.config._adapters import TabularConfigParams

cfg = TabularConfigParams(
    id_col="customer_id",
    time_col="month",
    time_varying_real_cols=("spend",),
    static_real_cols=(),
    static_categorical_cols=(),
    time_varying_categorical_cols=(),
    lookback=12,
    min_periods=3,
    min_periods_predict=1,
    max_categorical_cardinality=10,
)
assert cfg.lookback == 12
```
