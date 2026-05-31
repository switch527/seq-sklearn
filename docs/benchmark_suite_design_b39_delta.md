# B39 delta: TSC categorical one-hot handling (D-B12.6)

## Requirements

R-B39-1 (closes D-B12.6): `panel_to_tensor` SHALL one-hot
encode `spec.feature_categorical_cols` when a non-None
`categorical_categories` mapping is supplied. Each
categorical column contributes one channel per reference
category. Unseen category values (not in the reference) map
to an all-zero one-hot vector for that column. Total
channels = `len(feature_real_cols) + sum(len(reference[c])
for c in feature_categorical_cols)`.

R-B39-2: A new helper
`compute_categorical_categories(panel, spec)` returns the
per-column reference mapping from a panel's observed
categories. Each column's categories are sorted ascending
for deterministic channel ordering across processes.

R-B39-3: When `categorical_categories=None` (default), the
v1 drop-categoricals behavior is preserved. Existing
`test_raw_mts_drops_categorical_channels` and
`test_raw_mts_all_categorical_panel_raises` must keep
passing without modification.

R-B39-4: The `_TSCAdapter` family wires the helper into its
fit-time reshape and stores the resulting reference on the
adapter instance; predict-time reshape passes the stored
reference so train-set categories pin the channel layout.

## Non-requirements

- v1 does NOT add ordinal or target encoding. The original
  D-B12.6 deferral explicitly rejected these on metric-space
  grounds (ROCKET's kernel distance assumes float channels).
- v1 does NOT use aeon's `ColumnEnsembleClassifier` or any
  aeon helper. The protocol layer stays aeon-free.
- v1 does NOT change the all-categorical-panel error
  semantics: a spec with empty `feature_real_cols` still
  raises `RawMTSError("zero real-valued ...")`. The one-hot
  channels are additive on top of the real-channel base;
  zero real-channels remains an unsupported configuration.

## B39.0 Background

D-B12.6 (in `docs/benchmark_suite_design_b12_delta.md:97`)
deferred categorical handling: ordinal encoding violates
ROCKET's metric-space assumption (`x_cat=1` and `x_cat=2`
are not "distance 1 apart" in any meaningful sense), and
unbounded one-hot inflates the channel count
unpredictably. The "predictable channel count" objection is
resolved by pinning categories at fit time and reusing them
at predict time. Unseen predict-time categories map to
all-zero so the channel count stays fixed across fit/predict.

The TSC adapter test suite is aeon-gated and currently
skips when aeon is unavailable; the protocol-layer
`panel_to_tensor` tests cover the encoding semantics
directly without invoking the adapter.

## B39.1 Helper signature

`benchmarks/protocol/raw_mts.py`:

```python
def compute_categorical_categories(
    panel: pd.DataFrame, spec: DatasetSpec
) -> dict[str, tuple[Any, ...]]:
    """Reference categories per categorical column, sorted
    ascending for deterministic channel ordering.

    NaN values are dropped (a NaN categorical at predict time
    will map to all-zero one-hot, same as an unseen value).
    """
```

`panel_to_tensor` adds a keyword-only parameter:

```python
def panel_to_tensor(
    panel: pd.DataFrame,
    spec: DatasetSpec,
    *,
    override_lookback: int | None = None,
    categorical_categories: Mapping[str, Sequence[Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
```

When `categorical_categories=None`: current drop-categoricals
behavior is unchanged.

When `categorical_categories` is supplied: every column in
`spec.feature_categorical_cols` MUST appear as a key. Missing
keys raise `RawMTSError` (defensive — silently dropping a
declared column would inflate or shrink the channel count
unpredictably).

## B39.2 Algorithm

```
1. Validate categorical_categories (if not None) covers every
   spec.feature_categorical_cols entry.
2. Real-channel branch unchanged.
3. For each kept entity's trailing-L window `kept`:
   a. real_block = kept[real_cols].to_numpy(float32).T  -> (R, L)
   b. If categorical_categories is None:
        block = real_block
      else:
        cat_blocks = []
        for cat_col in spec.feature_categorical_cols:
            ref = categorical_categories[cat_col]
            col_values = kept[cat_col].to_numpy()           # (L,)
            # (C, L) one-hot: 1 where col_values[t] == ref[c]
            one_hot = (col_values[None, :] == np.asarray(ref)[:, None]).astype(np.float32)
            cat_blocks.append(one_hot)
        block = np.concatenate([real_block, *cat_blocks], axis=0)
   c. Append block to instance_blocks.
4. np.stack as before.
```

Per-row comparison handles arbitrary categorical dtypes
(string, int, pd.Categorical) uniformly: `==` works for
numpy/pandas Series of any dtype.

## B39.3 Adapter wiring

`benchmarks/adapters/tsc.py` `_TSCAdapter`:

- Add field `_fit_categorical_categories: dict | None = None`
- In `fit`: compute via `compute_categorical_categories(panel, spec)`,
  store on `self`, pass through `_reshape` to `panel_to_tensor`.
- In `_reshape`: forward the stored reference.
- Invalidate the panel-id reshape cache so a re-fit on a new
  panel doesn't return a stale tensor computed against the
  previous fit's category reference.

## B39.4 Tests

Baseline (post-B38): 1112.

### B39.4.1 Helper

1. `test_compute_categorical_categories_returns_sorted_unique_per_col`:
   panel with cat col 'c' values
   `['b', 'a', 'b', 'a', 'c']` → reference `('a', 'b', 'c')`.
2. `test_compute_categorical_categories_drops_nan_values`:
   panel with NaN in 'c' → NaN absent from the reference
   tuple.
3. `test_compute_categorical_categories_empty_when_no_cat_cols`:
   spec with empty `feature_categorical_cols` → empty dict.

### B39.4.2 panel_to_tensor one-hot path

4. `test_raw_mts_one_hot_encodes_categoricals`: 1 real col +
   1 cat col with 2 categories → tensor channel dim == 3.
   Per-instance per-timestep cat channels match the one-hot
   of that row's cat value.
5. `test_raw_mts_unseen_category_maps_to_all_zero`: reference
   excludes a value present in the panel → that row's cat
   channels are all zero.
6. `test_raw_mts_multiple_categorical_columns`: 2 cat cols
   with sizes 2 and 3 → channel dim == real + 5.
7. `test_raw_mts_categorical_categories_none_preserves_drop`:
   default behavior unchanged (channels == real-only).
8. `test_raw_mts_missing_categorical_reference_raises`:
   passing a categorical_categories dict missing a declared
   cat_col raises `RawMTSError`.

### B39.4.3 Expected test delta

Baseline: 1112.
- 8 new tests.
- Existing tests (including the 2 D-B12.6 drop-categoricals
  pins) unchanged.
- Total: 1112 + 8 = 1120.

Adapter wiring is exercised via the existing aeon-gated TSC
adapter tests (currently skipped without aeon installed); no
new tests added there since the install path is environment-
specific.

## B39.5 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B39-Risk-1 | Predict-time panel introduces a category absent from the fit-time reference → all-zero one-hot silently degrades the signal for those rows. | Low | Documented behavior. The alternative (raising) would crash a run on benign category drift; the all-zero convention is the same one used for unseen categories in sklearn's OneHotEncoder(handle_unknown="ignore"). |
| R-B39-Risk-2 | Train-only categories absent from a fold's test set still consume channels → zero-variance channel for that fold. | Low | Aeon's ROCKET and MultiRocket tolerate zero-variance channels (the kernel response is constant and contributes zero discrimination). Catch22 features include count-style features that handle constant channels. |
| R-B39-Risk-3 | Non-hashable categorical dtype (e.g., a numpy array as a value) breaks `np.unique` in the helper. | Low | The DatasetSpec contract requires categorical columns to be hashable; this is a precondition violation upstream of B39. |

## Deferred

(None added.)
