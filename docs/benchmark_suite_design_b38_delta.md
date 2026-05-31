# B38 delta: sufficient-stats bootstrap optimization (D-B13.7)

## Requirements

R-B38-1 (closes D-B13.7): add a new fast-path bootstrap
primitive `entity_block_bootstrap_ci_mean_fast` in
`benchmarks/metrics/bootstrap.py` that uses per-entity
sufficient statistics `(sum_loss, count_rows)` instead of
concatenating resampled-entity rows per resample.
Correctness-equivalent (modulo float drift) to the naive
path when the metric is `np.nanmean` (or
`sqrt(np.nanmean(x))` for RMSE).

R-B38-2 (R-B38-1 scope): supports both `ci_method="percentile"`
and `ci_method="bca"`. The BCa path uses leave-one-entity-
out sufficient-stats jackknife instead of re-applying
metric_fn to concatenated leave-out rows.

## Non-requirements

- v1 does NOT wire the fast path into the 5 aggregators.
  The deferral explicitly framed this as "the optimization
  is a B13-followup once a full-tier dataset surfaces the
  R-B13-3 ceiling". Aggregators can opt in via a future
  per-experiment flag if/when a large dataset surfaces the
  ceiling.
- v1 does NOT remove or modify `entity_block_bootstrap_ci`
  (the naive path). The fast path is additive.
- v1 does NOT support arbitrary `metric_fn`. The fast path
  is hardwired to `mean` semantics with an opt-in
  `sqrt_mean` switch for the RMSE case (matches the v1
  metric_fns named in the D-B13.7 deferral body).

## B38.0 Background

`entity_block_bootstrap_ci` at
`benchmarks/metrics/bootstrap.py:153-310` concatenates the
rows of resampled entities per resample (O(N) memory
traffic per resample). For the v1 metric_fns (`np.nanmean`
classification, `lambda x: float(np.sqrt(np.nanmean(x)))`
regression), the metric reduces to
`sum(losses) / count(non_nan_losses)` (or sqrt of that).
Per-entity (sum, count) sufficient statistics let each
resample compute the bootstrap statistic in O(E) time
without materializing the concatenated row vector. For the
full-tier Amex dataset (~500k entities vs 6M rows) this is
a ~12x memory + 12x throughput improvement.

The BCa jackknife similarly benefits: leave-one-entity-out
becomes `(total_sum - entity_sums[i]) / (total_count -
entity_counts[i])`, O(1) per leave-one-out vs O(N) for the
naive path.

## B38.1 Fast-path helper signature

In `benchmarks/metrics/bootstrap.py`:

```python
def entity_block_bootstrap_ci_mean_fast(
    losses: np.ndarray,
    entity_ids: np.ndarray,
    *,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    confidence: float = _DEFAULT_CONFIDENCE,
    seed: int = _DEFAULT_SEED,
    ci_method: Literal["percentile", "bca"] = "bca",
    sqrt_mean: bool = False,
) -> BootstrapResult:
    """Sufficient-statistics fast path for the
    entity-block bootstrap, hardwired to `nanmean` semantics
    (or `sqrt(nanmean(x))` when `sqrt_mean=True`).

    Numerically equivalent to `entity_block_bootstrap_ci(...
    metric_fn=np.nanmean)` (or the sqrt variant) modulo
    float-order drift. The naive path takes O(N) memory
    traffic per resample (row-vector concatenation); this
    fast path uses pre-computed per-entity
    (sum_loss, count_non_nan) statistics, reducing memory
    traffic to O(E) per resample.

    Use cases:
    - v1 classification aggregators with metric_fn=np.nanmean.
    - v1 regression aggregators with sqrt_mean=True.
    A custom metric_fn that is NOT expressible from
    sum/count sufficient statistics (e.g., median, ROC-AUC)
    must continue to use `entity_block_bootstrap_ci`.
    """
```

## B38.2 Algorithm

```
1. Validate shapes + confidence + n_resamples (same as naive).
2. Compute per-row NaN mask + safe-fill.
3. Group by entity:
   - entity_sums[i] = sum of safe-filled losses for entity i
   - entity_counts[i] = count of non-NaN losses for entity i
4. total_sum = entity_sums.sum(); total_count = entity_counts.sum().
5. If total_count == 0: raise ValueError (all-NaN).
6. ground_truth_mean = total_sum / total_count (sqrt if sqrt_mean).
7. If n_entities <= 1: return degenerate triple (same as naive).
8. RNG = PCG64(seed). For each resample r in [0, n_resamples):
   - picked = rng.integers(0, n_entities, size=n_entities)
   - sum_picked = entity_sums[picked].sum()
   - count_picked = entity_counts[picked].sum()
   - resampled[r] = NaN if count_picked == 0 else sum_picked / count_picked
   - apply sqrt if sqrt_mean
9. alpha = (1 - confidence) / 2.
10. If ci_method == "percentile":
    ci_lo, ci_hi from np.percentile(resampled).
    return BootstrapResult(..., fallback_reason=None).
11. ci_method == "bca":
    - p0 = mean(resampled <= ground_truth_mean).
    - jackknife[i] = (total_sum - entity_sums[i]) / (total_count - entity_counts[i])
      (sqrt if sqrt_mean).
    - a = _compute_acceleration_from_jackknife(jackknife).
    - alpha_1, alpha_2, fallback = _bca_percentile_points(p0, a, confidence).
    - ci_lo, ci_hi from np.percentile.
    return BootstrapResult(..., fallback_reason=fallback).
```

## B38.3 Tests

Baseline (post-B37): 1103.

### B38.3.1 Equivalence tests

1. `test_fast_path_matches_naive_for_percentile_mean`: same
   fixture; assert `fast.mean == approx(naive.mean)`,
   `fast.ci_lo == approx(naive.ci_lo)`,
   `fast.ci_hi == approx(naive.ci_hi)`, both fallbacks
   None. Uses identical seed; outputs match within float
   tolerance.
2. `test_fast_path_matches_naive_for_bca_mean`: same shape,
   `ci_method="bca"`.
3. `test_fast_path_matches_naive_for_percentile_sqrt_mean`:
   `sqrt_mean=True` on fast; matching `lambda x:
   float(np.sqrt(np.nanmean(x)))` on naive.
4. `test_fast_path_matches_naive_for_bca_sqrt_mean`: same
   with BCa.

### B38.3.2 NaN-handling

5. `test_fast_path_handles_partial_nan_rows`: input with
   NaN rows; ground-truth-mean ignores them; resample loop
   correctly handles `count_picked == 0` edge case.
6. `test_fast_path_raises_on_all_nan_input`: all losses
   NaN; assert ValueError.

### B38.3.3 Degenerate single-entity

7. `test_fast_path_single_entity_returns_degenerate_ci`:
   one entity; assert ci_lo == ci_hi == mean.

### B38.3.4 Determinism + seed

8. `test_fast_path_deterministic_at_fixed_seed`: two calls
   with same seed yield identical BootstrapResult.
9. `test_fast_path_different_seeds_diverge`: two calls
   with different seeds yield different bounds.

### B38.3.5 Expected test delta

Baseline: 1103.
- 9 new tests.
- Existing tests unchanged.
- Total: 1103 + 9 = 1112.

## B38.4 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B38-Risk-1 | Float-order drift between sum-of-sums (fast) and sum-of-rows (naive) breaks equivalence. | Low | `pytest.approx` with `abs=1e-9` for the mean and percentile bounds (ULP-level drift). The BCa bounds carry a wider tolerance (`abs=1e-2`) because the bias-correction `p0 = mean(resampled <= ground_truth)` discretely depends on a `<=` comparison: when a resample sits within float drift of the ground-truth mean, the two paths can disagree by one sample on which side it falls, shifting `p0` by `1/n_resamples`. That shift propagates nonlinearly through `norm.ppf` and the BCa percentile transform and can move the bounds by tenths of a percent of the CI width — orders of magnitude larger than the input drift but still semantically equivalent. Equivalence is asserted strictly on the mean and on the percentile-method bounds; BCa-bound assertions use the relaxed tolerance with this footnote. |
| R-B38-Risk-2 | The fast path's per-resample `entity_sums[picked].sum()` is still O(E); is it actually faster? | Low | E << N for realistic datasets (Amex: 500k entities vs 6M rows). The fast path also avoids per-resample memory allocation of concatenated row vectors. |
| R-B38-Risk-3 | BCa jackknife sufficient-stats path requires `total_count - entity_counts[i] > 0` for every i; degenerate cases need handling. | Low | When n_entities >= 2 and total_count >= 2, every leave-one-out has at least one non-NaN row. The function early-returns for n_entities <= 1, so this is structurally satisfied. |

## Deferred

- **D-B38.1**: wire `entity_block_bootstrap_ci_mean_fast`
  into the 5 aggregators behind a per-experiment flag.
  Currently each aggregator calls
  `entity_block_bootstrap_ci` with `np.nanmean` (or
  sqrt-nanmean for regression). Adding a fast-path opt-in
  flag would let users trade CI computation time for the
  same numerical output. Deferred until a consumer surfaces
  the performance ceiling.
