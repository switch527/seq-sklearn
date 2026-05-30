# B21 design delta: BCa CI on the entity-block bootstrap (D-B16.2 / D-B13.5)

**Scope**: D-B16.2 (inherited from D-B13.5) replaces the
percentile bootstrap CI with the bias-corrected and accelerated
(BCa) bootstrap CI on the shared `entity_block_bootstrap_ci`
primitive. All 5 rollup aggregators (B5 raw-loss, B6 pairwise,
B7 training-time, B8 HPO-uplift, B16 ensemble-lift) get an
audit field `bootstrap_ci_method` recording which method was
used. v1 default flips to `"bca"`; the `"percentile"` path is
retained as an opt-in for parity testing and as a fallback when
the BCa degenerate paths fire.

The BCa improvement matters for skewed bootstrap distributions
(e.g., RMSE on small-N regression cells, Δloss with strong
asymmetry between baseline and seq families). Percentile CIs
under-cover on skewed distributions; BCa adjusts both endpoints
via a bias-correction `z0` and an acceleration `a` derived from
a leave-one-entity-out jackknife. The math is standard
(Efron 1987); the implementation is `O(n_entities)` extra
metric_fn calls per dataset, which is fast at typical
n_entities ≤ 100.

## Requirements

The grading rubric for every reviewer finding traces back to
one of these.

- **R-B21-1** Add a `ci_method: Literal["percentile", "bca"]`
  keyword parameter to `entity_block_bootstrap_ci` (default
  `"bca"`). The percentile path is unchanged; the BCa path
  computes:
  - **Bias correction `z0`**: `z0 = norm.ppf(p_0)` where
    `p_0 = mean(resampled <= ground_truth)` is the proportion
    of resampled statistics at or below the unresampled
    ground-truth statistic.
  - **Acceleration `a`**: jackknife over UNIQUE ENTITIES.
    For each unique entity `i`, compute `theta_minus_i =
    metric_fn(losses_view[all_rows_except_entity_i])`. Then
    `a = sum((m_dot - theta_minus_i)^3) / (6 * (sum(
    (m_dot - theta_minus_i)^2))^1.5)` where `m_dot =
    mean(theta_minus_i)`.
  - **BCa percentiles**: with `z_lo = norm.ppf(alpha)`,
    `z_hi = norm.ppf(1 - alpha)`, `alpha = (1 - confidence)
    / 2`:
    - `alpha_1 = norm.cdf(z0 + (z0 + z_lo) / (1 - a*(z0 +
      z_lo)))`
    - `alpha_2 = norm.cdf(z0 + (z0 + z_hi) / (1 - a*(z0 +
      z_hi)))`
    - Return `(ground_truth, np.percentile(resampled,
      100*alpha_1, method="linear"), np.percentile(resampled,
      100*alpha_2, method="linear"))`.
- **R-B21-2** Degenerate fallback to percentile. The BCa
  formula has three failure modes:
  - `p_0 == 0.0` or `p_0 == 1.0` (ground truth at the edge
    of the resampled distribution): `norm.ppf` returns
    `-inf` / `+inf`. Fall back to the percentile CI for this
    dataset.
  - Jackknife denominator `sum((m_dot - theta_minus_i)^2)
    == 0` (all jackknife values equal): no acceleration is
    computable. Set `a = 0.0` (BCa reduces to BC, the
    bias-corrected percentile).
  - `1 - a*(z0 + z_*)` is `0` or negative for either
    endpoint (acceleration overshoots): fall back to the
    percentile CI for this dataset.
  In all three fallback cases the returned CI is the
  percentile interval; the aggregator's audit field
  `bootstrap_ci_method` records the FALLBACK reason via a
  new column `bootstrap_ci_fallback_reason: str | None`
  (one of `None`, `"p0_at_edge"`, `"a_overshoot"`).
- **R-B21-3** Add `bootstrap_ci_method: str = "percentile"`
  audit field to all 5 RollupRow schemas in
  `benchmarks/bootstrap_manifest.py` (B5 `RollupRow`, B6
  `PairwiseRollupRow`, B7 `TrainingTimeRollupRow`, B8
  `HPOUpliftRollupRow`, B16 `EnsembleLiftRollupRow`).
  Schema default `"percentile"` preserves old parquet shard
  semantics on load; aggregators pass `"bca"` explicitly to
  the row constructor. Sentinel rows pass
  `bootstrap_ci_method=BOOTSTRAP_DEFAULT_CI_METHOD` (the
  new module constant defaulting to `"bca"`).
- **R-B21-4** Add `bootstrap_ci_fallback_reason: str | None
  = None` audit field to all 5 RollupRow schemas. Non-None
  values indicate a per-dataset BCa fallback to percentile
  (the row carries percentile bounds despite
  `bootstrap_ci_method="bca"`).
- **R-B21-5** All 5 aggregators pass `ci_method="bca"` to
  `entity_block_bootstrap_ci` AND propagate the new
  primitive return (mean, lo, hi, fallback_reason) into the
  4-field write (the 3 existing + 1 new fallback reason).
  The primitive's return signature widens from
  `tuple[float, float, float]` to `tuple[float, float,
  float, str | None]` for the BCa path; the percentile path
  returns `(mean, lo, hi, None)` for shape parity.
- **R-B21-6** Add a new module constant
  `BOOTSTRAP_DEFAULT_CI_METHOD: str = "bca"` in
  `benchmarks/report/_bootstrap_aggregate.py` alongside the
  existing seeds + ceiling. Aggregators read this constant
  at the call site (NOT captured at import time) so a future
  test can monkeypatch the default without touching every
  aggregator module.
- **R-B21-7** The B16 ensemble-lift aggregator's TWO
  bootstrap calls (main Δloss + oracle Δ) BOTH switch to
  BCa with the same `ci_method` parameter; the existing
  `seed=BOOTSTRAP_DEFAULT_SEED` vs `seed=BOOTSTRAP_DEFAULT_SEED
  ^ BOOTSTRAP_ORACLE_SEED_OFFSET` independence contract
  (R-B20-2a) is preserved. The two oracle and main
  fallback_reason values are surfaced independently on the
  rollup row via `bootstrap_ci_fallback_reason` (main) and
  a new `oracle_ci_fallback_reason: str | None = None`
  (oracle); this is the only schema asymmetry between the
  two BCa invocations.
- **R-B21-8** No change to the byte-pin fixtures' literal
  CI bound values: B17 byte-pin fixtures construct
  `EnsembleLiftRollupRow` directly with hardcoded `(0.20,
  0.15, 0.25)` and `(0.40, ..., 0.45)` bounds, NOT
  bootstrap-computed values. The new `bootstrap_ci_method`
  field gets the documentary default `"bca"` and
  `bootstrap_ci_fallback_reason=None` (oracle version
  similar). The byte-pin regex matches the CI cell shape,
  which is unchanged.

## B21.0 Why BCa, why now

Percentile CIs assume the bootstrap distribution is
symmetric around the unresampled statistic. When the
distribution is skewed (e.g., long-tail Δloss on small
cells, RMSE on regression cells where `sqrt(nanmean)` is
applied per resample), percentile CIs systematically
under-cover the true parameter: nominal 95% intervals
achieve ~85-90% coverage on moderately skewed
distributions (Efron & Tibshirani 1993, §14).

BCa fixes this by adjusting both endpoints:
- The bias correction `z0` shifts the entire interval to
  compensate for systematic bias between the bootstrap
  mean and the unresampled statistic.
- The acceleration `a` adjusts the interval ASYMMETRICALLY
  based on the third moment of the jackknife distribution,
  which captures the skewness.

For symmetric bootstrap distributions BCa and percentile
agree. The cost is `O(n_entities)` extra metric_fn calls
per dataset (the jackknife pass), which at typical
n_entities ≤ 100 is negligible compared to the
`O(n_resamples)` resample loop.

## B21.1 Primitive changes

Current signature
(`benchmarks/metrics/bootstrap.py:59-67`):

```python
def entity_block_bootstrap_ci(
    losses: np.ndarray,
    entity_ids: np.ndarray,
    *,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    confidence: float = _DEFAULT_CONFIDENCE,
    seed: int = _DEFAULT_SEED,
    metric_fn: Callable[[np.ndarray], float] = _default_metric_fn,
) -> tuple[float, float, float]:
```

New signature:

```python
def entity_block_bootstrap_ci(
    losses: np.ndarray,
    entity_ids: np.ndarray,
    *,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    confidence: float = _DEFAULT_CONFIDENCE,
    seed: int = _DEFAULT_SEED,
    metric_fn: Callable[[np.ndarray], float] = _default_metric_fn,
    ci_method: Literal["percentile", "bca"] = "bca",
) -> tuple[float, float, float, str | None]:
```

The return tuple gains a 4th element: the fallback reason
(None on the happy path; one of `"p0_at_edge"` or
`"a_overshoot"` when BCa falls back to percentile). All 5
aggregator call sites + ensemble_lift's two call sites
update to unpack 4 values; the new value is written to the
rollup row's `bootstrap_ci_fallback_reason` (or
`oracle_ci_fallback_reason` for the oracle bootstrap).

### B21.1.1 BCa math

After the existing resample loop produces `resampled:
np.ndarray` of shape `(n_resamples,)` and the unresampled
`ground_truth_mean: float`:

```python
def _bca_percentiles(
    resampled: np.ndarray,
    ground_truth: float,
    rows_by_entity: list[np.ndarray],
    losses_view: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    confidence: float,
) -> tuple[float, float, str | None]:
    """Return (alpha_1, alpha_2, fallback_reason). The caller
    feeds the returned percentile points to np.percentile."""
    # Bias correction
    p0 = float(np.mean(resampled <= ground_truth))
    if p0 <= 0.0 or p0 >= 1.0:
        alpha = (1.0 - confidence) / 2.0
        return alpha, 1.0 - alpha, "p0_at_edge"
    z0 = float(norm.ppf(p0))

    # Acceleration via leave-one-entity-out jackknife
    n_entities = len(rows_by_entity)
    all_row_indices = np.concatenate(rows_by_entity)
    jackknife = np.empty(n_entities, dtype=np.float64)
    for i in range(n_entities):
        # Rows EXCLUDING entity i
        leave_out_rows = np.concatenate(
            [rows_by_entity[j] for j in range(n_entities) if j != i]
        )
        jackknife[i] = float(metric_fn(losses_view[leave_out_rows]))
    m_dot = float(np.mean(jackknife))
    deviations = m_dot - jackknife
    num = float(np.sum(deviations ** 3))
    denom = 6.0 * float(np.sum(deviations ** 2)) ** 1.5
    a = 0.0 if denom == 0.0 else num / denom

    # BCa percentile points
    alpha = (1.0 - confidence) / 2.0
    z_lo = float(norm.ppf(alpha))
    z_hi = float(norm.ppf(1.0 - alpha))
    denom_lo = 1.0 - a * (z0 + z_lo)
    denom_hi = 1.0 - a * (z0 + z_hi)
    if denom_lo <= 0.0 or denom_hi <= 0.0:
        return alpha, 1.0 - alpha, "a_overshoot"
    alpha_1 = float(norm.cdf(z0 + (z0 + z_lo) / denom_lo))
    alpha_2 = float(norm.cdf(z0 + (z0 + z_hi) / denom_hi))
    return alpha_1, alpha_2, None
```

The all-rows-except-entity-i view is materialized via
`np.concatenate`; the inner loop is `O(n_entities)`
metric_fn calls on `(n_rows - rows_for_entity_i)` arrays.
For symmetric rosters (e.g., 5 seeds × 5 folds = 25 cells
where each cell is one entity) this is 25 extra metric_fn
calls, each on a `(25-1) = 24-row` array → milliseconds.

### B21.1.2 Edge cases

- `n_entities <= 1`: existing degenerate path returns
  `(ground_truth, ground_truth, ground_truth)`. Now widens
  to `(ground_truth, ground_truth, ground_truth, None)`.
  BCa has nothing to correct on a single entity.
- `p0 == 0.0` (every resampled stat > ground_truth) or
  `p0 == 1.0` (every resampled stat ≤ ground_truth): the
  bootstrap is heavily biased AGAINST the unresampled
  estimate. Returning the percentile CI is the standard
  fallback (Efron & Tibshirani §14.4); the BCa correction
  formula's `norm.ppf(p0)` returns infinity which propagates
  invalid `(alpha_1, alpha_2)` values.
- Acceleration overshoot (`1 - a*(z0 + z_*) <= 0`): the
  acceleration estimate is large enough that the BCa
  transform sends one endpoint outside `(0, 1)`. Standard
  fallback per the same reference.
- Jackknife denominator 0 (all leave-one-out values
  equal): `a = 0`, the BCa formula reduces to BC
  (bias-corrected percentile), which is the same as
  percentile when `z0 == 0` and shifted by `z0` otherwise.
  This is not a fallback; it's a degenerate-acceleration
  case the BCa formula handles natively.

## B21.2 Schema changes

Add TWO new audit fields to all 5 RollupRow schemas in
`benchmarks/bootstrap_manifest.py`:

```python
# BOOTSTRAP CONFIG group (existing)
bootstrap_seed: int
bootstrap_n_resamples: int = Field(ge=0)
bootstrap_confidence: float = 0.95
bootstrap_rng_algorithm: str = "PCG64"
# B21 / D-B16.2: NEW audit fields
bootstrap_ci_method: str = "percentile"  # "percentile" | "bca"
bootstrap_ci_fallback_reason: str | None = None  # None | "p0_at_edge" | "a_overshoot"
bootstrap_numpy_version: str
```

`EnsembleLiftRollupRow` gets ONE additional field for the
oracle bootstrap's fallback reason (the oracle bootstrap is
INDEPENDENT from the main and may fall back independently;
R-B20-2a contract preserved):

```python
# B21 / D-B16.2 + B20: per-row oracle bootstrap fallback
# reason; None when the main bootstrap path produced BCa
# bounds without fallback OR when n_oracle_cells_paired == 0.
oracle_ci_fallback_reason: str | None = None
```

The schema defaults preserve pre-B21 parquet shard
semantics on load: any shard written before B21 had
percentile bounds, so `"percentile"` is the safe default.
Aggregators that ARE running B21 code pass `"bca"`
explicitly.

## B21.3 Aggregator changes

All 5 aggregators (B5 `bootstrap_rollup.py`, B6
`bootstrap_pairwise.py`, B7 `bootstrap_training_time.py`,
B8 `bootstrap_hpo_uplift.py`, B16 `bootstrap_ensemble_lift.py`)
update their `entity_block_bootstrap_ci` call sites:

```python
# Before
mean, ci_lo, ci_hi = entity_block_bootstrap_ci(
    losses, entity_ids,
    n_resamples=n_resamples,
    confidence=BOOTSTRAP_CONFIDENCE,
    seed=BOOTSTRAP_DEFAULT_SEED,
    metric_fn=...,
)

# After
mean, ci_lo, ci_hi, fallback_reason = entity_block_bootstrap_ci(
    losses, entity_ids,
    n_resamples=n_resamples,
    confidence=BOOTSTRAP_CONFIDENCE,
    seed=BOOTSTRAP_DEFAULT_SEED,
    metric_fn=...,
    ci_method=_bootstrap_aggregate.BOOTSTRAP_DEFAULT_CI_METHOD,
)
```

The rollup row constructor adds:

```python
RollupRow(
    ...
    bootstrap_ci_method=_bootstrap_aggregate.BOOTSTRAP_DEFAULT_CI_METHOD,
    bootstrap_ci_fallback_reason=fallback_reason,
    ...
)
```

Sentinel rows pass `bootstrap_ci_method=_bootstrap_aggregate.
BOOTSTRAP_DEFAULT_CI_METHOD` (currently `"bca"`) and
`bootstrap_ci_fallback_reason=None`. The renderer does NOT
display either audit field; both are parquet-shard audit
columns only.

The B16 ensemble-lift aggregator's TWO bootstrap calls
both pass `ci_method="bca"`; the main writes to
`bootstrap_ci_fallback_reason`, the oracle writes to
`oracle_ci_fallback_reason`.

## B21.4 New module constant

In `benchmarks/report/_bootstrap_aggregate.py`:

```python
# B21 / D-B16.2: default CI method for all 5 aggregators.
# Aggregators read this via late-binding lookup so a test
# can monkeypatch the canonical source module to verify
# the percentile-path fallback contract end-to-end (test
# #13 below).
BOOTSTRAP_DEFAULT_CI_METHOD: str = "bca"
```

Exported via `__all__`. The 5 aggregators all read it
late-bound (per the B20 R1 arch-I1 closure pattern):
`_bootstrap_aggregate.BOOTSTRAP_DEFAULT_CI_METHOD`.

## B21.5 Dependency surface

`scipy.stats.norm` is already a transitive dependency
(`benchmarks/stats/friedman.py` uses
`scipy.stats.friedmanchisquare`;
`benchmarks/experiments/ensemble_lift.py` uses
`scipy.stats.wilcoxon`). No new top-level dependency.

Import locally inside `entity_block_bootstrap_ci` to
avoid module-load cost on the percentile-only call sites
that may exist in tests:

```python
def entity_block_bootstrap_ci(...):
    ...
    if ci_method == "bca":
        from scipy.stats import norm
        ...
```

## B21.6 Test surface

### Existing tests touched

Each fixture site that constructs a RollupRow needs 2 (or
3 for `EnsembleLiftRollupRow`) new kwargs:

1. **`tests/benchmarks/test_bootstrap_manifest.py`** —
   each of the 5 RollupRow factory helpers
   (`_make_*_rollup_row`) defaults
   `bootstrap_ci_method="bca"` (documentary; matches the
   v1 default after migration) and
   `bootstrap_ci_fallback_reason=None`. The
   `EnsembleLiftRollupRow` factory also defaults
   `oracle_ci_fallback_reason=None`. Field-round-trip
   tests get one assertion per new field.
2. **5 aggregator-test files**: existing
   `aggregate_*` integration tests assert CI bound
   ordering (`lo <= mean <= hi`); these continue to pass
   on BCa bounds. NO test asserts specific BCa bound
   values (would be fragile to seed changes).
3. **`tests/benchmarks/test_b17_byte_identity_pins.py`**:
   the 5 fixture sites add `bootstrap_ci_method="bca"`
   and the fallback fields (and the oracle fallback for
   `EnsembleLiftRollupRow`). The byte-pin regex on the
   rendered CI cell shape is unchanged (the renderer
   doesn't surface these audit fields).
4. **`tests/benchmarks/test_b19_n_pair_grid.py`** + **
   `tests/benchmarks/test_b20_oracle_delta_ci.py`**: the
   helper-constructed `EnsembleLiftRollupRow` rows get
   the 3 new fields (`bootstrap_ci_method="bca"`, both
   `*_fallback_reason=None`).
5. **`tests/benchmarks/test_ensemble_lift_report_b16.py`**:
   the `_make_rollup_row` factory adds the 3 new oracle-
   inclusive fields.

### NEW B21 tests

`tests/benchmarks/test_b21_bca_ci.py` (NEW; 13 tests):

1. `test_bca_returns_4tuple_with_none_fallback_on_happy_path`:
   call `entity_block_bootstrap_ci(losses, entity_ids,
   ci_method="bca")` on a fixture with skewed losses
   (e.g., `losses = [0.1, 0.1, 0.2, 0.3, 0.5, 1.0]`,
   `entity_ids = [0, 1, 2, 3, 4, 5]`). Assert the return
   is a 4-tuple, the first three are `(mean, lo, hi)` with
   `lo <= mean <= hi`, and the fourth is `None` (no
   fallback fired).
2. `test_bca_lo_lt_hi_on_non_degenerate_fixture`: same
   fixture, assert `lo < hi` strictly (non-degenerate
   width).
3. `test_bca_falls_back_on_p0_at_zero_edge`: construct a
   fixture where every resampled stat is GREATER than the
   ground-truth mean. Easiest path: monkeypatch `metric_fn`
   to add a constant so the bootstrap distribution is
   shifted above the unresampled estimate. Assert the
   4-tuple's fallback_reason is `"p0_at_edge"` AND the
   `(lo, hi)` match the percentile CI on the same
   resampled distribution.
4. `test_bca_falls_back_on_p0_at_one_edge`: symmetric
   fixture to test #3, where every resampled stat is at
   or below the ground-truth mean. Same assertions.
5. `test_bca_falls_back_on_acceleration_overshoot`:
   construct a fixture where the acceleration `a` is large
   enough that `1 - a*(z0 + z_alpha) <= 0` for at least
   one endpoint. Easiest path: extreme skew (`losses =
   [0, 0, 0, 0, 0, 100]` with 6 distinct entities).
   Assert fallback_reason is `"a_overshoot"`.
6. `test_bca_a_eq_zero_when_jackknife_denominator_zero`:
   fixture with all entities producing the SAME
   `metric_fn(losses_minus_entity)` value, e.g., 3
   identical entities each carrying the same row losses.
   The jackknife `deviations` vector is all-zero, denom is
   0, `a` is set to 0.0. Assert fallback_reason is `None`
   (this is NOT a fallback; BCa reduces to BC).
7. `test_bca_matches_percentile_when_distribution_symmetric`:
   fixture where the bootstrap distribution is symmetric
   around the ground-truth mean (e.g., `losses = [0.4,
   0.6]` with 2 entities, `n_resamples=10_000`). Assert
   BCa `(lo, hi)` is within `1e-3` of percentile `(lo,
   hi)` (Monte-Carlo tolerance).
8. `test_bca_differs_from_percentile_when_distribution_skewed`:
   skewed fixture; assert `abs(bca_lo -
   percentile_lo) > 1e-3` OR `abs(bca_hi - percentile_hi)
   > 1e-3` (at least one endpoint shifted).
9. `test_bca_n_entities_one_returns_collapsed_tuple`:
   single entity → existing degenerate return widens to
   `(mean, mean, mean, None)`.
10. `test_primitive_default_ci_method_is_bca`: assert the
    default value of the new `ci_method` kwarg is `"bca"`
    via `inspect.signature(entity_block_bootstrap_ci)`.
11. `test_aggregator_writes_bootstrap_ci_method_bca`:
    run one aggregator (e.g., B5 raw-loss) end-to-end on
    a stub fixture; assert the emitted RollupRow's
    `bootstrap_ci_method == "bca"` AND
    `bootstrap_ci_fallback_reason is None`.
12. `test_aggregator_writes_fallback_reason_when_bca_falls_back`:
    monkeypatch `_bootstrap_aggregate.
    BOOTSTRAP_DEFAULT_CI_METHOD` to `"bca"` (already
    default) AND inject a stub fixture that triggers
    `p0_at_edge`. Assert the emitted row's
    `bootstrap_ci_fallback_reason == "p0_at_edge"` AND
    the bounds match the percentile interval (cross-
    column inequality with the BCa would-be-bounds is too
    fragile; assert just the audit field + bound
    ordering).
13. `test_ensemble_lift_aggregator_writes_independent_oracle_fallback_reason`:
    fixture with main bootstrap on a BCa-happy
    distribution AND oracle bootstrap on a `p0_at_edge`
    distribution (asymmetric). Assert the emitted
    `EnsembleLiftRollupRow` has
    `bootstrap_ci_fallback_reason is None` (main) AND
    `oracle_ci_fallback_reason == "p0_at_edge"` (oracle).
    Pins the R-B21-7 independence contract.

Expected test delta after the build:
- Existing tests: 886 → 886 (no count change; fixtures
  updated in place across 6 sites).
- B21-new: 13 tests.
- Total: 886 + 13 = 899 expected post-refactor.

## B21.7 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B21-Risk-1 | Adding 2 (or 3 for ensemble_lift) audit fields to all 5 RollupRow schemas breaks every existing fixture site. | Medium | The 6 fixture sites are enumerated in B21.6. Each gets documentary defaults; existing tests pass byte-equivalent. Schema defaults `"percentile"` + None preserve pre-B21 parquet shard semantics. |
| R-B21-Risk-2 | The primitive's return signature widens from 3-tuple to 4-tuple. Every existing call site (5 aggregators + 2 in ensemble_lift) must unpack 4 values; missing one is a silent runtime error. | Medium | A grep audit over `entity_block_bootstrap_ci(` in `benchmarks/` enumerates all 7 call sites; each gets updated in this phase. Pyright catches the unpack arity mismatch at type-check time. |
| R-B21-Risk-3 | The BCa fallback paths are HARD TO TRIGGER in production fixtures, so the fallback tests rely on constructed-skew or monkeypatch injection seams. | Medium | Tests #3, #4, #5 construct fixtures with explicit skew that produces deterministic fallback behavior. Test #6 covers the `a == 0` degenerate (NOT a fallback). The fallback paths are also exercised at the primitive level (tests #3-#5) AND end-to-end (test #12). |
| R-B21-Risk-4 | BCa CI bounds DIFFER from percentile bounds on the same fixture; any existing test asserting specific bound values would break. | Low | Grep audit shows ALL existing aggregator tests assert ordering (`lo <= mean <= hi`) or qualitative properties (`hi - lo > 0`), not specific bound values. Byte-pin fixtures (B17, B19, B20) construct rows directly with hardcoded bounds and are decoupled from the actual bootstrap. |
| R-B21-Risk-5 | Jackknife `O(n_entities)` extra metric_fn calls per dataset slows down the aggregator wall-clock. | Low | n_entities is typically ≤ 100 (5 seeds × 5 folds × ≤ 4 datasets per cell pair). Each jackknife call is metric_fn on `(n_rows - rows_for_entity_i)` which is microseconds. For Amex-tier 6M-row datasets the n_entities is still small; the jackknife cost is dominated by the resample loop. |
| R-B21-Risk-6 | scipy.stats.norm import inside the primitive triggers a lazy module load on first BCa call; aggregator wall-clock includes the load. | Low | scipy is already imported by `benchmarks/stats/friedman.py` and `benchmarks/experiments/ensemble_lift.py`; the load typically happens at process start. Even on a cold process the load is ~30ms. |

## B21.8 Implementation outline

1. **Constant**: add `BOOTSTRAP_DEFAULT_CI_METHOD: str =
   "bca"` to `benchmarks/report/_bootstrap_aggregate.py`
   next to the existing seeds + ceiling. Export via
   `__all__`.
2. **Primitive**: extend `entity_block_bootstrap_ci` in
   `benchmarks/metrics/bootstrap.py` with the `ci_method`
   kwarg + the BCa branch + the 4-tuple return. Implement
   `_bca_percentiles` helper as a private module-level
   function. Import `scipy.stats.norm` lazily inside the
   BCa branch.
3. **Schemas**: add `bootstrap_ci_method: str =
   "percentile"` and `bootstrap_ci_fallback_reason: str |
   None = None` to all 5 RollupRow schemas in
   `benchmarks/bootstrap_manifest.py`.
   `EnsembleLiftRollupRow` additionally gets
   `oracle_ci_fallback_reason: str | None = None`.
4. **Aggregators**: update all 5 `entity_block_bootstrap_ci`
   call sites (7 total, 2 in ensemble_lift) to unpack 4
   values AND pass `ci_method=_bootstrap_aggregate.
   BOOTSTRAP_DEFAULT_CI_METHOD`. Plumb the new fallback
   reason into the row constructor.
5. **Sentinel emit helpers**: update each aggregator's
   `_emit_sentinel_row` helper to hardcode
   `bootstrap_ci_method=_bootstrap_aggregate.
   BOOTSTRAP_DEFAULT_CI_METHOD` and
   `bootstrap_ci_fallback_reason=None`. The
   `EnsembleLiftRollupRow` sentinel also hardcodes
   `oracle_ci_fallback_reason=None`.
6. **Update existing fixtures**: the 6 sites enumerated in
   B21.6 get the new fields with documentary defaults.
7. **NEW tests**: add `tests/benchmarks/test_b21_bca_ci.py`
   with the 13 tests.
8. **Verify**: ruff + pyright clean; 899 tests pass.

## Deferred

- **D-B21.1**: surface `bootstrap_ci_method` and
  `bootstrap_ci_fallback_reason` in the renderer markdown
  output as a footnote when ANY row in the report has a
  non-None fallback reason. v1 keeps these as parquet-
  shard audit columns only; rendering them surfaces
  per-dataset BCa health to readers without making the
  table cells noisier.
- **D-B21.2**: configurable `ci_method` per-experiment via
  `ExperimentSpec.bootstrap_ci_method`. v1 hard-defaults to
  BCa via `BOOTSTRAP_DEFAULT_CI_METHOD`; per-experiment
  override would let a reviewer test the percentile path
  on a specific aggregator without monkeypatching.
- **D-B21.3**: ABC (approximate bootstrap confidence)
  intervals as a third method. ABC is faster than BCa
  (no jackknife) but requires a smooth `metric_fn`. The
  current metric_fns (nanmean, sqrt(nanmean)) qualify, but
  the deferral is intentional to keep B21 scope tight.
