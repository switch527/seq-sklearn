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
(Efron & Tibshirani 1993, §14.3); the implementation is
`O(n_entities)` extra
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
  - `1 - a*(z0 + z_*)` is at or below a small positive
    epsilon `_BCA_DENOM_EPS = 1e-12` for either endpoint
    (acceleration overshoots, OR finite-precision
    saturates the denominator near 0): fall back to the
    percentile CI for this dataset. The epsilon guard
    (arch-R1-I5 closure) catches the documented
    `denom <= 0` case AND the near-zero finite-precision
    case where `norm.cdf` would produce a value too close
    to 0 or 1 to be numerically meaningful.
  In all three fallback cases the returned CI is the
  percentile interval; the aggregator's audit field
  `bootstrap_ci_method` records the FALLBACK reason via a
  new column `bootstrap_ci_fallback_reason: str | None`
  (one of `None`, `"p0_at_edge"`, `"a_overshoot"`).
- **R-B21-3** Add `bootstrap_ci_method: str = "percentile"`
  audit field to all 5 RollupRow schemas in
  `benchmarks/bootstrap_manifest.py` (B5 `RollupRow`, B6
  `PairwiseRollupRow`, B7 `TrainingTimeRollupRow`, B8
  `HPOUpliftRollupRow`, B16 `EnsembleLiftRollupRow`). The
  schema default `"percentile"` is intentional and
  asymmetric with the aggregator default `"bca"` (arch-R1-I3
  closure): schema-default `"percentile"` matches the
  pre-B21 parquet shard semantics (any shard written before
  this phase used percentile CIs; loading it must label the
  bounds correctly), while aggregators that ARE running B21
  code pass `"bca"` explicitly via the late-bound
  `BOOTSTRAP_DEFAULT_CI_METHOD` constant. The schema
  default is therefore a backward-compat marker, NOT a
  live-default; live emitters always supply the value
  explicitly. Sentinel rows pass
  `bootstrap_ci_method=BOOTSTRAP_DEFAULT_CI_METHOD` (the
  new module constant defaulting to `"bca"`) for the same
  reason.
- **R-B21-4** Add `bootstrap_ci_fallback_reason: str | None
  = None` audit field to all 5 RollupRow schemas. Non-None
  values indicate a per-dataset BCa fallback to percentile
  (the row carries percentile bounds despite
  `bootstrap_ci_method="bca"`).
- **R-B21-5** All 6 existing call sites in 5 aggregator
  modules (4 single-bootstrap aggregators + 2 in B16
  ensemble_lift) pass `ci_method` derived from the
  late-bound `BOOTSTRAP_DEFAULT_CI_METHOD` constant AND
  propagate the new primitive return (mean, lo, hi,
  fallback_reason) into the 4-field write. The primitive's
  return signature widens from `tuple[float, float, float]`
  to `tuple[float, float, float, str | None]` for both
  paths; the percentile path returns `(mean, lo, hi, None)`
  for shape parity. A frozen `BootstrapResult` BaseModel
  return is deferred under D-B21.4 (arch-R1-I4 closure: the
  4-tuple is shippable at v1; a structured return becomes
  useful when D-B21.1 surfaces additional audit fields).
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
  a new `bootstrap_oracle_ci_fallback_reason: str | None =
  None` (oracle); this is the only schema asymmetry between
  the two BCa invocations. The field name uses the
  `bootstrap_*` prefix to parallel the global-audit
  convention (arch-R1-C1 closure: the original name was
  `oracle_ci_fallback_reason` which broke the
  `bootstrap_*` global-audit vs `oracle_metric_*` /
  `primary_metric_*` per-bootstrap-result pattern; renamed
  to `bootstrap_oracle_ci_fallback_reason` so the
  `bootstrap_*` prefix signals "audit channel" while the
  body word `oracle` signals "for the oracle
  invocation").
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
`"a_overshoot"` when BCa falls back to percentile). All
SIX existing call sites unpack 4 values (4 single-bootstrap
aggregators at `bootstrap_pairwise.py:203`,
`bootstrap_hpo_uplift.py:285`, `bootstrap_rollup.py:341`,
`bootstrap_training_time.py:181`, plus 2 in
`bootstrap_ensemble_lift.py:263` (main) + `:318` (oracle);
arch-R1-I1 closure: the original "7" count miscounted).
The new value is written to the rollup row's
`bootstrap_ci_fallback_reason` (or
`bootstrap_oracle_ci_fallback_reason` for the oracle bootstrap).

### B21.1.1 BCa math

After the existing resample loop produces `resampled:
np.ndarray` of shape `(n_resamples,)` and the unresampled
`ground_truth_mean: float`:

The BCa transform is factored into TWO pure helpers
(qa-R2-C1 + arch-R2-I3 closures: separating the jackknife
loop from the percentile-point transform lets test #5
unit-test `a_overshoot` with a constructed `a` value
bypassing the Cauchy-Schwarz bound `|a| <= 1/(6*sqrt(n))`
that makes the overshoot branch theoretically unreachable
via any real `metric_fn`):

```python
from scipy.stats import norm  # module-level import


def _compute_acceleration_from_jackknife(
    jackknife: np.ndarray,
) -> float:
    """Pure function: acceleration `a` from a jackknife array.

    Returns 0.0 when the denominator is 0 (all jackknife values
    equal; BCa reduces to BC).
    """
    m_dot = float(np.mean(jackknife))
    deviations = m_dot - jackknife
    num = float(np.sum(deviations ** 3))
    denom = 6.0 * float(np.sum(deviations ** 2)) ** 1.5
    return 0.0 if denom == 0.0 else num / denom


def _bca_percentile_points(
    p0: float, a: float, confidence: float,
) -> tuple[float, float, str | None]:
    """Pure function: BCa percentile points given `p0`, `a`,
    `confidence`. Returns `(alpha_1, alpha_2, fallback_reason)`.

    The caller feeds the returned percentile points to
    `np.percentile`. `fallback_reason` is one of `None`,
    `"p0_at_edge"`, `"a_overshoot"`.
    """
    alpha = (1.0 - confidence) / 2.0
    if p0 <= 0.0 or p0 >= 1.0:
        return alpha, 1.0 - alpha, "p0_at_edge"
    z0 = float(norm.ppf(p0))
    z_lo = float(norm.ppf(alpha))
    z_hi = float(norm.ppf(1.0 - alpha))
    denom_lo = 1.0 - a * (z0 + z_lo)
    denom_hi = 1.0 - a * (z0 + z_hi)
    # arch-R1-I5 + R2-I2 closure: epsilon guard catches the
    # canonical `denom <= 0` overshoot AND the near-zero
    # finite-precision case where the BCa transform would
    # saturate. _BCA_DENOM_EPS = 1e-12 is conservative; values
    # in (0, 1e-12] would produce CIs essentially identical to
    # the percentile fallback after norm.cdf rounds them.
    if denom_lo <= _BCA_DENOM_EPS or denom_hi <= _BCA_DENOM_EPS:
        return alpha, 1.0 - alpha, "a_overshoot"
    alpha_1 = float(norm.cdf(z0 + (z0 + z_lo) / denom_lo))
    alpha_2 = float(norm.cdf(z0 + (z0 + z_hi) / denom_hi))
    return alpha_1, alpha_2, None


def _bca_percentiles(
    resampled: np.ndarray,
    ground_truth: float,
    rows_by_entity: list[np.ndarray],
    losses_view: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    confidence: float,
) -> tuple[float, float, str | None]:
    """Return `(alpha_1, alpha_2, fallback_reason)`. The caller
    feeds the returned percentile points to `np.percentile`."""
    # Bias correction
    p0 = float(np.mean(resampled <= ground_truth))
    # Acceleration via leave-one-entity-out jackknife
    n_entities = len(rows_by_entity)
    jackknife = np.empty(n_entities, dtype=np.float64)
    for i in range(n_entities):
        # Rows EXCLUDING entity i
        leave_out_rows = np.concatenate(
            [rows_by_entity[j] for j in range(n_entities) if j != i]
        )
        jackknife[i] = float(metric_fn(losses_view[leave_out_rows]))
    a = _compute_acceleration_from_jackknife(jackknife)
    return _bca_percentile_points(p0, a, confidence)
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
  fallback (Efron & Tibshirani 1993, §14.4); the BCa correction
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
bootstrap_oracle_ci_fallback_reason: str | None = None
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
`bootstrap_oracle_ci_fallback_reason`.

## B21.4 New module constants

Two new module-level constants. One in
`benchmarks/report/_bootstrap_aggregate.py` (the
aggregator-side default):

```python
# B21 / D-B16.2: default CI method for all 5 aggregators.
# Aggregators read this via late-binding lookup so a test
# can monkeypatch the canonical source module to verify
# the seam end-to-end (test #12 below).
BOOTSTRAP_DEFAULT_CI_METHOD: str = "bca"
```

Exported via `__all__`. The 5 aggregators all read it
late-bound (per the B20 R1 arch-I1 closure pattern):
`_bootstrap_aggregate.BOOTSTRAP_DEFAULT_CI_METHOD`.

The second (arch-R2-I2 closure) in
`benchmarks/metrics/bootstrap.py` alongside the existing
`_DEFAULT_N_RESAMPLES` / `_DEFAULT_SEED` /
`_DEFAULT_CONFIDENCE` private constants:

```python
# B21 / D-B16.2: epsilon guard for the BCa acceleration
# transform's denominator. Catches `denom <= 0` overshoot
# AND the near-zero finite-precision saturation case
# (arch-R1-I5 closure). Lives in the same module as
# `_bca_percentile_points` which consumes it; module-private
# (`_`-prefix) because no external consumer should depend
# on the exact value.
_BCA_DENOM_EPS: float = 1e-12
```

NOT exported. Consumed by `_bca_percentile_points` at the
same module-level scope.

## B21.5 Dependency surface

`scipy.stats.norm` is already a transitive dependency
(`benchmarks/stats/friedman.py` uses
`scipy.stats.friedmanchisquare`;
`benchmarks/experiments/ensemble_lift.py` uses
`scipy.stats.wilcoxon`). No new top-level dependency.

Import at module top in `benchmarks/metrics/bootstrap.py`
(arch-R2-I3 closure: the module-level
`_bca_percentile_points` helper consumes `norm.ppf` /
`norm.cdf`; a lazy import inside the outer primitive does
not scope into a separately-defined module-level helper):

```python
# benchmarks/metrics/bootstrap.py top-of-file
import numpy as np
from scipy.stats import norm  # B21 / D-B16.2: BCa transform
```

The import cost is ~30ms on a cold process and is
amortised across every aggregator import that already
pulls in `benchmarks/stats/friedman.py` (which itself
imports scipy.stats); the previously-considered lazy
import added complexity without measurable benefit.

## B21.6 Test surface

### Existing tests touched

Each fixture site that constructs a RollupRow needs 2 (or
3 for `EnsembleLiftRollupRow`) new kwargs:

1. **`tests/benchmarks/test_bootstrap_manifest.py`**:
   each of the 5 RollupRow factory helpers
   (`_make_*_rollup_row`) defaults
   `bootstrap_ci_method="bca"` (documentary; matches the
   v1 default after migration) and
   `bootstrap_ci_fallback_reason=None`. The
   `EnsembleLiftRollupRow` factory also defaults
   `bootstrap_oracle_ci_fallback_reason=None`. Field-round-trip
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

`tests/benchmarks/test_b21_bca_ci.py` (NEW; 15 tests).

**Note on test seams (qa-R1-C1 + qa-R1-C2 closure)**:
the R1 swarm proved by live computation that:
- Test #3's original "shift metric_fn by constant" injection
  is algebraically impossible: `p_0 = mean(resampled <=
  ground_truth)` is invariant under additive shifts of
  `metric_fn` because both sides shift equally.
- Test #5's original "extreme skew (losses = [0,...,0,100])"
  fixture is analytically bounded: with `metric_fn=nanmean`,
  the jackknife acceleration `a` is upper-bounded by
  `(n-1)^1.5 / (6 * n^1.5)` which converges to `1/6 ~ 0.167`
  from below, while `a_overshoot` requires `a >= 1/|z0 +
  z_*|` which is at least `1/1.96 ~ 0.510` at `z0 = 0`. The
  branch is unreachable with any nanmean fixture.

R2 update (qa-R2-C1 closure): the R1 plan for test #5
proposed `metric_fn = exp(nanmean)` on `[0,0,0,0,5]` claiming
`a ~ 1.5`. The R2 swarm verified by manual algebra that this
produces `a = 0.112` (the jackknife values are
`[exp(1.25), exp(1.25), exp(1.25), exp(1.25), exp(0)]` with
deviations `[0.498, 0.498, 0.498, 0.498, -1.992]`, giving
`a = 0.112` not `1.5`). Combined with the
Cauchy-Schwarz bound `|a| <= 1/(6*sqrt(n))`, the
`a_overshoot` branch is THEORETICALLY UNREACHABLE for any
real `metric_fn` on any n_entities >= 2.

The B21.1.1 pseudocode therefore splits the BCa transform
into a pure `_bca_percentile_points(p0, a, confidence)`
helper (see section above). Tests #3, #4, #5 are restated
as UNIT tests on `_bca_percentile_points` with directly-
constructed `(p0, a, confidence)` triples that reach each
fallback branch independent of any metric_fn / jackknife
constraint. Test #6 (a=0 BCa-reduces-to-BC) becomes a
unit test on `_compute_acceleration_from_jackknife`. The
public-primitive happy path is exercised by tests #1, #2,
#7, #8.

1. `test_bca_returns_4tuple_with_none_fallback_on_happy_path`:
   call `entity_block_bootstrap_ci(losses, entity_ids,
   ci_method="bca")` on a fixture with non-degenerate
   losses (`losses = [0.1, 0.1, 0.2, 0.3, 0.5, 1.0]`,
   `entity_ids = [0, 1, 2, 3, 4, 5]`). Assert the return
   is a 4-tuple, the first three are `(mean, lo, hi)` with
   `lo <= mean <= hi`, and the fourth is `None` (no
   fallback fired).
2. `test_bca_lo_lt_hi_on_non_degenerate_fixture`: same
   fixture, assert `lo < hi` strictly (non-degenerate
   width).
3. `test_bca_percentile_points_returns_p0_at_edge_when_p0_is_zero`
   (qa-R1-C1 closure: UNIT test on
   `_bca_percentile_points`):
   call `_bca_percentile_points(p0=0.0, a=0.5,
   confidence=0.95)` directly. Assert returned tuple is
   `(0.025, 0.975, "p0_at_edge")`. Concrete value pin: at
   `confidence=0.95`, `alpha = (1 - 0.95)/2 = 0.025`, so
   the percentile alpha points are exactly `(0.025,
   0.975)`. `a=0.5` is arbitrary; the `p0 <= 0.0` branch
   short-circuits before `a` is consulted.
4. `test_bca_percentile_points_returns_p0_at_edge_when_p0_is_one`:
   symmetric: call `_bca_percentile_points(p0=1.0,
   a=0.5, confidence=0.95)`. Assert returned tuple is
   `(0.025, 0.975, "p0_at_edge")`.
5. `test_bca_percentile_points_returns_a_overshoot_when_denom_at_or_below_epsilon`
   (qa-R1-C2 + qa-R2-C1 closure: UNIT test on
   `_bca_percentile_points` with a directly-passed `a`
   that bypasses the Cauchy-Schwarz `|a| <= 1/(6*sqrt(n))`
   bound):
   call `_bca_percentile_points(p0=0.5, a=10.0,
   confidence=0.95)`. At `p0=0.5`, `z0 = norm.ppf(0.5) =
   0.0`. At `confidence=0.95`, `z_hi = norm.ppf(0.975) =
   1.96`. Then `denom_hi = 1 - 10.0 * (0.0 + 1.96) =
   -18.6`, which is `<= _BCA_DENOM_EPS = 1e-12`. Assert
   returned tuple is `(0.025, 0.975, "a_overshoot")`.
   The directly-passed `a=10.0` exceeds the
   theoretical-jackknife upper bound; the test is
   defensive on the fallback routing, not on whether
   real-world data can produce such an `a`.
6. `test_compute_acceleration_returns_zero_when_jackknife_all_equal`
   (qa-R1 closure: UNIT test on
   `_compute_acceleration_from_jackknife`): call the
   helper with `jackknife = np.array([0.5, 0.5, 0.5])`.
   Deviations are `[0.0, 0.0, 0.0]`, denominator is
   `6 * 0^1.5 = 0`. Assert returned `a == 0.0`. Pin the
   degenerate-denominator branch; the BCa transform
   downstream reduces to BC (bias-corrected percentile).
7. `test_bca_matches_percentile_when_distribution_symmetric`:
   fixture where the bootstrap distribution is symmetric
   around the ground-truth mean. Concrete construction:
   call `entity_block_bootstrap_ci` TWICE on the same
   fixture (`losses = [0.4, 0.5, 0.6]` with 3 distinct
   entities, `n_resamples=10_000`, `seed=42`), once with
   `ci_method="percentile"`, once with `ci_method="bca"`.
   Assert `abs(bca_lo - percentile_lo) < 5e-3` AND
   `abs(bca_hi - percentile_hi) < 5e-3` (Monte-Carlo
   tolerance; qa-R1-N2 closure: cross-check is the two-
   invocation comparison with explicit tolerance).
8. `test_bca_differs_from_percentile_when_distribution_skewed`:
   skewed fixture (`losses = [0.1, 0.1, 0.2, 0.5, 1.0,
   2.0]` with 6 distinct entities, same seed); same
   two-invocation cross-check. Assert `abs(bca_lo -
   percentile_lo) > 1e-3` OR `abs(bca_hi - percentile_hi)
   > 1e-3` (at least one endpoint shifted; qa-R1-N2
   closure).
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
12. `test_aggregator_late_binding_ci_method_default_seam`
    (arch-R1-C2 closure: genuine seam test):
    monkeypatch `_bootstrap_aggregate.
    BOOTSTRAP_DEFAULT_CI_METHOD` to `"percentile"`. Run
    one aggregator end-to-end on a stub fixture; assert
    the emitted RollupRow's `bootstrap_ci_method ==
    "percentile"` AND `bootstrap_ci_fallback_reason is
    None` (percentile path doesn't carry fallback
    semantics; the field stays None). This exercises the
    late-binding seam, NOT a no-op self-monkeypatch.
13. `test_aggregator_writes_fallback_reason_when_bca_falls_back`
    (qa-R1-I2 + qa-R2-I2 closure: concrete stateful
    metric_fn): inject a stub `metric_fn` via the
    aggregator's test seam that produces a bootstrap
    distribution where `p_0 = 0.0`. Concrete construction
    (closing the R1 circular spec):

    ```python
    call_count = 0
    def stateful_metric_fn(x: np.ndarray) -> float:
        nonlocal call_count
        call_count += 1
        # Call #1 is the ground-truth unresampled call;
        # all subsequent calls are bootstrap subsets.
        # Pinning ground_truth = -1e9 (below every
        # subsequent stat) yields p_0 = mean([1e9 <= -1e9
        # ...]) = 0.0, reliably tripping `p0_at_edge`.
        return -1e9 if call_count == 1 else 1e9
    ```

    Assert the emitted row's
    `bootstrap_ci_fallback_reason == "p0_at_edge"` AND
    `bootstrap_ci_method == "bca"` AND `lo <= mean <=
    hi`.
14. `test_ensemble_lift_aggregator_writes_independent_oracle_fallback_reason`:
    fixture with main bootstrap on a non-degenerate
    distribution AND oracle bootstrap engineered via the
    same call-counter stateful metric_fn pattern (with a
    SEPARATE call_count for the oracle metric_fn so the
    two bootstraps' first calls are independently the
    "ground-truth" calls). Assert the emitted
    `EnsembleLiftRollupRow` has
    `bootstrap_ci_fallback_reason is None` (main) AND
    `bootstrap_oracle_ci_fallback_reason == "p0_at_edge"`
    (oracle). Pins the R-B21-7 independence contract.
    Cross-reference: B20 test #13 covers R-B20-2a's
    seed-side; this B21 test adds the
    fallback-reason-independence side.
15. `test_b21_audit_fields_survive_parquet_round_trip`
    (qa-R1-I1 closure): construct one row of each of the
    5 RollupRow types with `bootstrap_ci_method="bca"`,
    `bootstrap_ci_fallback_reason="p0_at_edge"` (and
    `bootstrap_oracle_ci_fallback_reason="p0_at_edge"`
    for `EnsembleLiftRollupRow`); write + load via the
    respective `write_*_rollup` / `load_*_rollup`
    functions; assert each new field survives the
    round-trip on each row type. Also construct a second
    set of rows with `bootstrap_ci_method="percentile"`
    and `bootstrap_ci_fallback_reason=None` to verify the
    `pd.NA -> None` coercion on the nullable field.

**Inline pins on existing tests** (qa-R1-I3 closure;
NOT a new test): extend each of the 4 B17 byte-pin
renderer tests (`test_render_pairwise_byte_identity_post_rename`,
`test_render_training_time_byte_identity_post_rename`,
`test_render_hpo_uplift_byte_identity_post_rename`,
`test_render_ensemble_lift_byte_identity_post_rename` in
`tests/benchmarks/test_b17_byte_identity_pins.py`) with
two additional asserts:

```python
# B21 / D-B21.1 deferral: the new audit fields are
# parquet-shard columns only, NOT surfaced in the
# rendered markdown. A future renderer change that
# accidentally exposes them would fail here.
assert "bootstrap_ci_method" not in md
assert "bootstrap_ci_fallback_reason" not in md
```

The 4th test (ensemble-lift) additionally asserts
`"bootstrap_oracle_ci_fallback_reason" not in md`.

Expected test delta after the build:
- Existing tests: 886 → 886 (no count change; fixtures
  updated in place across 6 sites, plus 8 inline asserts
  on 4 byte-pin tests; qa-R1-N1 closure: the 886 baseline
  was confirmed via live pytest --collect-only count on
  the post-B20 main branch tip at commit 3358651).
- B21-new: 15 tests.
- Total: 886 + 15 = 901 expected post-refactor.

## B21.7 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B21-Risk-1 | Adding 2 (or 3 for ensemble_lift) audit fields to all 5 RollupRow schemas breaks every existing fixture site. | Medium | The 6 fixture sites are enumerated in B21.6. Each gets documentary defaults; existing tests pass byte-equivalent. Schema defaults `"percentile"` + None preserve pre-B21 parquet shard semantics. |
| R-B21-Risk-2 | The primitive's return signature widens from 3-tuple to 4-tuple. Every existing call site (4 single-bootstrap aggregators + 2 in ensemble_lift) must unpack 4 values; missing one is a silent runtime error. | Medium | A grep audit over `entity_block_bootstrap_ci(` in `benchmarks/` enumerates all 6 call sites; each gets updated in this phase. Pyright catches the unpack arity mismatch at type-check time. |
| R-B21-Risk-3 | The BCa fallback paths are HARD TO TRIGGER in production fixtures, so the fallback tests rely on constructed-skew or monkeypatch injection seams. | Medium | Tests #3, #4, #5 construct fixtures with explicit skew that produces deterministic fallback behavior. Test #6 covers the `a == 0` degenerate (NOT a fallback). The fallback paths are also exercised at the primitive level (tests #3-#5) AND end-to-end (test #12). |
| R-B21-Risk-4 | BCa CI bounds DIFFER from percentile bounds on the same fixture; any existing test asserting specific bound values would break. | Low | Grep audit shows ALL existing aggregator tests assert ordering (`lo <= mean <= hi`) or qualitative properties (`hi - lo > 0`), not specific bound values. Byte-pin fixtures (B17, B19, B20) construct rows directly with hardcoded bounds and are decoupled from the actual bootstrap. |
| R-B21-Risk-5 | Jackknife `O(n_entities)` extra metric_fn calls per dataset slows down the aggregator wall-clock. | Low | n_entities is typically ≤ 100 (5 seeds × 5 folds × ≤ 4 datasets per cell pair). Each jackknife call is metric_fn on `(n_rows - rows_for_entity_i)` which is microseconds. For Amex-tier 6M-row datasets the n_entities is still small; the jackknife cost is dominated by the resample loop. |
| R-B21-Risk-6 | scipy.stats.norm import inside the primitive triggers a lazy module load on first BCa call; aggregator wall-clock includes the load. | Low | scipy is already imported by `benchmarks/stats/friedman.py` and `benchmarks/experiments/ensemble_lift.py`; the load typically happens at process start. Even on a cold process the load is ~30ms. |

## B21.8 Implementation outline

1. **Constants**: add `BOOTSTRAP_DEFAULT_CI_METHOD: str =
   "bca"` to `benchmarks/report/_bootstrap_aggregate.py`
   next to the existing seeds + ceiling. Export via
   `__all__`. Also add `_BCA_DENOM_EPS: float = 1e-12`
   to `benchmarks/metrics/bootstrap.py` alongside the
   existing `_DEFAULT_*` constants (private,
   `_`-prefixed; not exported).
2. **Primitive**: extend `entity_block_bootstrap_ci` in
   `benchmarks/metrics/bootstrap.py` with the `ci_method`
   kwarg + the BCa branch + the 4-tuple return. Implement
   THREE module-level helpers per the B21.1.1 pseudocode:
   `_compute_acceleration_from_jackknife(jackknife)`,
   `_bca_percentile_points(p0, a, confidence)`, and
   `_bca_percentiles(resampled, ground_truth, rows_by_entity,
   losses_view, metric_fn, confidence)`. Add module-top
   `from scipy.stats import norm` import.
3. **Schemas**: add `bootstrap_ci_method: str =
   "percentile"` and `bootstrap_ci_fallback_reason: str |
   None = None` to all 5 RollupRow schemas in
   `benchmarks/bootstrap_manifest.py`.
   `EnsembleLiftRollupRow` additionally gets
   `bootstrap_oracle_ci_fallback_reason: str | None = None`.
4. **Aggregators**: update all `entity_block_bootstrap_ci`
   call sites (6 total: 4 single-bootstrap aggregators +
   2 in ensemble_lift) to unpack 4 values AND pass
   `ci_method=_bootstrap_aggregate.BOOTSTRAP_DEFAULT_CI_METHOD`.
   Plumb the new fallback reason into the row constructor.
5. **Sentinel emit helpers**: update each aggregator's
   `_emit_sentinel_row` helper to hardcode
   `bootstrap_ci_method=_bootstrap_aggregate.
   BOOTSTRAP_DEFAULT_CI_METHOD` and
   `bootstrap_ci_fallback_reason=None`. The
   `EnsembleLiftRollupRow` sentinel also hardcodes
   `bootstrap_oracle_ci_fallback_reason=None`.
6. **Update existing fixtures**: the 6 sites enumerated in
   B21.6 get the new fields with documentary defaults.
7. **NEW tests**: add `tests/benchmarks/test_b21_bca_ci.py`
   with the 15 tests (13 design-named + 2 added in R1
   closures: arch-R1-C2 split test #12 into seam +
   fallback, qa-R1-I1 added the parquet round-trip test
   #15).
8. **Verify**: ruff + pyright clean; 901 tests pass.

## Addressed

R1 swarm: architecture-reviewer (2C / 5I / 2N
REQUEST_CHANGES), qa-test-coverage (2C / 3I / 2N
REQUEST_CHANGES), style-reviewer (1C / 0I / 0N
REQUEST_CHANGES). Deduplicated total: 5 CRITICAL,
8 IMPROVEMENT, 4 NITPICK. Closures:

- **arch-R1-C1** (`oracle_ci_fallback_reason` broke the
  `bootstrap_*` global-audit vs `oracle_metric_*` /
  `primary_metric_*` per-bootstrap-result naming pattern
  from B20): renamed to
  `bootstrap_oracle_ci_fallback_reason`. The
  `bootstrap_*` prefix is the audit-channel marker; the
  body word `oracle` distinguishes the oracle bootstrap
  from the main. R-B21-7 + the schema spec at B21.2 now
  use the new name throughout.
- **arch-R1-C2** (test #12 monkeypatched the constant to
  its own default value, exercising no seam): split into
  two tests. Test #12 is now the genuine seam test
  (monkeypatch to `"percentile"`, assert aggregator
  surfaces `"percentile"`); test #13 is the fallback test
  with a stateful metric_fn injection that triggers
  `p0_at_edge`. Test count widens from 13 to 15 (the
  qa-R1-I1 parquet round-trip + qa-R1-I2 byte-pin
  non-exposure inline pins also land here).
- **qa-R1-C1** (test #3 / #4 `p0_at_edge` injection via
  constant-shift metric_fn is algebraically impossible
  because `+C` cancels in the `resampled <= ground_truth`
  comparison): tests #3 and #4 are restated as UNIT tests
  on the private `_bca_percentiles` helper with
  constructed `resampled` arrays (`np.ones(100)` and
  `np.zeros(100)`) that directly reach each fallback
  branch. The R1 swarm's live algebraic proof is
  documented inline in the B21.6 seam note.
- **qa-R1-C2** (test #5 `a_overshoot` fixture analytically
  bounded: jackknife `a` with `metric_fn=nanmean` is upper-
  bounded by ~0.167 but `a_overshoot` requires `a >=
  ~0.510`): test #5 is restated as a UNIT test on
  `_bca_percentiles` with an engineered `metric_fn =
  lambda x: float(np.exp(np.nanmean(x)))` and skewed
  `losses_view = [0, 0, 0, 0, 5]` that produces `a ~ 1.5`
  (verified by the R1 swarm via live execution).
- **style-R1-C1** (em dash at line 406 in the
  B21.6 enumeration): replaced with a colon.
- **arch-R1-I1** (call-site count: "5 aggregators + 2 in
  ensemble_lift = 7" actually counts 6 sites since 4
  single-bootstrap aggregators + 2 in ensemble_lift = 6):
  R-B21-5 and B21.1 prose updated to "6 existing call
  sites" with explicit path:line enumeration.
- **arch-R1-I2** (B21.6 says "6 fixture sites" but the
  enumeration shows ~10): B21.6 expanded to enumerate the
  6 fixture sites (`test_bootstrap_manifest.py` factory
  helpers, 5 aggregator-test files, `test_b17_byte_identity_pins.py`,
  `test_b19_n_pair_grid.py`, `test_b20_oracle_delta_ci.py`,
  `test_ensemble_lift_report_b16.py`) AND clarified that
  the byte-pin renderer tests get inline non-exposure
  asserts (qa-R1-I3 closure) rather than a separate test.
- **arch-R1-I3** (schema-default `"percentile"` vs
  sentinel-emit `"bca"` versioning asymmetry undocumented):
  R-B21-3 expanded with the explicit rationale (schema
  default is a backward-compat marker for pre-B21 parquet
  shards; live emitters always supply `"bca"` explicitly).
- **arch-R1-I4** (4-tuple return should be a frozen
  `BootstrapResult` BaseModel for forward-compat): DEFERRED
  under D-B21.4. The 4-tuple is shippable at v1; the
  structured return becomes useful when D-B21.1 surfaces
  additional audit fields in the renderer.
- **arch-R1-I5** (fallback enumeration omits finite-
  precision near-zero denom): R-B21-2 third bullet now
  mandates a `_BCA_DENOM_EPS = 1e-12` guard catching the
  documented `denom <= 0` case AND the near-zero
  precision case. The unit test #5 still asserts the
  fallback fires on the engineered `a > 0.51` overshoot
  case; the near-zero precision case is below the test
  threshold (would require a fixture deliberately tuned
  to land in the `(0, 1e-12]` window).
- **qa-R1-I1** (parquet round-trip for the 3 new audit
  fields unnamed): added test #15
  `test_b21_audit_fields_survive_parquet_round_trip`
  covering all 5 RollupRow types with both `"bca"` +
  None and `"percentile"` + None field values to verify
  the `pd.NA -> None` coercion on the nullable field.
- **qa-R1-I2** (test #12 injection seam not a genuine
  seam): same as arch-R1-C2 closure; test #12 is now the
  late-binding seam test and test #13 names the stateful-
  metric_fn fallback seam explicitly.
- **qa-R1-I3** (no test asserts the renderer does NOT
  surface the new audit fields): added 8 inline asserts
  on the 4 existing B17 byte-pin renderer tests
  (`assert "bootstrap_ci_method" not in md` AND
  `assert "bootstrap_ci_fallback_reason" not in md`,
  with the 4th test additionally asserting on
  `bootstrap_oracle_ci_fallback_reason`). The pin
  catches a future renderer change that accidentally
  exposes the audit channel; documented inline in
  B21.6 under "Inline pins on existing tests".
- **arch-R1-N1** (test #14 should defer the PCG64 stream-
  independence pin to the existing B20 seed-independence
  test): test #14 docstring now cross-references B20 test
  #13 explicitly; the seed-stream-independence pin stays
  at B20 #13 and the new B21 #14 covers only the
  fallback-reason-independence pin.
- **arch-R1-N2** (citation style mixes "Efron 1987" and
  "Efron-Tibshirani 1993 §14"): aligned to "Efron &
  Tibshirani 1993" throughout the B21.0 and B21.1.2
  prose; the textbook reference is canonical for the BCa
  formula derivation.
- **qa-R1-N1** (test count base of 886 unverified):
  captured the post-B20 main tip pytest count via the
  R1 R2 confirming run (commit 3358651, 886 passing).
  B21.6 "expected test delta" section records this
  explicitly.
- **qa-R1-N2** (tests #3 / #4 don't specify cross-check):
  tests #7 / #8 (the symmetric-vs-skewed comparison
  tests) now name the cross-check as "two invocations on
  the same fixture with `ci_method="percentile"` then
  `ci_method="bca"`, compare bound values with explicit
  Monte-Carlo tolerance" (5e-3 for symmetric agreement;
  1e-3 lower-bound for skewed divergence).

Test count after R1 closures: 15 new tests (was 13;
arch-R1-C2 split test #12 into seam + fallback;
qa-R1-I1 added parquet round-trip); total `886 + 15 =
901`.

### R2 confirming swarm closure

R2 confirming swarm on commit `f117413` (post R1 closures):
architecture-reviewer (1C / 5I / 2N REQUEST_CHANGES),
qa-test-coverage (1C / 2I / 1N REQUEST_CHANGES),
style-reviewer (0C / 0I / 0N APPROVE). Deduplicated total:
2 CRITICAL, 6 IMPROVEMENT, 2 NITPICK. Closures:

- **arch-R2-C1** (R-B21-7 prose at :129 had a doubled
  `bootstrap_bootstrap_` prefix typo on the field name AND
  the closure parenthetical at :133-138 inverted the
  before/after of the arch-R1-C1 rename, claiming the
  BROKEN original name was `bootstrap_oracle_ci_fallback_reason`
  when the Addressed log records the broken name as
  `oracle_ci_fallback_reason`): both fixed in-place at
  R-B21-7. The prose now correctly states the original
  name was `oracle_ci_fallback_reason`, renamed to
  `bootstrap_oracle_ci_fallback_reason`.
- **qa-R2-C1** (test #5 fixture
  `metric_fn=exp(nanmean)` on `[0,0,0,0,5]` does NOT
  produce `a ~ 1.5`; manual algebra shows `a = 0.112`,
  insufficient to trigger `a_overshoot`; combined with
  the Cauchy-Schwarz bound `|a| <= 1/(6*sqrt(n))`, the
  `a_overshoot` branch is THEORETICALLY UNREACHABLE for
  any real metric_fn): refactored `_bca_percentiles`
  into THREE module-level helpers
  (`_compute_acceleration_from_jackknife`,
  `_bca_percentile_points`, `_bca_percentiles`). Test #5
  now unit-tests `_bca_percentile_points(p0=0.5, a=10.0,
  confidence=0.95)` directly, bypassing the
  Cauchy-Schwarz bound entirely. The `a_overshoot`
  branch is now reachable with a constructed `a` that
  exceeds the theoretical jackknife bound.
- **arch-R2-I1** (Risk-2 at :663 + B21.8 step 4 at :688
  still said "7 call sites" after R1's I1 closure
  corrected R-B21-5 + B21.1 to 6): propagated the 6
  fix to both surfaces; Risk-2 now reads "4 single-
  bootstrap aggregators + 2 in ensemble_lift = 6 total"
  and step 4 reads "6 total: 4 single-bootstrap + 2 in
  ensemble_lift".
- **arch-R2-I2** (`_BCA_DENOM_EPS` referenced in 3 places
  but never declared in the B21.4 "New module constant"
  section): B21.4 renamed to "New module constants"
  (plural) and now declares `_BCA_DENOM_EPS: float = 1e-12`
  in `benchmarks/metrics/bootstrap.py` alongside the
  existing `_DEFAULT_*` constants. B21.8 step 1 mentions
  both new constants explicitly.
- **arch-R2-I3** (scipy.stats.norm lazy-import inside the
  outer primitive at B21.5 does not scope into the
  module-level `_bca_percentiles` helper): B21.5
  rewritten to mandate a module-top
  `from scipy.stats import norm` import in
  `benchmarks/metrics/bootstrap.py`. The previously-
  considered lazy import added complexity without
  measurable benefit; the ~30ms import cost is amortised
  across the friedman / wilcoxon scipy imports already
  in the aggregator transitive closure.
- **arch-R2-I4** (B21.8 step 7 said "13 tests" + step 8
  said "899 tests pass" after R1's C2 split widened to
  15 / 901): updated step 7 to "15 tests (13 design-named
  + 2 added in R1 closures: arch-R1-C2 split test #12
  into seam + fallback, qa-R1-I1 added the parquet
  round-trip test #15)" and step 8 to "901 tests pass".
- **arch-R2-I5** (`all_row_indices = np.concatenate(
  rows_by_entity)` computed in B21.1.1 pseudocode but
  never read; dead binding): dropped from the new
  `_bca_percentiles` helper pseudocode (the helper does
  not need the full row index since the jackknife
  operates on leave-one-out subsets).
- **qa-R2-I1** (tests #3, #4, #6 don't name the
  `confidence` argument and tests #3/#4 don't specify
  `resampled`): tests rewritten to call
  `_bca_percentile_points` directly with explicit
  `p0=0.0` / `p0=1.0` / `p0=0.5` AND `confidence=0.95`
  AND concrete `a` values. Test #6 calls
  `_compute_acceleration_from_jackknife` with an
  explicit `np.array([0.5, 0.5, 0.5])` jackknife. No
  argument is unnamed.
- **qa-R2-I2** (test #13 stateful metric_fn was self-
  referential: `ground_truth - epsilon` where
  `ground_truth` IS the return of metric_fn call #1):
  test #13 now spells the concrete construction in a
  fenced code block: `call_count = 0; def
  stateful_metric_fn(x): nonlocal call_count;
  call_count += 1; return -1e9 if call_count == 1 else
  1e9`. This pins `ground_truth = -1e9` AND every
  subsequent stat at `1e9`, deterministically yielding
  `p_0 = 0.0`.
- **arch-R2-N1** (R-B21-2 third bullet's enumeration of
  `"a_overshoot"` could be read as suggesting a third
  reason value): NOT changed. The collapse of "canonical
  `denom <= 0`" and "near-zero finite-precision" into
  one `"a_overshoot"` reason is the smaller, cleaner
  choice. The R-B21-2 third bullet now reads more
  clearly with the epsilon-guard explicitly mentioned.
- **arch-R2-N2** (test #14 docstring had embedded closure
  metadata `(arch-R1-N1 closure: ...)`): test #14
  rewritten to drop the inline closure tag; operational
  cross-reference to B20 test #13 retained ("B20 test
  #13 covers R-B20-2a's seed-side; this B21 test adds
  the fallback-reason-independence side").

Test count after R2 closures: 15 new tests; total
`886 + 15 = 901`.

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
- **D-B21.4** (arch-R1-I4 closure): replace the 4-tuple
  return of `entity_block_bootstrap_ci` with a frozen
  `BootstrapResult` pydantic BaseModel carrying named
  fields (`mean`, `ci_lo`, `ci_hi`, `ci_method`,
  `fallback_reason`). Becomes useful when D-B21.1
  surfaces additional audit fields (n_resamples actually
  used, BCa intermediate values for debugging, etc.); the
  v1 tuple is shippable and the structural-typing cost of
  the migration is small.
