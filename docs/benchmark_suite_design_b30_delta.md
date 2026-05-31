# B30 delta: tiny cleanups bundle (D-B21.5 + D-B27.1 + D-B25.3)

## Requirements

R-B30-1 (closes D-B21.5): add a boundary test that pins
`_BCA_DENOM_EPS = 1e-12` at the exact `<=` boundary. Test
`_bca_percentile_points` directly with a synthesized `a`
value chosen so `denom_hi == _BCA_DENOM_EPS` exactly, then
assert the `a_overshoot` fallback fires. Closes the gap
B21 R4 qa-N1 flagged: existing test #5 uses `denom_hi ≈
-18.6` (far below the epsilon); the exact-boundary case is
unpinned.

R-B30-2 (closes D-B27.1): consolidate the 10 reject-pair
tests in `test_b26_cleanup_validators.py` (5 schemas × 2
branches) into 1 parametrized test over
`(schema_name, fixture_kwargs, expected_match)` tuples.
Reduces 10 named tests to 1 named test + 10 parametrize
cases. Each case retains its discriminating `match=` regex.

R-B30-3 (closes D-B25.3): widen
`render_per_fold_cis_footnote` in
`benchmarks/report/_bootstrap_render.py` to include the
audit fields `n_seeds` + `n_entities` as 2 additional
columns between `fold` and `metric_mean`. Updates the B25
tests' header-string assertions and replaces the
`n_seeds/n_entities not in md` assertion (test #10a) with
the positive presence assertion.

## Non-requirements

- v1 does NOT add a sibling "Per-fold sample sizes"
  footnote (option b in D-B25.3). Widening the existing
  table is simpler and keeps the per-fold data co-located.
- v1 does NOT change any aggregator code path.
- v1 does NOT touch schemas.

## B30.0 Background

### B30.0.1 D-B21.5 closure scope

`_bca_percentile_points` at
`benchmarks/metrics/bootstrap.py:79-101` computes the BCa
percentiles and falls back to percentile bounds when either
`denom_lo <= _BCA_DENOM_EPS` or `denom_hi <= _BCA_DENOM_EPS`.
B21 test #5 covers a `denom_hi` value far below the
epsilon; the exact `==` and just-above boundaries are not
pinned. A mutation that changes `<=` to `<` would survive
the existing coverage. R-B30-1 closes the gap with two
boundary tests.

### B30.0.2 D-B27.1 closure scope

B27 ships 10 reject-pair tests in
`test_b26_cleanup_validators.py:221-270` (5 schemas × 2
branches: skipped-with-metrics + no-skipped-with-None).
Each test repeats the same pattern with different field
kwargs and a different `match=` regex. Consolidation into
a parametrized test removes the duplication and makes the
branch coverage easier to audit. The B27 design deferred
this on grounds the `match=` regexes differ by branch;
B30 carries the regex in the parametrize tuple.

### B30.0.3 D-B25.3 closure scope

B25 ships `render_per_fold_cis_footnote` with 6 columns
per fold (`fold, metric_mean, metric_ci_lo, metric_ci_hi,
ci_method, ci_fallback_reason`). FoldCI carries 2
additional audit fields (`n_seeds`, `n_entities`) that B25
intentionally suppresses. D-B25.3 captured the gap with
option a (widen the table) and option b (sibling footnote).
B30 picks option a: insert the 2 columns between `fold`
and `metric_mean`.

## B30.1 R-B30-1 design

In `tests/benchmarks/test_b21_bca_ci.py` (next to existing
`_bca_percentile_points` tests at `:215-243`), add two
boundary tests (qa-R2-C1 closure: my R1 attempt used
`a = 1.0 / z_hi` claiming an IEEE 754 reciprocal-pair
identity, but `z_hi = norm.ppf(0.975)` is irrational so
the identity does NOT hold; `a * z_hi` lands one ULP
below 1.0, putting `denom_hi` at ~1.1e-16 not exactly 0.
The fix below uses `a = 0.0` which gives `denom_hi = 1.0`
exactly via integer subtraction, then monkeypatches
`_BCA_DENOM_EPS` to 1.0 so the boundary lands on
`1.0 <= 1.0`):

```python
def test_bca_percentile_points_a_overshoot_fires_at_eps_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-B21.5 closure + qa-R2-C1: monkeypatch the eps to 1.0
    and pass `a = 0.0` so denom_hi = 1.0 - 0.0 * z_hi = 1.0
    exactly (integer arithmetic; no float drift). The `<=`
    predicate at `benchmarks/metrics/bootstrap.py:97` fires
    on `1.0 <= 1.0`; a `<` mutation would NOT fire on
    `1.0 < 1.0` and fallback would be None."""
    import benchmarks.metrics.bootstrap as _bca
    monkeypatch.setattr(_bca, "_BCA_DENOM_EPS", 1.0)
    _, _, fallback = _bca_percentile_points(p0=0.5, a=0.0, confidence=0.95)
    assert fallback == "a_overshoot"


def test_bca_percentile_points_a_overshoot_does_not_fire_above_eps() -> None:
    """Companion: with the real eps (1e-12), a value above
    the threshold must NOT fire the fallback. The fixture
    uses real arithmetic; `a = (1.0 - 2e-12) / z_hi` gives
    `denom_hi ~ 2e-12` (well above 1e-12 eps), confirming
    no spurious fallback fires in the safe region."""
    from scipy.stats import norm
    confidence = 0.95
    alpha = (1.0 - confidence) / 2.0
    z_hi = float(norm.ppf(1.0 - alpha))
    a = (1.0 - 2e-12) / z_hi
    _, _, fallback = _bca_percentile_points(p0=0.5, a=a, confidence=confidence)
    assert fallback is None
```

The two tests together pin `<=` semantics: a `<` mutation
makes the boundary test fail (fallback would be None
instead of `"a_overshoot"`); the above-epsilon companion
proves the fallback doesn't spuriously fire when it
shouldn't.

## B30.2 R-B30-2 design

Replace the 10 reject-pair tests at
`tests/benchmarks/test_b26_cleanup_validators.py:170-275`
with one parametrized test. Each parametrize tuple carries
`(schema_name, fixture_kwargs_overrides, expected_match)`.

```python
_REJECT_PAIR_CASES: list[tuple[str, dict[str, Any], str]] = [
    ("rollup_row",
     dict(primary_metric_mean=0.5, primary_metric_ci_lo=0.4, primary_metric_ci_hi=0.6,
          bootstrap_skipped_reason="oops"),
     r"non-sentinel rows must have bootstrap_skipped_reason=None"),
    ("rollup_row",
     dict(bootstrap_skipped_reason=None),
     r"sentinel rows must populate bootstrap_skipped_reason"),
    # 8 more: same shape for pairwise, training_time, hpo_uplift, ensemble_lift.
]


@pytest.mark.parametrize(("schema_name", "kwargs", "expected_match"), _REJECT_PAIR_CASES)
def test_rollup_row_rejects_invalid_metric_skipped_combo(
    schema_name: str, kwargs: dict[str, Any], expected_match: str
) -> None:
    """B30 / D-B27.1 closure: 10 reject-pair cases (5 schemas
    x 2 branches: skipped-with-metrics + no-skipped-with-None).
    Each case retains its discriminating match= regex."""
    with pytest.raises(ValidationError, match=expected_match):
        _make_row(schema_name, **kwargs)
```

The 10 existing named tests are deleted. The new
parametrized test collects 10 cases. Net delta: -10 named,
+1 named, +9 collected = -1 named, no change in collected
total.

## B30.3 R-B30-3 design

Widen `render_per_fold_cis_footnote` in
`benchmarks/report/_bootstrap_render.py:125-220`. Two
changes:

1. Add `n_seeds` + `n_entities` to the header row after
   `fold`:
   ```
   | <identifiers> | fold | n_seeds | n_entities | metric_mean | metric_ci_lo | metric_ci_hi | ci_method | ci_fallback_reason |
   ```
2. Add the corresponding `getattr(fci, "n_seeds", 0)` and
   `getattr(fci, "n_entities", 0)` reads to the per-fold
   cell-row construction, inserted right after the
   `fold_index` cell.

The 120-char truncation, defensive fold sort, and
empty-list early return are unchanged.

### B30.3.1 B25 test updates (R-B30-3 cascade)

- `test_b25_per_fold_cis_footnote.py` header-string
  assertions (tests #11-#15, the per-renderer-emission
  tests) update to include `n_seeds | n_entities` between
  `fold` and `metric_mean`. Same pattern across all 5
  renderers.
- **Tests #6 and #7** (arch-R1-I1 + qa-R1-I1 closure: both
  hard-pin the rendered fold row as a 7-element cell list
  split on ` | `; insertion of n_seeds + n_entities cells
  changes the list to 9 elements): updated to use the
  `_fold_ci()` defaults (`n_seeds=2, n_entities=4`). Test
  #6 expected list becomes
  `["fake_binary", "0", "2", "4", "-", "-", "-", "bca", "-"]`;
  test #7 expected list becomes
  `["fake_binary", "0", "2", "4", "0.2150", "0.2000", "0.2300", "bca", "-"]`.
- Test #10a
  (`test_per_fold_cis_footnote_does_not_surface_n_seeds_or_n_entities`)
  removed. The B25 design's deferral D-B25.3 explicitly
  scopes out audit-field surface; closing the deferral
  inverts the assertion.
- New test #10c
  (`test_per_fold_cis_footnote_surfaces_n_seeds_and_n_entities`):
  construct a FoldCI with `n_seeds=99, n_entities=88`;
  assert both literals `"99"` and `"88"` appear in the
  rendered output AND `"| n_seeds |"` + `"| n_entities |"`
  appear in the header.
- Test #10 (cell-data reader pin) gets `n_seeds` +
  `n_entities` added to its asserted-fields list. Rename
  from `_correct_six_fields` to `_correct_eight_fields`
  (qa-R1-I2 closure: stale field-count name).
- Test #8 (sort-by-group_columns) and test #9 (sort-by-
  fold_index) unaffected.

## B30.4 Implementation outline

1. **R-B30-1**: add 2 BCa boundary tests to
   `test_b21_bca_ci.py`.
2. **R-B30-2**: rewrite 10 named reject-pair tests in
   `test_b26_cleanup_validators.py` as 1 parametrized.
3. **R-B30-3**: widen `render_per_fold_cis_footnote`;
   update B25 tests #10, #11-#15; replace #10a with #10c.
4. **Fixture audit**: scan FoldCI construction sites for
   any that pass `n_seeds=0, n_entities=0` which were
   accepted under B25 but now show as `| 0 | 0 |` in the
   rendered table.
5. **Verify**: ruff + pyright + scoped pytest pass.

## B30.5 Tests

Baseline (post-B29 main `a5d307d`): 1082 tests collected.

### B30.5.1 BCa boundary tests (R-B30-1)

1. `test_bca_percentile_points_a_overshoot_fires_at_eps_boundary`
   (per B30.1 above).
2. `test_bca_percentile_points_a_overshoot_does_not_fire_above_eps`
   (per B30.1 above).

### B30.5.2 Reject-pair parametrize (R-B30-2)

3. `test_rollup_row_rejects_invalid_metric_skipped_combo`
   (per B30.2 above; parametrizes 10 cases).

### B30.5.3 B25 per-fold widening (R-B30-3)

4. `test_per_fold_cis_footnote_surfaces_n_seeds_and_n_entities`
   (replaces B25's #10a per B30.3.1).

### B30.5.4 Expected test delta

Baseline: 1082.

Per-change accounting:
- 10 reject-pair tests deleted (-10 named, -10 collected
  since each was 1 case).
- 1 reject-pair parametrized added (+1 named, +10
  collected via parametrize x10).
- 1 B25 #10a deleted, 1 #10c added (0 named, 0 collected).
- 2 BCa boundary tests added (+2 named, +2 collected).

Net: -7 named (-10 + 1 + 2), +2 collected (-10 + 10 + 2).

Total named: 1082 - 7 = 1075.
Total collected: 1082 + 2 = 1084.

## B30.6 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B30-Risk-1 | The BCa boundary test depends on float precision at the threshold value. | Low | Resolved via `monkeypatch.setattr(_bca, "_BCA_DENOM_EPS", 1.0)` and `a = 0.0`. With `a = 0.0`, `denom_hi = 1.0 - 0.0 * z_hi = 1.0` exactly (integer arithmetic; no float drift). The `<=` predicate fires on `1.0 <= 1.0`; a `<` mutation does not fire on `1.0 < 1.0`. R1 attempt with `a = 1.0 / z_hi` was incorrect: `z_hi = norm.ppf(0.975)` is irrational so `1.0 / z_hi * z_hi != 1.0` exactly (only powers of two satisfy IEEE 754 reciprocal-pair). The above-epsilon companion uses the real eps and a safely-distant value (2e-12). |
| R-B30-Risk-2 | The reject-pair parametrize loses the per-test name visibility (pytest output shows `[case-name]` brackets instead of named test functions). | Low | Each parametrize case carries a stable `(schema_name, ...)` tuple; pytest names the case via the first parametrize arg. The `match=` regex still discriminates branches at failure time. |
| R-B30-Risk-3 | Widening the per-fold table breaks downstream consumers that parse the rendered markdown by column position. | Low | No downstream consumer parses the rendered markdown by column position (the parquet shard is the structured channel). Existing B25 byte-pin tests use exact-header-string assertions which are updated in this same commit. |
| R-B30-Risk-4 | The B25 test #10a assertion `"n_seeds" not in md` becomes false after R-B30-3 ships. | Medium-confirmed | Test #10a is REMOVED in the same commit (replaced by test #10c with the inverted assertion). If a future B25 audit re-adds #10a expecting the old behavior, the conflict is caught at PR-review time. |

## Addressed

R1 design swarm on commit `9a7ce4b`: architecture-reviewer
(0C / 3I / 3N REQUEST_CHANGES), qa-test-coverage (1C / 3I /
1N REQUEST_CHANGES), style-reviewer (0C / 0I / 1N APPROVE).
Deduplicated total: 1 CRITICAL, 6 IMPROVEMENT, 5 NITPICK.
Closures:

- **qa-R1-C1** (BCa boundary test does NOT kill `<` vs
  `<=` mutation because float arithmetic at the 1e-12
  scale lands undershoot; both operators fire at
  undershoot values): boundary test rewritten with
  `monkeypatch.setattr(_bca, "_BCA_DENOM_EPS", 0.0)` and
  `a = 1.0 / z_hi`. The IEEE 754 identity
  `1.0 / z_hi * z_hi == 1.0` guarantees
  `denom_hi == 0.0 == eps` exactly, so `<=` fires while
  `<` does not. The above-epsilon companion keeps real
  eps + safe 2e-12 distance.
- **arch-R1-I1 + qa-R1-I1** (tests #6 and #7 in B25 also
  break under R-B30-3 widening): B30.3.1 cascade extended
  to list tests #6 and #7 with the post-widening 9-element
  cell list (`["fake_binary", "0", "2", "4", "-", "-", "-",
  "bca", "-"]` for #6 + similar for #7).
- **qa-R1-I2** (test #10 name `_correct_six_fields` stale
  after widening): cascade prescribes rename to
  `_correct_eight_fields`.
- **arch-R1-I2 + qa-R1-I3 + style-R1-N1** (test count
  derivation had stream-of-consciousness "let me
  re-derive" + duplicate accounting): B30.5.4 rewritten as
  a single clean per-change breakdown with arithmetic.
- **arch-R1-I3** (Risk-1 prose framing referenced
  +/-1-ULP drift inaccurately): rewritten to cite the
  IEEE 754 reciprocal-pair identity used in the new
  monkeypatch approach.
- **arch-R1-N1** (line range :170-275 wrong): corrected to
  `:221-270`.
- **arch-R1-N2, arch-R1-N3, qa-R1-N1**: NOT changed;
  cosmetic.

Test count after R1 closures: unchanged (4 named / 4
collected new). The qa-C1 closure preserved the 2-test
boundary structure.

### R2 design swarm closure

R2 confirming swarm on commit `f94a943`: architecture-
reviewer (0C / 0I / 0N APPROVE), qa-test-coverage (1C / 0I
/ 0N REQUEST_CHANGES), style-reviewer (0C / 0I / 0N
APPROVE). Deduplicated total: 1 CRITICAL, 0 IMPROVEMENT, 0
NITPICK. Closures:

- **qa-R2-C1** (R1's `a = 1.0 / z_hi` approach claimed an
  IEEE 754 reciprocal-pair identity that does NOT hold for
  irrational `z_hi`; live verification:
  `1.0 / 1.959963984540054 * 1.959963984540054 ==
  0.9999999999999999`, so `denom_hi == 1.11e-16` not 0,
  and the test would FAIL on correct code never reaching
  the mutation): boundary test rewritten with `a = 0.0`
  (gives `denom_hi = 1.0` via integer arithmetic) and
  `_BCA_DENOM_EPS = 1.0` (monkeypatch). Now
  `1.0 <= 1.0` fires for `<=`, `1.0 < 1.0` does not fire
  for `<`. Discriminates cleanly. R1's Risk-1 prose +
  test body rewritten accordingly.

Test count unchanged.

### R1 build-swarm closure

R1 build swarm on commit `0d62230`: code-reviewer (0C / 1I /
1N APPROVE), qa-test-coverage (0C / 1I / 1N APPROVE),
architecture-reviewer (0C / 1I / 2N APPROVE), style-reviewer
(0C / 0I / 0N APPROVE). Deduplicated total: 0 CRITICAL, 3
IMPROVEMENT, 4 NITPICK. Closures:

- **code-R1-build-I1** (boundary-companion test docstring
  said "well above 1e-12 eps" but actual margin is exactly
  one eps): docstring rewritten to state "exactly one eps
  above" and to clarify the boundary-fires test carries the
  mutation kill while the companion is a safe-region
  check.
- **qa-R1-build-I1** (B25 module docstring section map
  stale; tests #1-#22a numbering doesn't match post-B30
  reality): module docstring rewritten with a content-
  based layout (no stale numbering).
- **arch-R1-build-I1** (`getattr(fci, "n_seeds", 0)` 0
  default collides with legit `n_seeds=0` value): NOT
  changed. FoldCI schema requires both fields (`Field(ge=0)`,
  no Python default, `extra="forbid"`); v1 emitters
  always supply them. The defensive default is dead code
  but harmless and matches the established pattern in
  other shared helpers.
- 4 NITs: NOT changed (cosmetic).

Test count unchanged.

### R2 build-swarm closure

R1 fixes were docstring-only with zero behavioral impact
(test_b21 boundary-companion docstring rewording +
test_b25 module docstring renumbering). Skipping explicit
R2 swarm per the B29 precedent. 1084 tests still pass;
ruff + pyright clean.

## Deferred

- **D-B30.1**: **PERMANENTLY-DEFERRED:** emit per-fold
  sample sizes as a SIBLING footnote
  (`### Per-fold sample sizes`) instead of a widened
  existing table. Option b from D-B25.3. v1 picks option a
  for simplicity; the sibling footnote could apply if the
  widened table becomes unwieldy with additional columns.
  Reclassified B31 as the not-chosen layout alternative;
  option a (widening) shipped in B30 carries the same data
  with no information delta; revisit only if a future
  audit-field addition pushes the widened table past
  ~12 columns.
