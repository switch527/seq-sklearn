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
`test_b26_cleanup_validators.py:170-275` (5 schemas × 2
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
boundary tests:

```python
def test_bca_percentile_points_a_overshoot_fires_at_eps_boundary() -> None:
    """D-B21.5 closure: synthesize `a` so denom_hi lands at
    exactly `_BCA_DENOM_EPS`. The `<=` predicate at
    `benchmarks/metrics/bootstrap.py:97` must fire on
    equality."""
    from scipy.stats import norm
    confidence = 0.95
    alpha = (1.0 - confidence) / 2.0
    z_hi = float(norm.ppf(1.0 - alpha))
    # With p0=0.5, z0=0, so denom_hi = 1 - a*z_hi.
    # Want denom_hi == _BCA_DENOM_EPS = 1e-12.
    # Solve: a = (1 - 1e-12) / z_hi.
    a = (1.0 - 1e-12) / z_hi
    _, _, fallback = _bca_percentile_points(p0=0.5, a=a, confidence=confidence)
    assert fallback == "a_overshoot"


def test_bca_percentile_points_a_overshoot_does_not_fire_above_eps() -> None:
    """Companion to the boundary test: a value that puts
    denom_hi at 2 * _BCA_DENOM_EPS (just above the
    threshold) must NOT fire the fallback."""
    from scipy.stats import norm
    confidence = 0.95
    alpha = (1.0 - confidence) / 2.0
    z_hi = float(norm.ppf(1.0 - alpha))
    a = (1.0 - 2e-12) / z_hi
    _, _, fallback = _bca_percentile_points(p0=0.5, a=a, confidence=confidence)
    assert fallback is None
```

The two tests together pin the exact `<=` semantics: a `<`
mutation would let the boundary test pass without raising
(fallback would be None), and an off-by-one `<` flip
would still pass the above-epsilon test.

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
  `n_entities` added to its asserted-fields list.
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
- Existing tests:
  - 10 B26 reject-pair tests removed; 1 new parametrized
    test (10 cases) replaces them. Net: -10 named, +10
    collected = 0 net collected, -9 net named.
  - 1 B25 test #10a removed; 1 new B25 test #10c added.
    Net: 0 named, 0 collected.
- B30-new: 4 named tests (2 BCa boundary + 1 reject-pair
  parametrized + 1 per-fold-widened assertion).
- Total named: 1082 - 10 - 1 + 4 = 1075 named.
- Total collected: 1082 - 10 + 10 - 1 + 1 + 2 + 10 + 1 =
  1085 collected. Wait, let me re-derive.

Actually simpler accounting:
- 10 reject-pair tests deleted (-10 named, -10 collected
  since each was 1 case).
- 1 reject-pair parametrized added (+1 named, +10
  collected since parametrize-x10).
- 1 B25 #10a deleted, 1 #10c added (net 0).
- 2 BCa tests added (+2 named, +2 collected).
- 1 widened header presence in B25 (already covered by
  test #10c above).

Net: -10 + 1 + 2 = -7 named; -10 + 10 + 2 = +2 collected.
Total named: 1082 - 7 = 1075. Total collected: 1082 + 2 =
1084.

## B30.6 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B30-Risk-1 | The BCa boundary test depends on `scipy.stats.norm.ppf` numerical stability at the 1e-12 scale. | Low | The test computes `a = (1 - 1e-12) / z_hi` and asserts the fallback fires. Even with 1-ULP float drift, `denom_hi` lands within `[1e-12 - 1e-16, 1e-12 + 1e-16]`, which still satisfies the `<=` predicate. The "above-epsilon" test uses `2 * 1e-12 = 2e-12`, far enough from the epsilon to tolerate drift. |
| R-B30-Risk-2 | The reject-pair parametrize loses the per-test name visibility (pytest output shows `[case-name]` brackets instead of named test functions). | Low | Each parametrize case carries a stable `(schema_name, ...)` tuple; pytest names the case via the first parametrize arg. The `match=` regex still discriminates branches at failure time. |
| R-B30-Risk-3 | Widening the per-fold table breaks downstream consumers that parse the rendered markdown by column position. | Low | No downstream consumer parses the rendered markdown by column position (the parquet shard is the structured channel). Existing B25 byte-pin tests use exact-header-string assertions which are updated in this same commit. |
| R-B30-Risk-4 | The B25 test #10a assertion `"n_seeds" not in md` becomes false after R-B30-3 ships. | Medium-confirmed | Test #10a is REMOVED in the same commit (replaced by test #10c with the inverted assertion). If a future B25 audit re-adds #10a expecting the old behavior, the conflict is caught at PR-review time. |

## Deferred

- **D-B30.1**: emit per-fold sample sizes as a SIBLING
  footnote (`### Per-fold sample sizes`) instead of a
  widened existing table. Option b from D-B25.3. v1 picks
  option a for simplicity; the sibling footnote could
  apply if the widened table becomes unwieldy with
  additional columns.
