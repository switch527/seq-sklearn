# B27 delta: small cleanups bundle (D-B22.5 + D-B26.2 + D-B26.3)

## Requirements

R-B27-1 (closes D-B22.5): correct the B22 R1 closure block
header in `docs/benchmark_suite_design_b22_delta.md`. The
header reads "Deduplicated total: 4 CRITICAL, 7 IMPROVEMENT,
4 NITPICK"; the enumerated closure entries below total
4 CRITICAL + 6 IMPROVEMENT + 3 NITPICK. Update the header to
match the enumeration (1 IMP + 1 NIT were dedup-merged into
other entries without separate listings; no original-review
material to reconstruct).

R-B27-2 (closes D-B26.2): compose the CI-sentinel
`@model_validator(mode="after")` into `EnsembleLiftRollupRow`
alongside its existing `_validate_row_count_invariants`. The
new validator has the same body as the 4 other RollupRow
schemas' `_validate_ci_sentinel_consistency` (B26 / D-B23.2).
EnsembleLift becomes the 5th schema with the CI-sentinel
invariant.

R-B27-3 (closes D-B26.3): extend mixed-reject test coverage
to all 6 partially-set combinations per schema for the
B26-introduced validator. v1 of B26 shipped 2 variants per
schema (mean-only-set, ci_lo-only-set); B27 adds 4 more
(ci_hi-only-set, mean+ci_lo-set, mean+ci_hi-set,
ci_lo+ci_hi-set). All 5 schemas (4 from B26 + 1 from
R-B27-2) get the full 6-variant parametrize.

## Non-requirements

- v1 does NOT change validator semantics. R-B27-2 just adds
  a second validator on the existing schema. The two
  validators compose (both run via pydantic's `mode="after"`
  chain in declaration order).
- v1 does NOT touch the 4 R-B26 schemas' validators (already
  in place from B26).
- v1 does NOT add the D-B26.1 cell-count invariant on
  HPOUpliftRollupRow (still deferred).

## B27.0 Background

### B27.0.1 What B22 left deferred (R-B27-1)

The R1 closure block at
`docs/benchmark_suite_design_b22_delta.md` reports 4C/7I/4N
in the dedup-header but the enumeration only carries 13 of
the implied 15 entries. The 2 missing entries
(1 IMPROVEMENT + 1 NITPICK) were dedup-merged into other
closures during R1 without separate listings. D-B22.5
captured this bookkeeping gap. The fix: update the header
to match the enumeration; the substantive R1 closures are
already complete.

### B27.0.2 What B26 left deferred (R-B27-2)

B23 added a 3-invariant cross-field validator
(`_validate_row_count_invariants`) to `EnsembleLiftRollupRow`.
B26 added a CI-sentinel validator to the OTHER 4 RollupRow
schemas. The 5th schema (`EnsembleLiftRollupRow`) was
deliberately scoped out so the B26 phase did not need to
compose two validators on the same class. D-B26.2 captured
the gap with an explicit "compose by chaining a second
`@model_validator(mode="after")`" prescription noted at
`bootstrap_manifest.py:794-798` (the B26 NOTE comment).

### B27.0.3 What B26 left deferred (R-B27-3)

B26 shipped 2 mixed-reject variants per schema
(mean-only-set, ci_lo-only-set). The `all()` predicate over
a 3-tuple is positionally symmetric so those 2 variants
kill the relevant mutation surfaces. v1 deferred the full
6-variant parametrize as D-B26.3. B27 adds the remaining 4
variants per schema for full position-coverage.

## B27.1 R-B27-1 design

Single doc edit. Change line at
`docs/benchmark_suite_design_b22_delta.md` (the R1 dedup
header) from:

```
Deduplicated total: 4 CRITICAL, 7 IMPROVEMENT, 4 NITPICK.
```

to:

```
Deduplicated total: 4 CRITICAL, 6 IMPROVEMENT, 3 NITPICK
(matches the 13 enumerated closure entries below; 1 IMP +
1 NIT were dedup-merged at R1 without separate listings).
```

## B27.2 R-B27-2 design

In `benchmarks/bootstrap_manifest.py`, add a second
`@model_validator(mode="after")` to `EnsembleLiftRollupRow`
immediately after the existing `_validate_row_count_invariants`:

```python
@model_validator(mode="after")
def _validate_ci_sentinel_consistency(self) -> "EnsembleLiftRollupRow":
    # B27 / D-B26.2 closure: same CI-sentinel invariant as RollupRow.
    # IDENTICAL BODY TO RollupRow._validate_ci_sentinel_consistency;
    # keep all 5 copies in sync.
    metric_fields = (
        self.primary_metric_mean,
        self.primary_metric_ci_lo,
        self.primary_metric_ci_hi,
    )
    all_none = all(f is None for f in metric_fields)
    all_set = all(f is not None for f in metric_fields)
    if not (all_none or all_set):
        raise ValueError(...)
    ...
    return self
```

Update the NOTE comment on
`_validate_row_count_invariants` to remove the D-B26.2
forward pointer (the deferral is now closed).

The two validators compose naturally via pydantic's
mode="after" execution chain. Both run on every construction
including `model_validate`.

**Backward-compat audit**: the new validator is strict on
EnsembleLiftRollupRow. Existing fixtures that construct
EnsembleLiftRollupRow rows must satisfy the invariant.
B17 byte-pin fixtures (non-sentinel rows with populated
metrics): COMPATIBLE. B22/B23 schema-default zero-cell
fixtures: must be audited. The B22 fixture audit in B26
already added `bootstrap_skipped_reason="test_fixture"` to
the shared `common` dict at
`test_b22_per_fold_cis.py:217-224`, which includes the
EnsembleLiftRollupRow row at `:264-275`. So that fixture
is already CI-sentinel-compatible.

B27.3 (fixture audit) verifies live during build.

## B27.3 R-B27-3 design

Replace the per-schema mixed-A + mixed-B tests at
`tests/benchmarks/test_b26_cleanup_validators.py` with a
parametrize covering all 6 variants per schema. For each
of the 5 RollupRow schemas (4 from B26 + EnsembleLift from
B27), parametrize:

| Variant | mean | ci_lo | ci_hi |
|---|---|---|---|
| mixed-1 (mean-only) | 0.5 | None | None |
| mixed-2 (ci_lo-only) | None | 0.4 | None |
| mixed-3 (ci_hi-only) | None | None | 0.6 |
| mixed-4 (mean+ci_lo) | 0.5 | 0.4 | None |
| mixed-5 (mean+ci_hi) | 0.5 | None | 0.6 |
| mixed-6 (ci_lo+ci_hi) | None | 0.4 | 0.6 |

5 schemas x 6 variants = 30 mixed-reject parametrize cases.
Each asserts `pytest.raises(ValidationError,
match=r"must be all-None or all-non-None")`.

The existing 8 mixed-reject tests (2 per schema x 4
schemas) from B26 are removed and replaced by the
parametrize. The EnsembleLift schema gets a new fixture
helper analogous to the 4 in B26.

The 8 reject-pair tests (#17-#24) from B26 stay; B27 adds 2
more for EnsembleLift (one all-set-with-reason, one
all-None-without-reason). Total reject-pair tests: 10.

## B27.4 Implementation outline

1. **R-B27-1**: edit B22 design doc header.
2. **R-B27-2**: add second `@model_validator` to
   `EnsembleLiftRollupRow` in `bootstrap_manifest.py`;
   update the B23 validator's NOTE comment.
3. **R-B27-3**: rewrite `test_b26_cleanup_validators.py`
   to use parametrize for mixed-reject coverage; add
   `_ENSEMBLE_LIFT_BASE` fixture helper + 2 new
   EnsembleLift reject-pair tests + happy non-sentinel +
   happy sentinel + 6 mixed parametrize entries.
4. **Fixture audit**: run the full suite; any EnsembleLift
   construction site that violates the CI-sentinel
   invariant must be repaired in the same commit.
5. **Verify**: ruff + pyright + scoped pytest pass at
   1033 + delta.

## B27.5 Tests

Baseline (post-B26 main `99f5128`): 1033 tests collected.

### B27.5.1 Test rewrite (R-B27-3)

The B26 file is restructured:
- Per-schema happy non-sentinel (5 named tests, was 4).
- Per-schema happy sentinel (5 named tests, was 4).
- Per-schema mixed-reject parametrized (1 named test,
  parametrized 30 ways: 5 schemas x 6 variants).
- Per-schema reject-pair: skipped-with-metrics (5 named,
  was 4) + no-skipped-with-None-metrics (5 named, was 4).
- Helper guard (1 named, unchanged).
- B17 backstop (1 named, unchanged).

Named test count: 5 + 5 + 1 + 5 + 5 + 1 + 1 = 23. Collected:
23 + 29 parametrize extras (30 - 1) = 52.

Baseline B26 file: 26 named, 26 collected. Delta: -3 named,
+26 collected = +23 net.

### B27.5.2 Expected test delta

Baseline (post-B26): 1033 tests.
- Existing B26 file: 26 -> 23 named (-3 net from
  parametrize consolidation).
- Collected: 26 -> 52 (+26 from 30-way parametrize - 4
  old mixed-reject tests).
- Total: 1033 - 4 + 30 = 1059 collected.

Net new logical assertions: 22 (4 new variants x 5 schemas
+ 2 new EnsembleLift reject-pair tests).

## B27.6 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B27-Risk-1 | The new EnsembleLiftRollupRow CI-sentinel validator rejects an existing test fixture. | Low | The B26 fixture-audit already covered the b22 schema-default tests which include the EnsembleLift construction. Other EnsembleLift fixtures in B17/B19/B20/B23 byte-pin tests construct non-sentinel rows with populated metrics. Live verification via the full suite is the gate. |
| R-B27-Risk-2 | Composing two `@model_validator(mode="after")` on the same class causes ordering issues. | Low | Pydantic v2 documents that mode="after" validators run in declaration order. The two are independent (different field sets); ordering does not affect correctness. |
| R-B27-Risk-3 | Test rewrite drops coverage of the original 8 mixed-reject tests. | Low | The 30-way parametrize SUPERSETS the original 8 (which were mixed-1 + mixed-2 per schema). Coverage strictly increases. |
| R-B27-Risk-4 | Header correction in B22 design doc is a historical edit. | None | Documentation accuracy only; no behavior change. |

## Deferred

- **D-B27.1**: extend the mixed-reject parametrize to cover
  ALL invalid (metric_*, bootstrap_skipped_reason) combos
  including the cross-field rejects (#17-#24). Would
  collapse all reject coverage into a single parametrized
  test with 5 schemas x N variants. v1 keeps the
  reject-pair tests as named tests because the discriminator
  `match=` regexes differ by branch.
