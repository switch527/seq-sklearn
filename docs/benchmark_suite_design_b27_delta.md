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
schemas' `_validate_ci_sentinel_consistency` (B26 / D-B23.2)
and guards ONLY the `primary_metric_*` triple. The
`oracle_metric_*` triple is a separate partially-settable
surface unique to EnsembleLiftRollupRow (controlled by
`n_oracle_cells_paired` rather than `bootstrap_skipped_reason`);
v1 of B27 keeps it out of scope and captures it as D-B27.2.

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

**Backward-compat audit** (arch-R1-C1 + arch-R1-N1 +
arch-R2-C1 closure): per-site enumeration of all 18
EnsembleLiftRollupRow construction sites in
`tests/benchmarks/`:

| Site | Pattern | Validator compat |
|---|---|---|
| `test_b17_byte_identity_pins.py:332` | non-sentinel | COMPATIBLE |
| `test_b19_n_pair_grid.py:268` (factory) | non-sentinel | COMPATIBLE |
| `test_b19_n_pair_grid.py:328` | non-sentinel | COMPATIBLE |
| `test_b19_n_pair_grid.py:363` | sentinel (all-None + reason) | COMPATIBLE |
| `test_b19_n_pair_grid.py:397` | `pytest.raises` on `n_pair_grid=-1` | COMPATIBLE (`Field(ge=0)` fires before model_validator) |
| `test_b20_oracle_delta_ci.py:248` (factory) | non-sentinel | COMPATIBLE |
| `test_b20_oracle_delta_ci.py:813` | non-sentinel | COMPATIBLE |
| `test_b20_oracle_delta_ci.py:836` | non-sentinel | COMPATIBLE |
| `test_b21_bca_ci.py:638` | non-sentinel | COMPATIBLE |
| `test_b21_bca_ci.py:815` | sentinel (all-None + `"test_fixture"` reason from B26 repair to `common_base`) | COMPATIBLE |
| `test_b22_per_fold_cis.py:264` (zero-cell sentinel via `common`) | sentinel | COMPATIBLE post-B26 fixture repair |
| `test_b22_per_fold_cis.py:1235` | non-sentinel | COMPATIBLE |
| `test_b23_b20_nits_bundle.py:84` (factory) | non-sentinel | COMPATIBLE |
| `test_b24_bca_health_footnote.py:306` (factory) | non-sentinel | COMPATIBLE |
| `test_b25_per_fold_cis_footnote.py:330` (factory) | non-sentinel | COMPATIBLE |
| `test_bootstrap_manifest.py:532` | non-sentinel | COMPATIBLE |
| `test_ensemble_lift_report_b16.py:109` (factory) | non-sentinel default | COMPATIBLE |
| `test_ensemble_lift_report_b16.py:561` (the OR-guard mutation pin) | **invalid: populated metrics + `bootstrap_skipped_reason="all_cells_skipped_in_manifest"`** | **NOT COMPATIBLE; needs `model_construct` escape hatch** |

The single problem site is the renderer OR-guard mutation
pin at `test_ensemble_lift_report_b16.py:561`. That test
deliberately constructs a forbidden row shape (populated
metrics + sentinel reason) to verify the renderer's
defensive OR guard at
`_render_complete_table_with_ci:148` correctly falls
through to `(no CI)`. Under the new validator the row
cannot be constructed via the normal `EnsembleLiftRollupRow(...)`
call.

**Resolution**: rewrite the test to use pydantic v2's
`EnsembleLiftRollupRow.model_construct(...)` documented
validation-bypass. `model_construct` skips ALL validators
(both `_validate_row_count_invariants` and the new
`_validate_ci_sentinel_consistency`). The test's intent is
to verify renderer behavior on a known-bad row shape that
SHOULDN'T exist in production but COULD be produced by a
hypothetical buggy aggregator; `model_construct` preserves
that intent while letting the validator work for normal
construction paths.

B27.4 step 4 (fixture audit) verifies live during build.

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
4. **Fixture repair** (arch-R1-C1 closure): replace the
   `_make_rollup_row(...)` call at
   `tests/benchmarks/test_ensemble_lift_report_b16.py:561-569`
   with `EnsembleLiftRollupRow.model_construct(...)` for
   the same field values (the helper at `:76-131` returns
   `EnsembleLiftRollupRow(...)`; the test must bypass
   validators to construct the deliberately-invalid row).
   No other repairs needed per the per-site enumeration
   in B27.2 above.
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
- Existing B26 file: 26 -> 23 named (qa-R1-I1 closure
  decomposes the -3 net as: -8 old mixed-reject named
  tests (mixed-A + mixed-B x 4 schemas) + 4 new
  EnsembleLift named tests (happy non-sentinel + happy
  sentinel + 2 reject-pair) + 1 new mixed-reject
  parametrized test = -8 + 4 + 1 = -3 net named).
- Collected delta: -8 old mixed-reject + 4 new EnsembleLift
  + 30 mixed parametrize cases = +26 net collected.
- Total: 1033 + 26 = 1059 collected.

Net new logical assertions: 22 (4 new variants x 5 schemas
+ 2 new EnsembleLift reject-pair tests).

## B27.6 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B27-Risk-1 | The new EnsembleLiftRollupRow CI-sentinel validator rejects an existing test fixture. | Medium-confirmed | R1 design swarm identified one site: `test_ensemble_lift_report_b16.py:561-569` (renderer OR-guard mutation pin) constructs an `EnsembleLiftRollupRow` with populated metrics AND `bootstrap_skipped_reason="all_cells_skipped_in_manifest"`. The repair is prescribed in B27.4 step 4: rewrite as `EnsembleLiftRollupRow.model_construct(...)` to bypass the new validator. All 17 other EnsembleLift sites are CI-sentinel-compatible per the per-site enumeration in B27.2. |
| R-B27-Risk-2 | Composing two `@model_validator(mode="after")` on the same class causes ordering issues. | Low | Pydantic v2 documents that mode="after" validators run in declaration order. The two are independent (different field sets); ordering does not affect correctness. |
| R-B27-Risk-3 | Test rewrite drops coverage of the original 8 mixed-reject tests. | Low | The 30-way parametrize SUPERSETS the original 8 (which were mixed-1 + mixed-2 per schema). Coverage strictly increases. |
| R-B27-Risk-4 | Header correction in B22 design doc is a historical edit. | None | Documentation accuracy only; no behavior change. |

## Addressed

R1 design swarm on commit `f617ada`: architecture-reviewer
(1C / 1I / 1N REQUEST_CHANGES), qa-test-coverage (0C / 2I /
1N APPROVE), style-reviewer (0C / 0I / 0N APPROVE).
Deduplicated total: 1 CRITICAL, 3 IMPROVEMENT, 2 NITPICK.
Closures:

- **arch-R1-C1** (`test_ensemble_lift_report_b16.py:561-569`
  constructs invalid row that the new validator would
  reject): R-B27-2 backward-compat audit rewritten with the
  per-site enumeration; R-B27-Risk-1 elevated to
  "Medium-confirmed" with the b16 site named; B27.4 step 4
  prescribes the `EnsembleLiftRollupRow.model_construct(...)`
  rewrite to bypass validators for the deliberate
  invalid-row mutation pin.
- **arch-R1-I1 + qa-R1-I1** (test-count delta math
  attribution wrong: "-4 old mixed-reject" should be "-8
  old + 4 new EnsembleLift named"): B27.5.2 rewritten with
  the correct decomposition (-8 + 4 + 1 = -3 net named;
  -8 + 4 + 30 = +26 net collected).
- **qa-R1-I2** (oracle_metric_* triple on
  EnsembleLiftRollupRow not covered by the new validator):
  R-B27-2 scope clarified to explicitly state oracle_metric_*
  is out of scope and controlled by n_oracle_cells_paired
  rather than bootstrap_skipped_reason. Added deferral
  D-B27.2 for the oracle triple's separate CI-sentinel
  surface.
- **arch-R1-N1** (backward-compat audit names only 3
  fixture families): subsumed into the arch-R1-C1 closure
  above; the per-site enumeration table now covers all 12
  EnsembleLift construction sites.
- **qa-R1-N1** (R-B27-Risk-3 superset claim correct as
  written): NOT changed; reviewer confirmed the claim
  holds.

Test count after R1 closures: 23 named (no change from
R1 draft; the delta arithmetic was clarified, not modified);
1059 collected.

### R2 design swarm closure

R2 confirming swarm on commit `7f08728`: architecture-
reviewer (1C / 0I / 1N REQUEST_CHANGES), qa-test-coverage
(0C / 0I / 1N APPROVE), style-reviewer (0C / 0I / 0N
APPROVE). Deduplicated total: 1 CRITICAL, 0 IMPROVEMENT, 2
NITPICK. Closures:

- **arch-R2-C1** (per-site table claimed "all 12 sites"
  but actually omitted 6 sites under the full grep of
  EnsembleLiftRollupRow construction calls): table
  expanded to all 18 sites with the 6 additional rows
  (test_b19_n_pair_grid at :328, :363, :397;
  test_b20_oracle_delta_ci at :813, :836;
  test_b21_bca_ci at :815). The compatibility conclusion
  is unchanged: only `test_ensemble_lift_report_b16.py:561`
  needs the `model_construct` rewrite.
- **arch-R2-N1 + qa-R2-N1** (`bootstrap_ensemble_lift.py`
  path prefix missing in D-B27.2 citation): full path
  prefix added (`benchmarks/report/bootstrap_ensemble_lift.py:320-321`).

Test count after R2 closures: 23 named / 1059 collected
(unchanged from R1; all R2 closures were doc accuracy
only).

### R1 + R2 build-swarm closure

R1 build swarm on commit `ba6d20c`: code-reviewer (0C / 0I /
1N APPROVE), qa-test-coverage (0C / 0I / 1N APPROVE),
architecture-reviewer (0C / 2I / 2N APPROVE), style-reviewer
(0C / 0I / 0N APPROVE). The single recurring NITPICK is the
oracle-sentinel-shape note: `_ENSEMBLE_LIFT_BASE` supplies
non-None oracle metrics so the sentinel test exercises a
row shape that real aggregators do not emit. Deferred to
D-B27.2 (oracle CI-sentinel invariant); harmless under v1.

R2 confirming build swarm on commit `ba6d20c`: code-reviewer
(0C / 0I / 0N APPROVE), qa-test-coverage (0C / 0I / 1N
APPROVE; D-B27.2 carryover), architecture-reviewer (0C / 0I
/ 0N APPROVE), style-reviewer (0C / 0I / 0N APPROVE). Build
consensus reached: zero CRITICAL, zero IMPROVEMENT, 1
NITPICK (deferred to D-B27.2).

## Deferred

- **D-B27.2**: extend the CI-sentinel
  `@model_validator(mode="after")` on
  `EnsembleLiftRollupRow` to ALSO guard the
  `oracle_metric_*` triple. v1 of B27 scopes the new
  validator to the `primary_metric_*` triple (matches the
  4 other RollupRow schemas). The oracle triple is
  controlled by `n_oracle_cells_paired` rather than
  `bootstrap_skipped_reason`: a row with
  `n_oracle_cells_paired == 0` has all oracle metrics None,
  and a row with `n_oracle_cells_paired > 0` has all oracle
  metrics set (per
  `benchmarks/report/bootstrap_ensemble_lift.py:320-321`).
  A future invariant of the form `oracle_metric_* all-None
  iff n_oracle_cells_paired == 0` would close this gap.
- **D-B27.1**: extend the mixed-reject parametrize to cover
  ALL invalid (metric_*, bootstrap_skipped_reason) combos
  including the cross-field rejects (#17-#24). Would
  collapse all reject coverage into a single parametrized
  test with 5 schemas x N variants. v1 keeps the
  reject-pair tests as named tests because the discriminator
  `match=` regexes differ by branch.
