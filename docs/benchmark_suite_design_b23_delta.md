# B23 design delta: B20 NITs bundle (D-B20.1 + D-B20.2 + D-B20.3)

**Scope**: B23 bundles three small B20 deferrals into a single
phase to close the per-deferral overhead. Each item was below
the size threshold for a standalone delta but together they
form a coherent "tighten the B20 contract" pass:

- **D-B20.1**: explicit footnote table block for the oracle CI
  partial-coverage asterisk in the B16 ensemble-lift CI
  renderer.
- **D-B20.2** (carries arch-R1-build-N2 from B20): tighten the
  oracle vs main bootstrap raise-message discriminator from
  the prose suffix `"on the oracle delta"` to a stable token
  the test `match=` clauses can pin against without prose
  drift.
- **D-B20.3** (carries qa-R1-build-N1 from B20): add a
  pydantic model_validator on `EnsembleLiftRollupRow`
  enforcing the cross-field invariant
  `n_oracle_cells_paired <= n_cells_paired`. v1 of B16
  guarantees this structurally via the aggregator's list-
  comprehension; the validator becomes defense-in-depth for a
  future refactor that drops the structural guarantee.

None of the three items require schema additivity (the
validator constrains an existing field pair), so this phase
ships zero new schema fields. The renderer change is additive
on the rendered markdown only (parquet shard unchanged).

## Requirements

The grading rubric for every reviewer finding traces back to
one of these.

- **R-B23-1** (D-B20.1): extend
  `_render_complete_table_with_ci` in
  `benchmarks/report/ensemble_lift.py` to emit a footnote
  table block when ANY row in the rendered output has
  `n_oracle_cells_paired < n_pair_grid AND n_pair_grid > 0`
  (the oracle partial-coverage flag). The footnote table
  block parallels the existing partial-coverage footnote
  block (already used by the main Δloss column) and lists
  each affected dataset with the oracle counts: `dataset`,
  `n_oracle_cells_paired`, `n_pair_grid`, and the implied
  `n_missing_oracle_cells = n_pair_grid - n_oracle_cells_paired`.
  When no row has the oracle partial flag set, NO footnote
  block is emitted (silent on the happy path).
- **R-B23-2** (D-B20.2): rewrite the two oracle raise sites
  in `benchmarks/report/bootstrap_ensemble_lift.py`
  (`_build_dataset_rollup`, the `np.isfinite` guard and the
  OOM gate) to use stable discriminator tokens that test
  `match=` clauses pin against. Specifically:
  - Replace `"on the oracle delta"` suffixes with prefixes
    that name the schema field: `"oracle_metric_mean:"` for the
    isfinite guard message and
    `"n_oracle_cells_paired:"` for the OOM gate message.
  - Update B20 tests #5 (`test_aggregator_oracle_oom_gate_raises`)
    and #6 (`test_aggregator_oracle_nan_delta_raises_via_stub`)
    to `match=r"n_oracle_cells_paired:"` and
    `match=r"oracle_metric_mean:"` respectively. The new
    `match=` tokens are stable schema-field names rather than
    prose.
- **R-B23-3** (D-B20.3 + arch-R1-I1 closure): add a
  pydantic `@model_validator` on `EnsembleLiftRollupRow`
  enforcing THREE cross-field invariants (all
  structurally true in the v1 aggregator + assumed by the
  v1 renderer at `ensemble_lift.py:166`):
  - `n_oracle_cells_paired <= n_cells_paired`
  - `n_oracle_cells_paired <= n_pair_grid`
  - `n_cells_paired <= n_pair_grid` (B19 peer invariant)
  The validator raises a pydantic `ValueError` (which
  pydantic wraps into `ValidationError`) with a
  deterministic message naming both fields. Sentinel rows
  are exempt because all three counts are 0 in the
  sentinel emit, satisfying the invariants trivially.
- **R-B23-4** No new schema fields. No new module-level
  constants. No new ExperimentSpec flags. The bundle is
  purely behavioral on existing surfaces.
- **R-B23-5** Backward-compat with pre-B23 parquet shards:
  old shards loaded post-B23 are validated through the new
  model_validator. R-B16.0 noted that bench-run shards are
  short-lived; any pre-B23 shard whose data satisfies the
  structural invariants of B16's v1 aggregator (which
  guarantees them) loads cleanly. A theoretical
  corrupt-shard scenario where the invariants fail produces
  a `ValidationError` at load time, which is the desired
  catch.

## B23.0 What the change actually adds

### B23.0.1 D-B20.1 renderer footnote

Pre-B23: when an oracle CI cell has the trailing asterisk
(partial coverage on the oracle bootstrap), the renderer
emits the asterisk but no per-dataset footnote table. A
reader sees the asterisk and must infer "fewer oracle cells
than expected" without knowing HOW MANY were missing.

Post-B23: when at least one row has
`n_oracle_cells_paired < n_pair_grid`, the renderer emits
ONE markdown block immediately after the complete-rows
table:

```markdown
### Oracle partial-coverage footnotes

| dataset | n_oracle_cells_paired | n_pair_grid | n_missing |
| --- | --- | --- | --- |
| ds_alpha | 3 | 4 | 1 |
| ds_beta | 2 | 4 | 2 |
```

The block is sorted by dataset name ascending. The
`n_missing` column is computed as
`n_pair_grid - n_oracle_cells_paired`. The block is silent
when no row has the oracle partial flag.

### B23.0.2 D-B20.2 raise-message discriminators

Pre-B23 `_build_dataset_rollup` raise sites:

```python
# isfinite guard
raise RawRollupError(
    f"aggregate_bootstrap_ensemble_lift_rollup: dataset="
    f"{dataset_name!r} has a paired cell with a non-finite "
    "oracle delta; the upstream predictions shard is corrupt"
)

# OOM gate
raise RawRollupError(
    f"aggregate_bootstrap_ensemble_lift_rollup: dataset="
    f"{dataset_name!r} with n_oracle_cells_paired="
    f"{n_oracle_cells_paired} * n_resamples={n_resamples} "
    f"exceeds the bootstrap-row-count ceiling "
    f"({_bootstrap_aggregate.BOOTSTRAP_ROW_COUNT_CEILING}) "
    "on the oracle delta"
)
```

Post-B23 messages (with stable schema-name tokens):

```python
# isfinite guard
raise RawRollupError(
    f"aggregate_bootstrap_ensemble_lift_rollup: oracle_metric_mean: "
    f"dataset={dataset_name!r} has a paired cell with a "
    "non-finite oracle delta; the upstream predictions shard "
    "is corrupt"
)

# OOM gate
raise RawRollupError(
    f"aggregate_bootstrap_ensemble_lift_rollup: n_oracle_cells_paired: "
    f"dataset={dataset_name!r} with n_oracle_cells_paired="
    f"{n_oracle_cells_paired} * n_resamples={n_resamples} "
    f"exceeds the bootstrap-row-count ceiling "
    f"({_bootstrap_aggregate.BOOTSTRAP_ROW_COUNT_CEILING})"
)
```

The new prefixes are stable schema field names. Test
`match=` clauses now grep against
`r"oracle_metric_mean:"` and `r"n_oracle_cells_paired:"`
instead of the prose `r"oracle delta"`.

### B23.0.3 D-B20.3 cross-field validator

Add to `EnsembleLiftRollupRow` in
`benchmarks/bootstrap_manifest.py`:

```python
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EnsembleLiftRollupRow(BaseModel):
    ...
    n_oracle_cells_paired: int = Field(ge=0)
    ...

    @model_validator(mode="after")
    def _validate_row_count_invariants(self) -> "EnsembleLiftRollupRow":
        # B23 / D-B20.3 + arch-R1-I1 closure: three structural
        # invariants from the v1 B16 aggregator (the oracle
        # bootstrap operates on a subset of the paired cells; the
        # paired cells are a subset of the intersection grid).
        # The renderer at ensemble_lift.py:166 already assumes
        # `n_cells_paired <= n_pair_grid` (the partial-coverage
        # flag computation).
        if self.n_oracle_cells_paired > self.n_cells_paired:
            raise ValueError(
                f"n_oracle_cells_paired ({self.n_oracle_cells_paired}) "
                f"exceeds n_cells_paired ({self.n_cells_paired})"
            )
        if self.n_oracle_cells_paired > self.n_pair_grid:
            raise ValueError(
                f"n_oracle_cells_paired ({self.n_oracle_cells_paired}) "
                f"exceeds n_pair_grid ({self.n_pair_grid})"
            )
        if self.n_cells_paired > self.n_pair_grid:
            raise ValueError(
                f"n_cells_paired ({self.n_cells_paired}) "
                f"exceeds n_pair_grid ({self.n_pair_grid})"
            )
        return self
```

Sentinel rows trivially satisfy all three invariants
because their `n_cells_paired = n_pair_grid =
n_oracle_cells_paired = 0`.

## B23.1 Renderer changes

In `benchmarks/report/ensemble_lift.py`:

```python
def _render_oracle_partial_coverage_footnote(
    affected: list[tuple[str, int, int, int]],
) -> str:
    """B23 / D-B20.1: markdown footnote block for the oracle
    CI partial-coverage asterisk. Pure renderer; caller
    pre-filters the affected rows (arch-R1-I3 closure: filter
    at the call site, mirrors the existing
    `_render_partial_coverage_footnote` pattern)."""
    if not affected:
        return ""
    affected_sorted = sorted(affected, key=lambda t: t[0])
    lines = [
        "### Oracle partial-coverage footnotes",
        "",
        "| dataset | n_oracle_cells_paired | n_pair_grid | n_missing |",
        "| --- | --- | --- | --- |",
    ]
    for dataset_name, n_oracle, n_grid, n_missing in affected_sorted:
        lines.append(
            f"| {dataset_name} | {n_oracle} | {n_grid} | {n_missing} |"
        )
    lines.append("")
    return "\n".join(lines)
```

The caller in `_render_with_ci` filters affected rows
before calling the helper:

```python
# Inside _render_with_ci, after the complete-rows table:
affected_oracle: list[tuple[str, int, int, int]] = []
for row in complete:
    rollup_row = rollup_index.get(row.dataset_name)
    if rollup_row is None or rollup_row.bootstrap_skipped_reason is not None:
        continue
    if (
        rollup_row.n_oracle_cells_paired < rollup_row.n_pair_grid
        and rollup_row.n_pair_grid > 0
    ):
        affected_oracle.append(
            (
                row.dataset_name,
                rollup_row.n_oracle_cells_paired,
                rollup_row.n_pair_grid,
                rollup_row.n_pair_grid - rollup_row.n_oracle_cells_paired,
            )
        )
table_parts.append(_render_oracle_partial_coverage_footnote(affected_oracle))
```

The function returns `""` on the happy path so existing
reports remain byte-equivalent when no oracle partial flag
fires.

## B23.2 Aggregator changes

The two oracle raise sites in `_build_dataset_rollup`
(`bootstrap_ensemble_lift.py`) get their prose suffix
replaced with stable schema-field prefixes per R-B23-2. No
other aggregator behavior changes; the raise classes, the
gates' boolean conditions, and the return shape are all
unchanged.

## B23.3 Schema changes

Add the `@model_validator(mode="after")` to
`EnsembleLiftRollupRow`. No new fields. The validator runs
on every construction including parquet load; any pre-B23
shard whose data satisfies the v1 aggregator's structural
invariants loads cleanly.

## B23.4 Existing tests touched

- **`tests/benchmarks/test_b20_oracle_delta_ci.py`**:
  - Test #5 (`test_aggregator_oracle_oom_gate_raises`):
    update the `pytest.raises(RawRollupError, match=...)`
    clause from `match=r"oracle delta"` to
    `match=r"n_oracle_cells_paired:"`.
  - Test #6 (`test_aggregator_oracle_nan_delta_raises_via_stub`):
    update from `match=r"non-finite oracle delta"` to
    `match=r"oracle_metric_mean:"` (arch-R2-I2 closure:
    the new prefix is the only stable discriminator; the
    prose body `"non-finite oracle delta"` is intentionally
    NOT anchored because D-B20.2 set out to eliminate
    prose-drift coupling).
  - 12 fixture sites construct `EnsembleLiftRollupRow`
    directly (in `test_b17_byte_identity_pins.py`,
    `test_b19_n_pair_grid.py`, `test_b20_oracle_delta_ci.py`,
    `test_b21_bca_ci.py`, `test_b22_per_fold_cis.py`,
    `test_bootstrap_manifest.py`,
    `test_ensemble_lift_report_b16.py`). Eleven satisfy
    the new validator. ONE site requires repair
    (arch-R1-C1 + qa-R1-C1 closure):
    `tests/benchmarks/test_b17_byte_identity_pins.py:339-350`
    sets `n_cells_paired=1, n_oracle_cells_paired=2,
    n_pair_grid=2` so the main Δloss CI cell's
    mandatory-asterisk regex fires on `n_cells_paired <
    n_pair_grid`. The `n_oracle_cells_paired=2 >
    n_cells_paired=1` combination violates the new
    validator. Fix: change to `n_oracle_cells_paired=1`
    (matches the new floor `<= n_cells_paired=1`). The
    B17 byte-pin regex targets the main Δloss column only
    and is indifferent to the oracle cell content; the
    oracle column will have its own partial asterisk
    (because `n_oracle_cells_paired (1) < n_pair_grid (2)`)
    but the regex still matches.

## B23.5 NEW B23 tests

`tests/benchmarks/test_b23_b20_nits_bundle.py` (NEW; 13
tests).

**Renderer footnote (D-B20.1)**:

1. `test_renderer_oracle_partial_footnote_fires_when_any_row_partial`:
   construct a result + rollup with one dataset whose
   `n_oracle_cells_paired=3, n_pair_grid=4`. Assert the
   rendered markdown contains the header `"Oracle
   partial-coverage footnotes"` AND the row
   `"| ds_one | 3 | 4 | 1 |"`.
2. `test_renderer_oracle_partial_footnote_silent_on_happy_path`:
   all rows have `n_oracle_cells_paired == n_pair_grid`.
   Assert the rendered markdown does NOT contain
   `"Oracle partial-coverage footnotes"`.
3. `test_renderer_oracle_partial_footnote_skips_sentinel_rows`
   (qa-R1-I4 closure: the actual short-circuit mechanism is
   `bootstrap_skipped_reason is not None`, NOT `n_pair_grid
   == 0`; the `all_cells_skipped_in_manifest` sentinel can
   carry a non-zero `n_pair_grid`. Parametrize over all
   three sentinel reasons):
   for each of `("no_gbm_predictions", "no_seq_predictions",
   "all_cells_skipped_in_manifest")`, construct a fixture
   with one sentinel row (with `n_pair_grid=4` to defeat the
   wrong "zero-grid" rationale) AND one happy-path row with
   no oracle asterisk. Assert the rendered markdown does NOT
   contain `"Oracle partial-coverage footnotes"` in any of
   the three sentinel cases.
4. `test_renderer_oracle_partial_footnote_sorts_by_dataset_name`:
   two affected rows with dataset names `"z_alpha"` and
   `"a_beta"`. Assert the footnote table lists `"a_beta"`
   before `"z_alpha"`.
5. `test_renderer_oracle_partial_footnote_n_missing_column`:
   one row with `n_oracle_cells_paired=2, n_pair_grid=5`.
   Assert the `n_missing` column reads `"3"`.

**Raise-message discriminators (D-B20.2)**:

6. `test_aggregator_oracle_nan_raise_message_contains_oracle_metric_token`:
   re-uses the B20 test #6 setup. Assert
   `pytest.raises(RawRollupError, match=r"oracle_metric_mean:")`.
7. `test_aggregator_oracle_oom_raise_message_contains_n_oracle_cells_paired_token`:
   re-uses the B20 test #5 setup. Assert
   `pytest.raises(RawRollupError, match=r"n_oracle_cells_paired:")`.

**Cross-field validator (D-B20.3)**:

8. `test_ensemble_lift_validator_rejects_n_oracle_cells_paired_exceeds_n_cells_paired`:
   construct a row with
   `n_cells_paired=4, n_oracle_cells_paired=5,
   n_pair_grid=10`. Assert
   `pytest.raises(ValidationError, match=r"n_oracle_cells_paired.*exceeds n_cells_paired")`.
9. `test_ensemble_lift_validator_rejects_n_oracle_cells_paired_exceeds_n_pair_grid`:
   construct a row with
   `n_cells_paired=10, n_oracle_cells_paired=5,
   n_pair_grid=4`. Assert
   `pytest.raises(ValidationError, match=r"n_oracle_cells_paired.*exceeds n_pair_grid")`.
10. `test_ensemble_lift_validator_accepts_sentinel_row_zero_counts`:
    construct a sentinel row with
    `n_cells_paired=0, n_oracle_cells_paired=0,
    n_pair_grid=0`. Assert the row constructs successfully
    (sentinel trivially satisfies all three invariants).

**Additional R1 closure tests (qa-R1-I1 + qa-R1-I3 +
arch-R1-I1)**:

11. `test_renderer_oracle_partial_footnote_mixed_partial_and_full_rows`
    (qa-R1-I1 closure): construct a fixture with three
    rows: one fully paired (`n_oracle == n_pair_grid`),
    one partially paired (`n_oracle < n_pair_grid`), one
    sentinel. Assert the footnote table contains exactly
    one row (the partially-paired dataset), NOT the
    fully-paired or sentinel rows. Kills a filter
    predicate mutation that would over-include the
    fully-paired row.
12. `test_ensemble_lift_validator_accepts_positive_equality_boundary`
    (qa-R1-I3 closure): construct a row with
    `n_cells_paired=5, n_oracle_cells_paired=5,
    n_pair_grid=5`. Assert construction succeeds. Kills a
    mutation that changes `>` to `>=` in any of the three
    validator clauses.
13. `test_ensemble_lift_validator_rejects_n_cells_paired_exceeds_n_pair_grid`
    (arch-R1-I1 closure): construct a row with
    `n_cells_paired=5, n_oracle_cells_paired=0,
    n_pair_grid=4`. Assert
    `pytest.raises(ValidationError, match=r"n_cells_paired.*exceeds n_pair_grid")`.
    Pins the third invariant added by arch-R1-I1.

Expected test delta after the build:
- Existing tests: 930 → 930 (B20 tests #5 and #6 are
  updated in place; B17 byte-pin fixture's
  `n_oracle_cells_paired` value changes from 2 to 1; no
  count change).
- B23-new: 13 tests.
- Total: 930 + 13 = 943 expected post-refactor.

## B23.6 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B23-Risk-1 | The new `@model_validator` rejects a pre-B23 parquet shard whose data violates the structural invariants. | Low | Bench-run shards are short-lived (B17 R-B17-3 precedent). The B16 aggregator's structural guarantee from v1 means no production shard CAN violate the invariants; the validator only catches future-corrupt data. |
| R-B23-Risk-2 | The renderer footnote block adds markdown to existing reports, changing the byte content of any report that previously had an oracle partial-coverage asterisk. | Low | After the B23.4 fixture mutation the B17 ensemble-lift byte-pin fixture has `n_oracle_cells_paired=1, n_pair_grid=2`, so the new footnote block DOES fire on the byte-pin reports. The B17 byte-pin test (`tests/benchmarks/test_b17_byte_identity_pins.py`) tolerates the addition because its assertions use `_CI_CELL_RE.search(md)` and absent-substring checks, NOT full-document byte equality, so the footnote text appearing in the rendered markdown does not invalidate the pin. Test #2 in B23 pins the silent-on-happy-path contract for the renderer itself; the R1 build-swarm CRITICAL closure for test #3 strengthens the sentinel-suppression coverage. |
| R-B23-Risk-3 | The raise-message rewrite breaks any downstream consumer that parses the message text. | Low | The B16 contract is to surface `RawRollupError` to the CLI wrapper which writes the message verbatim to a sentinel file. The sentinel file is read by humans, not parsed. No structured downstream consumer of the message text exists. |
| R-B23-Risk-4 | The `@model_validator` runs on every load, including the large per-fold round-trip tests from B22. | Low | The validator is O(1) per row (two integer comparisons). No measurable load-time impact. |

## B23.7 Implementation outline

1. **Validator**: add the `@model_validator(mode="after")`
   to `EnsembleLiftRollupRow` in
   `benchmarks/bootstrap_manifest.py` per R-B23-3.
2. **Raise messages**: rewrite the two oracle raise sites
   in `_build_dataset_rollup`
   (`benchmarks/report/bootstrap_ensemble_lift.py`) per
   R-B23-2.
3. **Renderer footnote**: add
   `_render_oracle_partial_coverage_footnote` helper in
   `benchmarks/report/ensemble_lift.py` and call it from
   `_render_with_ci` immediately after the complete-rows
   table per R-B23-1.
4. **Update existing tests**: B20 tests #5 and #6 get
   updated `match=` clauses; the B17 byte-pin fixture at
   `tests/benchmarks/test_b17_byte_identity_pins.py:339-350`
   gets `n_oracle_cells_paired` changed from 2 to 1
   (arch-R1-C1 + qa-R1-C1 closure).
5. **NEW tests**: add
   `tests/benchmarks/test_b23_b20_nits_bundle.py` with 13
   tests (10 design-named + 3 R1 closures: arch-R1-I1
   added #13, qa-R1-I1 added #11, qa-R1-I3 added #12).
6. **Verify**: ruff + pyright clean; 943 tests pass.

## Addressed

R1 design swarm: architecture-reviewer (1C / 4I / 2N
REQUEST_CHANGES), qa-test-coverage (1C / 3I / 1N
REQUEST_CHANGES), style-reviewer (0C / 0I / 0N APPROVE).
Deduplicated total: 1 CRITICAL (raised by both arch and
qa), 6 IMPROVEMENT, 3 NITPICK. Closures:

- **arch-R1-C1 + qa-R1-C1** (the B17 byte-pin fixture at
  `tests/benchmarks/test_b17_byte_identity_pins.py:339-350`
  sets `n_cells_paired=1, n_oracle_cells_paired=2,
  n_pair_grid=2`; the new validator rejects
  `n_oracle_cells_paired (2) > n_cells_paired (1)`. The
  B23.4 "no changes needed" audit was wrong): B23.4
  rewritten to enumerate 12 fixture sites and call out the
  B17 fixture as the ONE site requiring repair. Fix: change
  `n_oracle_cells_paired=1` (matches the new floor). The
  byte-pin regex matches the main Δloss column only and is
  indifferent to the oracle cell content.
- **arch-R1-I1** (R-B23-3 omits the peer
  `n_cells_paired <= n_pair_grid` invariant which is
  structurally guaranteed by the aggregator and already
  assumed by the renderer at `ensemble_lift.py:166`):
  R-B23-3 expanded to enforce THREE cross-field
  invariants. Added test #13 pinning the third invariant.
- **arch-R1-I2** (`oracle_metric_*:` is a glob, not a real
  schema field name; a grep for `oracle_metric_mean` will
  not find it, defeating the "rename safety" claim):
  raise-message prefix changed to `oracle_metric_mean:`
  throughout the design + test #6 `match=` clause.
- **arch-R1-I3** (helper folds sentinel filter inside;
  existing `_render_partial_coverage_footnote` pattern
  filters at the call site): B23.1 pseudocode rewritten to
  filter at the `_render_with_ci` call site and pass
  pre-filtered tuples to a pure-render helper.
- **arch-R1-I4** (test #3 narrative claims sentinel rows
  are skipped because `n_pair_grid=0`; the actual mechanism
  is `bootstrap_skipped_reason is not None`; the
  `all_cells_skipped_in_manifest` sentinel can carry a
  non-zero `n_pair_grid`): test #3 rewritten as a
  parametrize over all three sentinel reasons with
  `n_pair_grid=4` to defeat the wrong rationale.
- **qa-R1-I1** (no test for >2 affected rows with mixed
  partial/full coverage): added test #11 with three rows
  (one fully paired, one partially paired, one sentinel)
  asserting the footnote contains exactly the partially-
  paired row.
- **qa-R1-I3** (positive-equality boundary at validator
  not tested; `>` → `>=` mutation would survive): added
  test #12 constructing a row with `n_cells_paired=5,
  n_oracle_cells_paired=5, n_pair_grid=5` and asserting
  construction succeeds.
- **arch-R1-N1 + qa-R1-N1** (930 baseline + tests #6/#7
  redundancy): the 930 baseline was confirmed via live
  pytest on the post-B22 main tip; tests #6/#7 ARE the
  new behavioral pin for the prefix tokens (B20 tests #5
  / #6 still pin the body content). The two pin different
  things: B20 tests pin the body, B23 tests #6/#7 pin the
  new prefix.
- **arch-R1-N2** (cosmetic bullet formatting): NOT
  changed.
- **qa-R1-I2** (sentinel-reason parametrize over all 3):
  same as arch-R1-I4 closure; test #3 now covers all
  three.

Test count after R1 closures: 13 new tests (was 10;
arch-R1-I1 added #13, qa-R1-I1 added #11, qa-R1-I3 added
#12); total `930 + 13 = 943`.

### R2 confirming swarm closure

R2 confirming swarm on commit `dcf11eb`: architecture-
reviewer (1C / 2I / 1N REQUEST_CHANGES), qa-test-coverage
(1C / 0I / 0N REQUEST_CHANGES), style-reviewer (0C / 0I /
0N APPROVE). Deduplicated total: 1 CRITICAL (raised by
both arch and qa), 2 IMPROVEMENT, 1 NITPICK. Closures:

- **arch-R2-C1 + qa-R2-C1** (the R1 arch-R1-I2 closure
  changed the raise-message prefix to `oracle_metric_mean:`
  in the pseudocode AND in the arch-R1-I2 closure
  narrative, but FOUR `match=` regex sites still held the
  pre-closure `r"oracle_metric_\*:"` glob token; the
  `replace_all` edit caught only the standalone form
  `r"oracle_metric_*:"` and missed the regex-escaped
  `r"oracle_metric_\*:"` forms): updated all 4 sites
  (R-B23-2 at :59, B23.0.2 summary at :163, B23.4 at :297,
  B23.5 test #6 at :364) to read `r"oracle_metric_mean:"`.
- **arch-R2-I1** (B23.5 section header still said "NEW; 10
  tests" after R1 closures grew the count to 13):
  updated to "NEW; 13 tests".
- **arch-R2-I2** (test #6 `match=` clause still anchored
  on the prose body `"non-finite oracle delta"`, re-
  introducing the prose-drift coupling that D-B20.2
  explicitly rejects): dropped the body anchor; test #6
  now pins ONLY the prefix token
  `match=r"oracle_metric_mean:"`. The body content is no
  longer load-bearing for the discriminator pin.
- **arch-R2-N1** (section header lists three closure tags
  in the order I1, I3, arch-I1; tests #11/#12/#13 align):
  NOT changed; the ordering already matches.

Test count after R2 closures: 13 new tests; total
`930 + 13 = 943` (unchanged; R2 was discriminator-token
cleanup, no new tests).

### R1 build-swarm closure

R1 build swarm on commit `745fcf3`: code-reviewer
(0C / 1I / 2N APPROVE), qa-test-coverage (1C / 2I / 2N
REQUEST_CHANGES), architecture-reviewer (0C / 3I / 2N
APPROVE), style-reviewer (0C / 0I / 0N APPROVE).
Deduplicated total: 1 CRITICAL, 5 IMPROVEMENT, 6 NITPICK
(code-I1 dedupes into build-R1-C1).
Closures:

- **build-R1-C1 + code-I1** (test #3's
  `all_cells_skipped_in_manifest` arm uses `n_pair_grid=0`
  despite the docstring claiming `n_pair_grid=4`; the
  filter predicate `n_oracle_cells_paired < n_pair_grid
  AND n_pair_grid > 0` evaluates false on `0 < 0`, so the
  test passes even if the `bootstrap_skipped_reason is
  not None` guard at `ensemble_lift.py:436` were deleted):
  test #3 parametrized fixture now sets
  `n_cells_paired=2, n_pair_grid=4,
  n_oracle_cells_paired=2` for all three sentinel reasons;
  the trigger predicate now fires; the test is a real pin
  on the reason-guard for every sentinel arm.
- **qa-I2** (no test pins the
  `n_oracle_cells_paired=0, n_pair_grid>0` non-sentinel
  case): added test #5b
  `test_renderer_oracle_partial_footnote_fires_when_oracle_cells_zero_and_grid_nonzero`
  constructing a row with
  `n_cells_paired=4, n_pair_grid=4,
  n_oracle_cells_paired=0` and asserting the footnote
  fires with `n_missing=4`. Pins the design intent that 0
  of N oracle cells is treated as partial coverage, not
  as "no oracle" suppression.
- **arch-I1** (R-B23-Risk-2 prose was stale post B23.4
  fixture mutation): risk text rewritten to state the
  actual safety rationale (B17 byte-pin assertions use
  `_CI_CELL_RE.search(md)` + absent-substring checks, not
  full-document byte equality, so the new footnote block
  does not invalidate the pin).
- **arch-I2** (D-B23.2 example clause self-referenced
  `EnsembleLiftRollupRow`): deferral rewritten to scope to
  the other 4 RollupRow schemas with peer-invariant
  candidates (`n_cells_evaluated <= n_seeds * n_folds`,
  etc.) and to note `n_pair_grid` is exclusive to
  `EnsembleLiftRollupRow`.
- **arch-I3** (closure comments at
  `bootstrap_ensemble_lift.py:329, :343` named the
  discriminator field but not WHY it names the destination
  column for a defect on the input): both comments
  extended with one sentence noting the prefix names the
  destination rollup column so a future schema rename
  propagates via grep.
- **code-N1** (sentinel-row comment at
  `bootstrap_manifest.py:657-659` said "all counts == 0",
  factually wrong for `all_cells_skipped_in_manifest`
  sentinels which can carry non-zero `n_pair_grid`):
  comment rewritten to state `n_cells_paired` and
  `n_oracle_cells_paired` are 0 while `n_pair_grid` may
  be non-zero; the invariants still hold trivially.
- **qa-I3** (`_render_oracle_partial_coverage_footnote`'s
  empty-list early-return is unreachable through the
  public API): NOT changed. The internal guard mirrors
  the shape of other renderer helpers in the same file;
  removing it would create asymmetry without behavior
  change. Added to Deferred as D-B23.3.
- **code-N2** (same dead-guard concern as qa-I3): NOT
  changed; same rationale.
- **qa-N1** (test #7 docstring vs monkeypatch seam): NOT
  changed; cosmetic.
- **qa-N2** (defensive `n_missing > n_oracle_cells_paired`
  parametrize case): NOT changed; purely defensive.
- **arch-N1** (test docstring wording `4*2 <= 100` vs
  `8 <= 100`): NOT changed; cosmetic.
- **arch-N2** (table caption ambiguity at design
  `:413`): NOT changed; the R1 closure paragraph at
  `:507-509` already disambiguates.

Test count after R1 build-swarm closures: 14 new tests
(was 13; build-R1-qa-I2 added test #5b); collected count
is 16 (test #3 parametrized 3 ways). Total
`930 + 14 = 944` named; collected `930 + 16 = 946`.

## Deferred

- **D-B23.1**: extend the oracle partial-coverage footnote
  block to also surface the per-row `ci_method` and
  `bootstrap_ci_fallback_reason` (B21 / D-B21.1 deferral
  intersection). v1 keeps the footnote scoped to oracle
  coverage counts; the BCa fallback surface is a separate
  audit channel.
- **D-B23.2**: extend cross-field `@model_validator`
  coverage to the other 4 RollupRow schemas (B5 raw-loss,
  B6 holdout-loss, B7 hpo-uplift, B8 cv-uplift). Only
  `HPOUpliftRollupRow` carries an `n_folds` field, so the
  `n_cells_evaluated <= n_seeds * n_folds` form applies
  there; `RollupRow`, `PairwiseRollupRow`, and
  `TrainingTimeRollupRow` need invariants designed
  against their own field set (`n_cells_evaluated <=
  n_seeds` is one candidate). v1 of B23 is scoped to the
  B20-named D-B20.3 invariant on `EnsembleLiftRollupRow`
  only; `n_pair_grid` is exclusive to that schema, so the
  other 4 invariants are structurally distinct and warrant
  their own audit pass.
- **D-B23.3**: remove the internal `if not affected:
  return ""` guard at
  `benchmarks/report/ensemble_lift.py:490-491` since the
  caller at `:447` already gates on `if affected_oracle:`.
  The internal guard is dead code matching the shape of
  other renderer helpers in the same file; either both
  helpers should keep the guard or both should drop it.
  Deferred to a single sweep across `ensemble_lift.py`
  helpers rather than touching one site.
