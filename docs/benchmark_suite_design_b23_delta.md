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
    that name the schema field: `"oracle_metric_*:"` for the
    isfinite guard message and
    `"n_oracle_cells_paired:"` for the OOM gate message.
  - Update B20 tests #5 (`test_aggregator_oracle_oom_gate_raises`)
    and #6 (`test_aggregator_oracle_nan_delta_raises_via_stub`)
    to `match=r"n_oracle_cells_paired:"` and
    `match=r"oracle_metric_\*:"` respectively. The new
    `match=` tokens are stable schema-field names rather than
    prose.
- **R-B23-3** (D-B20.3): add a pydantic `@model_validator`
  on `EnsembleLiftRollupRow` enforcing
  `n_oracle_cells_paired <= n_cells_paired` AND
  `n_oracle_cells_paired <= n_pair_grid` (both
  cross-field invariants are structurally true in the v1
  aggregator). The validator raises a pydantic
  `ValueError` (which pydantic wraps into `ValidationError`)
  with a deterministic message naming both fields. Sentinel
  rows are exempt because `n_cells_paired=0` AND
  `n_oracle_cells_paired=0` AND `n_pair_grid=0` all satisfy
  the invariants trivially.
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
    f"aggregate_bootstrap_ensemble_lift_rollup: oracle_metric_*: "
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
`r"oracle_metric_\*:"` and `r"n_oracle_cells_paired:"`
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
    def _validate_oracle_cells_bounded(self) -> "EnsembleLiftRollupRow":
        # B23 / D-B20.3: structural invariant from the B16
        # aggregator's list-comprehension. The oracle bootstrap
        # operates on a subset of the paired cells; the count is
        # bounded by both n_cells_paired and n_pair_grid (the
        # intersection cardinality from B19).
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
        return self
```

Sentinel rows trivially satisfy both invariants because
their `n_cells_paired = n_pair_grid = n_oracle_cells_paired
= 0`.

## B23.1 Renderer changes

In `benchmarks/report/ensemble_lift.py`:

```python
def _render_oracle_partial_coverage_footnote(
    rows: list[PerDatasetLift],
    rollup_index: dict[str, EnsembleLiftRollupRow],
) -> str:
    """B23 / D-B20.1: markdown footnote block for the oracle
    CI partial-coverage asterisk."""
    affected: list[tuple[str, int, int, int]] = []
    for row in rows:
        rollup_row = rollup_index.get(row.dataset_name)
        if rollup_row is None or rollup_row.bootstrap_skipped_reason is not None:
            continue
        if (
            rollup_row.n_oracle_cells_paired < rollup_row.n_pair_grid
            and rollup_row.n_pair_grid > 0
        ):
            affected.append(
                (
                    row.dataset_name,
                    rollup_row.n_oracle_cells_paired,
                    rollup_row.n_pair_grid,
                    rollup_row.n_pair_grid - rollup_row.n_oracle_cells_paired,
                )
            )
    if not affected:
        return ""
    affected.sort(key=lambda t: t[0])
    lines = [
        "### Oracle partial-coverage footnotes",
        "",
        "| dataset | n_oracle_cells_paired | n_pair_grid | n_missing |",
        "| --- | --- | --- | --- |",
    ]
    for dataset_name, n_oracle, n_grid, n_missing in affected:
        lines.append(
            f"| {dataset_name} | {n_oracle} | {n_grid} | {n_missing} |"
        )
    lines.append("")
    return "\n".join(lines)
```

The new helper is called from `_render_with_ci` immediately
after the complete-rows table is appended. The function
returns `""` on the happy path so existing reports remain
byte-equivalent when no oracle partial flag fires.

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
    update from
    `match=r"non-finite oracle delta"` to
    `match=r"oracle_metric_\*:.*non-finite oracle delta"`
    (preserves the body content check + adds the new
    discriminator prefix).
  - 5 fixture sites that construct `EnsembleLiftRollupRow`
    directly (in `test_b17_byte_identity_pins.py`,
    `test_b19_n_pair_grid.py`, `test_b20_oracle_delta_ci.py`,
    `test_bootstrap_manifest.py`,
    `test_ensemble_lift_report_b16.py`): no changes needed.
    The existing default values satisfy
    `n_oracle_cells_paired <= n_cells_paired AND <=
    n_pair_grid` (e.g., `n_oracle_cells_paired=6,
    n_cells_paired=6, n_pair_grid=6` in the B20 factory).
  - Sites where a fixture sets
    `n_oracle_cells_paired > n_cells_paired` deliberately
    (e.g., to test the schema): NONE exist in the v1 suite
    by audit.

## B23.5 NEW B23 tests

`tests/benchmarks/test_b23_b20_nits_bundle.py` (NEW; 10
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
3. `test_renderer_oracle_partial_footnote_skips_sentinel_rows`:
   one sentinel row (`bootstrap_skipped_reason="no_gbm_predictions"`)
   AND one happy-path row with no oracle asterisk. Assert
   no footnote block. The sentinel row's
   `n_pair_grid=0` so it would not satisfy
   `n_pair_grid > 0`; the helper short-circuits.
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
   `pytest.raises(RawRollupError, match=r"oracle_metric_\*:")`.
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
    (sentinel trivially satisfies both invariants).

Expected test delta after the build:
- Existing tests: 930 → 930 (B20 tests #5 and #6 are
  updated in place; no count change).
- B23-new: 10 tests.
- Total: 930 + 10 = 940 expected post-refactor.

## B23.6 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B23-Risk-1 | The new `@model_validator` rejects a pre-B23 parquet shard whose data violates the structural invariants. | Low | Bench-run shards are short-lived (B17 R-B17-3 precedent). The B16 aggregator's structural guarantee from v1 means no production shard CAN violate the invariants; the validator only catches future-corrupt data. |
| R-B23-Risk-2 | The renderer footnote block adds markdown to existing reports, changing the byte content of any report that previously had an oracle partial-coverage asterisk. | Low | The byte-pin renderer tests (B17) all use fixtures where `n_oracle_cells_paired == n_pair_grid` on the byte-pin rows, so the footnote does NOT fire. The byte-pin regex match is unchanged. Test #2 explicitly pins the silent-on-happy-path contract. |
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
   updated `match=` clauses; no other test changes needed
   per B23.4 audit.
5. **NEW tests**: add
   `tests/benchmarks/test_b23_b20_nits_bundle.py` with the
   10 tests.
6. **Verify**: ruff + pyright clean; 940 tests pass.

## Deferred

- **D-B23.1**: extend the oracle partial-coverage footnote
  block to also surface the per-row `ci_method` and
  `bootstrap_ci_fallback_reason` (B21 / D-B21.1 deferral
  intersection). v1 keeps the footnote scoped to oracle
  coverage counts; the BCa fallback surface is a separate
  audit channel.
- **D-B23.2**: extend the model_validator coverage to the
  other 4 RollupRow schemas (e.g., assert
  `n_cells_paired <= n_pair_grid` on
  `EnsembleLiftRollupRow` ALSO, and similar invariants on
  B5 / B6 / B7 / B8 row counts). v1 of B23 is scoped to
  the B20-named D-B20.3 invariant; broader cross-field
  validation across all 5 schemas is a separate audit
  pass.
