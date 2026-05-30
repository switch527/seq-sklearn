# B24 delta: BCa health footnote bundle (D-B21.1 + D-B23.1)

## Requirements

R-B24-1 (closes D-B21.1): every `*_markdown_with_ci` report
surfaces a `Bootstrap CI method` footnote when any rollup
row in that report has `bootstrap_ci_fallback_reason is not
None`. The footnote lists per-row `bootstrap_ci_method` +
`bootstrap_ci_fallback_reason`. Rows without fallback are
omitted. The same trigger rule and helper apply to all 5
renderers (raw_loss, pairwise, training_time, hpo_uplift,
ensemble_lift) for parity.

R-B24-2 (closes D-B23.1): the ensemble_lift oracle
partial-coverage footnote (added in B23 / D-B20.1) gains two
extra columns: `ci_method` (from `bootstrap_ci_method`) and
`oracle_fallback_reason` (from
`bootstrap_oracle_ci_fallback_reason`). The footnote still
triggers on partial oracle coverage; the two new columns
annotate those rows with their BCa health. The
`oracle_fallback_reason` column reads `-` (literal hyphen)
when the rollup row's value is None.

R-B24-3 (parity invariant): for non-ensemble-lift renderers,
the single `bootstrap_ci_method` field on the rollup row
applies to all CI columns in that report. For ensemble_lift,
`bootstrap_ci_method` applies to BOTH the main delta CI and
the oracle delta CI (the aggregator at
`benchmarks/report/bootstrap_ensemble_lift.py:363` passes
`BOOTSTRAP_DEFAULT_CI_METHOD` to both calls). v1 carries no
separate `bootstrap_oracle_ci_method` field.

## Non-requirements

- v1 does NOT add a `bootstrap_oracle_ci_method` field. The
  oracle and main paths use the same constant; introducing a
  separate field is a feature, not a cleanup.
- v1 does NOT surface oracle BCa fallback on rollup rows
  with FULL oracle coverage (`n_oracle_cells_paired ==
  n_pair_grid`). Those rows do not enter the partial-coverage
  footnote, so their `oracle_fallback_reason` is invisible to
  the markdown surface. Captured as deferral D-B24.1 below.
- v1 does NOT modify the parquet schemas. All needed fields
  exist already.

## B24.0 Background

### B24.0.1 What B21 left deferred

B21 added BCa CI to the entity-block bootstrap. The 5 rollup
schemas already carry `bootstrap_ci_method: str` (default
`"percentile"`) and `bootstrap_ci_fallback_reason: str |
None` (`benchmarks/bootstrap_manifest.py:126/132,
:299/305, :350/356, :494/500, :629/635`). The 5 renderers
write but do not READ these fields. D-B21.1 captured this
gap: BCa health is observable in the parquet shard, invisible
in the markdown report.

### B24.0.2 What B23 left deferred

B23 added `_render_oracle_partial_coverage_footnote` in
`benchmarks/report/ensemble_lift.py:484-502` with 4 columns
(`dataset`, `n_oracle_cells_paired`, `n_pair_grid`,
`n_missing`). The `EnsembleLiftRollupRow` schema also carries
`bootstrap_oracle_ci_fallback_reason: str | None`
(`benchmarks/bootstrap_manifest.py:642`), surfaced nowhere in
markdown. D-B23.1 captured this gap.

### B24.0.3 Existing patterns to reuse

- `_bootstrap_render.py:36-73` already houses
  `render_rollup_skipped_footnote(rollup_skipped, *,
  group_columns, header_labels)`. The new helper follows the
  same signature shape: takes pre-filtered rollup rows and a
  pair of column-name / header-label sequences for the
  identifier columns.
- The caller-pre-filter pattern from B23's
  `_render_with_ci` (filter at the call site, pass cleaned
  rows to a pure render helper) applies here too.

## B24.1 R-B24-1 design: shared `Bootstrap CI method` footnote

### B24.1.1 New helper signature

In `benchmarks/report/_bootstrap_render.py`:

```python
def render_bca_health_footnote(
    rollup_with_fallback: Sequence[Any],
    *,
    group_columns: Sequence[str] = ("dataset_name", "model_name"),
    header_labels: Sequence[str] = ("Dataset", "Model"),
) -> str:
    """Render the 'Bootstrap CI method' footnote table.

    `rollup_with_fallback` is the pre-filtered list of rollup
    rows whose `bootstrap_ci_fallback_reason` is non-None.
    `group_columns` and `header_labels` follow the same
    contract as `render_rollup_skipped_footnote`: per-renderer
    identifier columns. The two appended columns are always
    `ci_method` + `fallback_reason`.

    Implicit field contract: each row in
    `rollup_with_fallback` must carry `bootstrap_ci_method:
    str` and `bootstrap_ci_fallback_reason: str | None`
    attributes (read via `getattr(row, name, "")`). All 5 v1
    RollupRow classes carry both fields; a future row type
    lacking them would silently render empty cells. Returns
    `""` on empty input (mirrors
    `_render_oracle_partial_coverage_footnote`).

    Raises `ValueError` when `group_columns` and
    `header_labels` differ in length.
    """
```

The helper is pure (no I/O, no module state, deterministic
sort by `group_columns[0]`). Pre-filter by the caller mirrors
the B23 oracle-footnote pattern.

### B24.1.2 Per-renderer call sites

For each of the 5 `_markdown_with_ci` renderers, add the
following AFTER the existing `render_rollup_skipped_footnote`
call and BEFORE the final `"\n".join(parts)`. Per-renderer
landing line (the line AFTER which the new block is
inserted):

| Renderer | Insert after |
|---|---|
| `raw_loss.py` | the `render_rollup_skipped_footnote` call near `:457` |
| `ensemble.py` (pairwise) | the `render_rollup_skipped_footnote` call near `:380` |
| `training_time.py` | the `render_rollup_skipped_footnote` call near `:387` |
| `hpo_uplift.py` | the `no_paired` block after `_render_partial_coverage_footnote` near `:752` |
| `ensemble_lift.py` | the `render_rollup_skipped_footnote` call at `:462-467` (i.e., AFTER `_render_oracle_partial_coverage_footnote`) |

Block to insert:

```python
rollup_with_fallback = [
    r for r in rollup if r.bootstrap_ci_fallback_reason is not None
]
if rollup_with_fallback:
    parts.append(
        render_bca_health_footnote(
            rollup_with_fallback,
            group_columns=(...),  # per-renderer
            header_labels=(...),  # per-renderer
        )
    )
```

Per-renderer column tuples (matching the existing
`render_rollup_skipped_footnote` invocations):

| Renderer | group_columns | header_labels |
|---|---|---|
| `raw_loss.py` | `("dataset_name", "model_name")` | `("Dataset", "Model")` |
| `ensemble.py` (pairwise) | `("dataset_name", "model_a", "model_b")` | `("Dataset", "Model A", "Model B")` |
| `training_time.py` | `("dataset_name", "model_name", "hardware_tier")` | `("Dataset", "Model", "Hardware tier")` |
| `hpo_uplift.py` | `("dataset_name", "model_name")` | `("Dataset", "Model")` |
| `ensemble_lift.py` | `("dataset_name",)` | `("Dataset",)` |

### B24.1.3 Footnote markdown layout

```
### Bootstrap CI method

| Dataset | Model | ci_method | fallback_reason |
| --- | --- | --- | --- |
| credit_default | tft | bca | a_overshoot |
| pmsm | lightgbm | bca | p0_at_edge |
```

The fallback reason literals (`"p0_at_edge"`,
`"a_overshoot"`) come from
`benchmarks/metrics/bootstrap.py:86-98` (the two BCa fallback
paths). The schema docstrings at
`benchmarks/bootstrap_manifest.py:128-131, :301-304,
:352-355, :496-499, :631-634` document these as the only
two possible values; `None` means no fallback fired.

Truncation: `fallback_reason` longer than 120 chars is
truncated to `"{first 117 chars}..."` matching the
`render_rollup_skipped_footnote` behavior at `:67-68`.

## B24.2 R-B24-2 design: oracle footnote ci_method + fallback columns

### B24.2.1 Signature change

`_render_oracle_partial_coverage_footnote(affected)` in
`benchmarks/report/ensemble_lift.py:484` changes from
`list[tuple[str, int, int, int]]` to
`list[tuple[str, int, int, int, str, str | None]]`. The two
appended tuple positions: `ci_method` (always str),
`oracle_fallback_reason` (str | None; rendered as `-` when
None).

The caller-side pre-filter in `_render_with_ci:433-446`
appends `rollup_row.bootstrap_ci_method` and
`rollup_row.bootstrap_oracle_ci_fallback_reason` to each
tuple.

### B24.2.2 Footnote markdown layout (post-B24)

```
### Oracle partial-coverage footnotes

| dataset | n_oracle_cells_paired | n_pair_grid | n_missing | ci_method | oracle_fallback_reason |
| --- | --- | --- | --- | --- | --- |
| ds_one | 3 | 4 | 1 | percentile | - |
| ds_two | 0 | 4 | 4 | bca | a_overshoot |
```

`-` for the `oracle_fallback_reason` column matches the
`render_rollup_skipped_footnote` placeholder style for empty
strings (`getattr(...) or ""` at `:66` would print empty; the
hyphen here is a small typographic improvement so a reader
distinguishes "no fallback" from "field absent").

Truncation: `oracle_fallback_reason` longer than 120 chars is
truncated to `"{first 117 chars}..."` matching the shared
helper convention.

## B24.3 Implementation outline

1. **Shared helper**: add `render_bca_health_footnote` to
   `benchmarks/report/_bootstrap_render.py` per B24.1.1.
2. **Call sites**: add the pre-filter + helper call to each
   of the 5 `_render_with_ci` functions per B24.1.2.
3. **Oracle footnote**: change
   `_render_oracle_partial_coverage_footnote` signature per
   B24.2.1; update the caller's tuple construction to append
   the two new positions.
4. **Tests**: add `tests/benchmarks/test_b24_bca_health_footnote.py`
   per B24.4.
5. **Verify**: ruff + pyright + scoped pytest pass.

## B24.4 Tests

New test module: `tests/benchmarks/test_b24_bca_health_footnote.py`.

Baseline test count verified live on commit `b569528` via
`.venv/bin/pytest tests/benchmarks/ --collect-only -q | tail`:
`946 tests collected`.

### B24.4.1 Shared helper (R-B24-1 core)

1. `test_bca_health_footnote_section_heading`: pass a
   non-empty rollup list; assert the rendered markdown
   contains `"### Bootstrap CI method"`.
2. `test_bca_health_footnote_unequal_lengths_raises`:
   call the helper directly with mismatched
   `group_columns` and `header_labels`; assert
   `pytest.raises(ValueError)` with the expected message
   (matches the existing
   `render_rollup_skipped_footnote` contract at `:55-59`).
3. `test_bca_health_footnote_long_reason_exact_boundary`
   (qa-R1-C2 closure): pass a row with a 121-char
   `bootstrap_ci_fallback_reason`; assert
   (a) the substring `reason[:117] + "..."` IS in the
   rendered output, AND (b) the full 121-char input is NOT
   in the rendered output. Mirrors
   `test_bootstrap_render_regression.py:141-157` which pins
   the analogous boundary for
   `render_rollup_skipped_footnote`. Kills a mutation that
   truncates at 100 chars or omits the `"..."` suffix.
4. `test_bca_health_footnote_empty_input_returns_empty_string`
   (qa-R1-I1 closure): call
   `render_bca_health_footnote([], group_columns=("dataset_name",),
   header_labels=("Dataset",))`; assert the return value is
   exactly `""`. Pins the empty-input contract documented in
   the helper docstring at B24.1.1 (mirrors
   `_render_oracle_partial_coverage_footnote`'s early-return
   at `ensemble_lift.py:490-491`).

### B24.4.2 Per-renderer present-fallback emission (R-B24-1 parity)

5. `test_raw_loss_renderer_emits_bca_health_when_fallback_present`:
   build a `RollupRow` list with one fallback row; render via
   `render_leaderboard_markdown_with_ci`; assert the footnote
   appears with header `"| Dataset | Model | ci_method |
   fallback_reason |"` exactly (qa-R1-N1 closure: assert
   exact header string so a column-rename PR cannot silently
   drift).
6. `test_pairwise_renderer_emits_bca_health_when_fallback_present`:
   same shape for `render_pairwise_markdown_with_ci`; assert
   exact header `"| Dataset | Model A | Model B | ci_method |
   fallback_reason |"`.
7. `test_training_time_renderer_emits_bca_health_when_fallback_present`:
   same for `render_training_time_markdown_with_ci`; assert
   exact header `"| Dataset | Model | Hardware tier |
   ci_method | fallback_reason |"`.
8. `test_hpo_uplift_renderer_emits_bca_health_when_fallback_present`:
   same for `render_hpo_uplift_markdown_with_ci`; assert
   exact header `"| Dataset | Model | ci_method |
   fallback_reason |"`.
9. `test_ensemble_lift_renderer_emits_bca_health_when_main_fallback_present`:
   same for `render_ensemble_lift_markdown_with_ci`; assert
   exact header `"| Dataset | ci_method | fallback_reason |"`.

### B24.4.3 Per-renderer silent-when-no-fallback (qa-R1-C1 closure)

For each of the 5 renderers, pin the absent-footnote contract
so a pre-filter logic inversion (e.g., `is None` vs `is not
None`) is caught:

10. `test_raw_loss_renderer_silent_when_no_fallback`:
    rollup with every row's `bootstrap_ci_fallback_reason=None`;
    assert `"### Bootstrap CI method"` does NOT appear in the
    rendered markdown.
11. `test_pairwise_renderer_silent_when_no_fallback`: same
    for pairwise.
12. `test_training_time_renderer_silent_when_no_fallback`:
    same for training_time.
13. `test_hpo_uplift_renderer_silent_when_no_fallback`: same
    for hpo_uplift.
14. `test_ensemble_lift_renderer_silent_when_main_fallback_absent`:
    same for ensemble_lift (main fallback only; oracle
    fallback is out of scope for the main footnote per
    R-B24-1).

### B24.4.4 Per-renderer mixed-rollup (qa-R1-I2 closure)

For each of the 5 renderers, pin that only the fallback rows
appear in the footnote when a rollup contains both fallback
and non-fallback rows. A logic bug that pre-filters on the
wrong field (e.g., `bootstrap_skipped_reason` instead of
`bootstrap_ci_fallback_reason`) would be caught here even
when the single-row tests at B24.4.2 pass.

15. `test_renderer_emits_only_fallback_rows_in_mixed_rollup`:
    parametrized over the 5 renderers
    (`raw_loss`, `pairwise`, `training_time`, `hpo_uplift`,
    `ensemble_lift`). Each parametrize case constructs a
    two-row rollup, one with a fallback reason, one without;
    asserts the renderer's BCa health footnote contains
    exactly the fallback row's identifier and does NOT
    contain the non-fallback row's identifier.

### B24.4.5 Oracle footnote extension (R-B24-2)

16. `test_oracle_partial_coverage_footnote_includes_ci_method_column`:
    construct an `EnsembleLiftRollupRow` with partial oracle
    coverage and `bootstrap_ci_method="bca"`; assert the
    rendered footnote row contains `"| bca |"`.
17. `test_oracle_partial_coverage_footnote_includes_fallback_column_when_set`:
    construct a row with partial coverage and
    `bootstrap_oracle_ci_fallback_reason="a_overshoot"`;
    assert the footnote row contains
    `"| a_overshoot |"` (uses the schema-documented literal
    from `benchmarks/metrics/bootstrap.py:86-98`).
18. `test_oracle_partial_coverage_footnote_dash_when_oracle_fallback_none`:
    construct a row with partial coverage and
    `bootstrap_oracle_ci_fallback_reason=None`; assert the
    footnote row contains `"| - |"` (literal hyphen).
19. `test_oracle_partial_coverage_footnote_long_oracle_reason_exact_boundary`
    (qa-R1-C2 closure): construct a row with a 121-char
    `bootstrap_oracle_ci_fallback_reason`; assert
    (a) `reason[:117] + "..."` IS in the rendered output AND
    (b) the full 121-char input is NOT. Same boundary pin
    semantics as test #3.
20. `test_oracle_partial_coverage_footnote_ci_method_column_reads_bootstrap_ci_method_field`
    (qa-R1-I3 closure): construct two
    `EnsembleLiftRollupRow` fixtures differing only in
    `bootstrap_ci_method` (one `"percentile"`, one `"bca"`),
    both with identical partial-coverage shape; render each
    via the full ensemble_lift renderer; assert the
    `ci_method` column in the oracle footnote actually
    changes between the two renderings. Pins R-B24-3 (the
    source binding); a wiring bug that reads a different
    field would survive tests #16-#18 individually.

### B24.4.6 Byte-pin regression (R-B24-2 + qa-R1-N1 closure)

21. `test_b17_byte_identity_pins.py` byte-pin renderer
    tests must continue to pass. The B17 fixtures all set
    `bootstrap_ci_fallback_reason=None` (verified via
    `grep -n "bootstrap_ci_fallback_reason" tests/benchmarks/test_b17_byte_identity_pins.py`)
    so the new BCa health footnote does NOT fire. The
    byte-pin assertions are `search` + absent-substring (per
    the R-B23-Risk-2 closure), so the new oracle-footnote
    columns appearing are still tolerated. AFTER B24 ships,
    the B17 absent-substring assertions on
    `"bootstrap_ci_fallback_reason"` test the raw field name
    NOT leaking; they do NOT test the new section heading
    `"### Bootstrap CI method"` is absent. The new test #10
    above (`test_raw_loss_renderer_silent_when_no_fallback`)
    covers the section-heading absent contract for raw_loss;
    tests #11-#14 cover it for the other 4 renderers. The
    B17 pins remain valid for what they test (the raw
    field name); no B17 test edit is needed for B24.

### B24.4.7 Expected test delta

Baseline (commit `b569528`): 946 tests collected.

- Existing tests: 946 → 946 unchanged (B23 byte-pin tests +
  B17 fixtures already absent-fallback).
- B24-new: 20 named tests + 4 extra parametrize collected
  on test #15 (5 renderers - 1 named = 4 extras).
- Total named: 946 + 20 = 966.
- Total collected: 946 + 20 + 4 = 970.

## B24.5 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B24-Risk-1 | The new `_render_oracle_partial_coverage_footnote` signature is a contract break for any caller outside `ensemble_lift.py`. | Low | Module-private helper (`_` prefix); single caller in `_render_with_ci`. `grep -rn _render_oracle_partial_coverage_footnote benchmarks tests` returns 2 hits (the def at `benchmarks/report/ensemble_lift.py:484` and the caller at `:448`); 0 hits in tests. |
| R-B24-Risk-2 | Adding the `Bootstrap CI method` footnote changes the byte content of every report that has a fallback. | Low | B17 byte-pin fixtures all have `bootstrap_ci_fallback_reason=None` (verified via grep). The byte-pin test uses `search` + absent-substring, not full-document byte equality (per R-B23-Risk-2 closure). |
| R-B24-Risk-3 | Per-renderer column tuples drift from the `render_rollup_skipped_footnote` per-renderer tuples. | Low | The tuples are the same per renderer (both helpers identify the same rollup row). A future schema change to group_columns updates both helpers symmetrically. |
| R-B24-Risk-4 | The shared helper's truncation length (120 chars) differs from a future reader's expectation. | Low | Matches the existing `render_rollup_skipped_footnote` convention at `:67-68`. Symmetric with the precedent. |

## Addressed

R1 design swarm on commit `b569528`: architecture-reviewer
(0C / 4I / 3N APPROVE), qa-test-coverage (2C / 3I / 2N
REQUEST_CHANGES), style-reviewer (1C / 0I / 0N
REQUEST_CHANGES). Deduplicated total: 3 CRITICAL, 7
IMPROVEMENT, 5 NITPICK. Closures:

- **style-R1-C1** (em dash on line 314): replaced the em
  dash with a period; rewrote the D-B24.1 deferral body.
- **qa-R1-C1** (silent-when-no-fallback only tested for
  raw_loss; 4 other renderers have no silent-path test): added
  tests #10-#14 (one per renderer) asserting
  `"### Bootstrap CI method"` does NOT appear when every
  rollup row has `bootstrap_ci_fallback_reason=None`.
- **qa-R1-C2** (truncation tests #5/#14 weak: assert
  `"..."` present but do not pin exactly 120-char total):
  tests rewritten as #3 + #19 with positive boundary asserts
  (`reason[:117] + "..."` present AND full 121-char input
  NOT present). Mirrors
  `test_bootstrap_render_regression.py:141-157`.
- **arch-R1-I1 + qa-R1-N2** (test count baseline 946
  unverified): cited the live count via
  `pytest --collect-only` at the top of B24.4.
- **arch-R1-I2** (helper docstring should call out implicit
  field contract): added paragraph documenting `getattr`
  reads + empty-input return contract.
- **arch-R1-I3** (D-B24.1 deferral hides real concern about
  full-coverage-plus-oracle-fallback): added Cauchy-Schwarz
  empirical-justification paragraph noting `a_overshoot`
  unreachable on small-n oracle surface and `p0_at_edge`
  requires degenerate distribution.
- **arch-R1-I4** (per-renderer footnote placement ambiguous):
  added landing-line table to B24.1.2 with per-renderer
  insert points.
- **qa-R1-I1** (no empty-input test for helper): added test
  #4 asserting `render_bca_health_footnote([], ...) == ""`.
- **qa-R1-I2** (mixed-rollup tested only for raw_loss): added
  parametrized test #15 covering all 5 renderers.
- **qa-R1-I3** (R-B24-3 source binding for `ci_method`
  untested): added test #20 mutating
  `bootstrap_ci_method` across two fixtures and asserting
  the oracle footnote column changes.
- **arch-R1-N5** (example fallback reason strings not from
  schema literals): replaced
  `"bca_a_overshoot_threshold"` and
  `"bca_bca_jackknife_zero_variance"` with the canonical
  `"p0_at_edge"` and `"a_overshoot"` from
  `benchmarks/metrics/bootstrap.py:86-98`; propagated to
  tests #17 and #19.
- **arch-R1-N6** (column-name casing drift not pinned):
  tests #5-#9 now assert exact header strings.
- **arch-R1-N7** (R-B24-Risk-1 grep claim not in doc):
  added the grep command and result to the risk-table cell.
- **qa-R1-N1** (post-B24 B17 deferral assertions test wrong
  thing): added explanation to B24.4.6 that the B17 raw-
  field-name pin remains valid for what it tests; new tests
  #10-#14 cover the section-heading absent contract.

## Deferred

- **D-B24.1**: surface oracle BCa fallback on rollup rows
  with FULL oracle coverage (`n_oracle_cells_paired ==
  n_pair_grid`). A row with full coverage but oracle BCa
  fallback currently surfaces nowhere in markdown. v1 of B24
  scopes the oracle ci_method/fallback columns to the
  partial-coverage footnote per the literal text of D-B23.1.
  Empirical justification: the BCa fallback paths
  (`"p0_at_edge"`, `"a_overshoot"`) fire when the bootstrap
  bias correction or acceleration overshoots; on the oracle
  surface (small-n, per-cell) the Cauchy-Schwarz bound from
  B20 (`|a| <= 1/(6*sqrt(n))`) makes `a_overshoot`
  unreachable through the canonical metric_fn, and
  `p0_at_edge` requires a degenerate distribution rarely
  observed with full coverage. A future "Oracle BCa health"
  footnote symmetric with the main `Bootstrap CI method`
  footnote would close this gap if observed in production.
- **D-B24.2**: introduce a separate `bootstrap_oracle_ci_method`
  field on `EnsembleLiftRollupRow` so the oracle and main
  paths can be configured independently. v1 reuses
  `bootstrap_ci_method` for both surfaces because the
  aggregator passes the same constant to both bootstrap
  calls. Tied to D-B21.2 (configurable ci_method per
  experiment).
