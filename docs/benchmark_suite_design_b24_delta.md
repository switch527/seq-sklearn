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

    Raises `ValueError` when `group_columns` and
    `header_labels` differ in length.
    """
```

The helper is pure (no I/O, no module state, deterministic
sort by `group_columns[0]`). Pre-filter by the caller mirrors
the B23 oracle-footnote pattern.

### B24.1.2 Per-renderer call sites

For each of the 5 `_markdown_with_ci` renderers, add the
following after the existing footnote sources and before the
final `"\n".join(parts)`:

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
| credit_default | tft | bca | bca_a_overshoot_threshold |
| pmsm | lightgbm | bca | bca_bca_jackknife_zero_variance |
```

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
| ds_two | 0 | 4 | 4 | bca | bca_a_overshoot_threshold |
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

### B24.4.1 Shared helper (R-B24-1)

1. `test_bca_health_footnote_section_heading`: pass a
   non-empty rollup list; assert the rendered markdown
   contains `"### Bootstrap CI method"`.
2. `test_bca_health_footnote_omits_rows_with_none_fallback`:
   the helper takes pre-filtered rows, so this is enforced
   at the call site, not in the helper. Instead, test the
   CALLER for one renderer (raw_loss): pass a rollup with
   one row having `bootstrap_ci_fallback_reason=None` and
   one having a fallback reason; assert the rendered
   markdown lists only the latter row.
3. `test_bca_health_footnote_silent_when_no_fallback`:
   for the same caller, pass a rollup where every row has
   `bootstrap_ci_fallback_reason=None`; assert the markdown
   does NOT contain `"### Bootstrap CI method"`.
4. `test_bca_health_footnote_unequal_lengths_raises`:
   call the helper directly with mismatched
   `group_columns` and `header_labels`; assert
   `pytest.raises(ValueError)` with the expected message
   (matches the existing
   `render_rollup_skipped_footnote` contract at `:55-59`).
5. `test_bca_health_footnote_long_reason_truncates`:
   pass a row with a `bootstrap_ci_fallback_reason` longer
   than 120 characters; assert the rendered row text
   contains `"..."` and is truncated to 120 chars total.

### B24.4.2 Per-renderer wiring (R-B24-1 parity)

6. `test_raw_loss_renderer_emits_bca_health_when_fallback_present`:
   build a `RollupRow` list with one fallback row; render via
   `render_leaderboard_markdown_with_ci`; assert the footnote
   appears with columns `Dataset | Model | ci_method |
   fallback_reason`.
7. `test_pairwise_renderer_emits_bca_health_when_fallback_present`:
   same shape for `render_pairwise_markdown_with_ci`; assert
   columns `Dataset | Model A | Model B | ci_method |
   fallback_reason`.
8. `test_training_time_renderer_emits_bca_health_when_fallback_present`:
   same for `render_training_time_markdown_with_ci`; assert
   columns `Dataset | Model | Hardware tier | ci_method |
   fallback_reason`.
9. `test_hpo_uplift_renderer_emits_bca_health_when_fallback_present`:
   same for `render_hpo_uplift_markdown_with_ci`; assert
   columns `Dataset | Model | ci_method | fallback_reason`.
10. `test_ensemble_lift_renderer_emits_bca_health_when_main_fallback_present`:
    same for `render_ensemble_lift_markdown_with_ci`; assert
    columns `Dataset | ci_method | fallback_reason`.

### B24.4.3 Oracle footnote extension (R-B24-2)

11. `test_oracle_partial_coverage_footnote_includes_ci_method_column`:
    construct an `EnsembleLiftRollupRow` with partial oracle
    coverage and `bootstrap_ci_method="bca"`; assert the
    rendered footnote row contains `"| bca |"`.
12. `test_oracle_partial_coverage_footnote_includes_fallback_column_when_set`:
    construct a row with partial coverage and
    `bootstrap_oracle_ci_fallback_reason="bca_a_overshoot_threshold"`;
    assert the footnote row contains
    `"| bca_a_overshoot_threshold |"`.
13. `test_oracle_partial_coverage_footnote_dash_when_oracle_fallback_none`:
    construct a row with partial coverage and
    `bootstrap_oracle_ci_fallback_reason=None`; assert the
    footnote row contains `"| - |"` (literal hyphen).
14. `test_oracle_partial_coverage_footnote_long_oracle_reason_truncates`:
    construct a row with a 200-char
    `bootstrap_oracle_ci_fallback_reason`; assert truncation
    to 120 chars including the trailing `"..."`.

### B24.4.4 Byte-pin regression

15. `test_b17_byte_identity_pins.py` byte-pin renderer
    tests must continue to pass. The B17 fixtures all set
    `bootstrap_ci_fallback_reason=None` so the new BCa health
    footnote does NOT fire. The byte-pin assertions are
    `search` + absent-substring (per the R-B23-Risk-2
    closure), so the new oracle-footnote columns appearing
    are still tolerated. No new B17 test needed; B23 fixture
    audit suffices.

### B24.4.5 Expected test delta

- Existing tests: 946 (post-B23 baseline) → 946 unchanged
  (B23 byte-pin tests + B17 fixtures already absent-fallback).
- B24-new: 14 tests.
- Total: 946 + 14 = 960 expected post-build.

## B24.5 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B24-Risk-1 | The new `_render_oracle_partial_coverage_footnote` signature is a contract break for any caller outside `ensemble_lift.py`. | Low | Module-private helper (`_` prefix); single caller in `_render_with_ci`. Grep confirms no other module imports it. |
| R-B24-Risk-2 | Adding the `Bootstrap CI method` footnote changes the byte content of every report that has a fallback. | Low | B17 byte-pin fixtures all have `bootstrap_ci_fallback_reason=None` (verified via grep). The byte-pin test uses `search` + absent-substring, not full-document byte equality (per R-B23-Risk-2 closure). |
| R-B24-Risk-3 | Per-renderer column tuples drift from the `render_rollup_skipped_footnote` per-renderer tuples. | Low | The tuples are the same per renderer (both helpers identify the same rollup row). A future schema change to group_columns updates both helpers symmetrically. |
| R-B24-Risk-4 | The shared helper's truncation length (120 chars) differs from a future reader's expectation. | Low | Matches the existing `render_rollup_skipped_footnote` convention at `:67-68`. Symmetric with the precedent. |

## Deferred

- **D-B24.1**: surface oracle BCa fallback on rollup rows
  with FULL oracle coverage (`n_oracle_cells_paired ==
  n_pair_grid`) — a row with full coverage but oracle BCa
  fallback currently surfaces nowhere in markdown. v1 of B24
  scopes the oracle ci_method/fallback columns to the
  partial-coverage footnote per the literal text of D-B23.1.
  A future "Oracle BCa health" footnote symmetric with the
  main `Bootstrap CI method` footnote would close this gap.
- **D-B24.2**: introduce a separate `bootstrap_oracle_ci_method`
  field on `EnsembleLiftRollupRow` so the oracle and main
  paths can be configured independently. v1 reuses
  `bootstrap_ci_method` for both surfaces because the
  aggregator passes the same constant to both bootstrap
  calls. Tied to D-B21.2 (configurable ci_method per
  experiment).
