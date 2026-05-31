# B37 delta: per-fold oracle CIs renderer (D-B35.1)

## Requirements

R-B37-1 (closes D-B35.1): extend
`render_per_fold_cis_footnote` to surface
`per_fold_oracle_cis` alongside `per_fold_cis`. Add a
`scope` column ("main" or "oracle") so a single table
shows both series unambiguously. Triggers when ANY row has
non-None+non-empty `per_fold_cis` OR `per_fold_oracle_cis`.
For non-EnsembleLift schemas, `per_fold_oracle_cis` is
absent; only main rows render.

## Non-requirements

- v1 does NOT add a separate `render_per_fold_oracle_cis_footnote`
  helper. Single helper handles both series via the scope
  column.
- v1 does NOT change `per_fold_cis` or `per_fold_oracle_cis`
  schema fields. Only the renderer changes.

## B37.0 Background

B25 added per-fold CIs renderer for the MAIN delta. B35
added the `per_fold_oracle_cis` schema field and aggregator
population for the ORACLE delta. B35 deferred the renderer
integration as D-B35.1 pending consumer ask. The user
explicitly asked.

## B37.1 Helper signature change

`render_per_fold_cis_footnote` in
`benchmarks/report/_bootstrap_render.py` keeps the same
signature. The body changes:

1. Header gets a `scope` column inserted between
   identifier columns and `fold`.
2. Sep row count increases by 1.
3. Inner loop reads BOTH `per_fold_cis` (scope="main")
   AND `per_fold_oracle_cis` (scope="oracle"; defaults to
   `[]` via `getattr` for schemas that don't carry the
   field). Emits main rows first (in fold_index order),
   then oracle rows (in fold_index order).

## B37.2 Caller pre-filter change

The 5 `_markdown_with_ci` renderers currently filter on
`per_fold_cis is not None and len(per_fold_cis) > 0`. Add
an OR clause for `per_fold_oracle_cis` (gated by
`getattr(r, "per_fold_oracle_cis", None) is not None and
len(per_fold_oracle_cis) > 0`).

For non-EnsembleLift renderers (raw_loss, pairwise,
training_time, hpo_uplift), `getattr(r, "per_fold_oracle_cis", None)`
always returns None (schema doesn't have the field); the OR
clause evaluates to the same as today.

For ensemble_lift, the OR clause fires whenever EITHER main
or oracle per-fold CIs are present.

## B37.3 Tests

Baseline (post-B36): 1097.

### B37.3.1 Helper

1. `test_per_fold_cis_footnote_emits_scope_column_in_header`:
   helper called with a row that has both `per_fold_cis`
   and `per_fold_oracle_cis`; assert `"| scope |"` appears
   in the header.
2. `test_per_fold_cis_footnote_emits_main_rows_with_scope_main`:
   row with `per_fold_cis` populated, no oracle; assert
   `"| main |"` appears and `"| oracle |"` does NOT.
3. `test_per_fold_cis_footnote_emits_oracle_rows_with_scope_oracle`:
   row with `per_fold_oracle_cis` populated, no main;
   assert `"| oracle |"` appears and main rows absent.
4. `test_per_fold_cis_footnote_emits_main_before_oracle_when_both_present`:
   row with both; assert "main" rows appear before
   "oracle" rows in the rendered output.

### B37.3.2 Per-renderer wiring

5. `test_ensemble_lift_renderer_emits_oracle_per_fold_rows_when_present`:
   construct an EnsembleLiftRollupRow with both fields
   populated; render via `render_ensemble_lift_markdown_with_ci`;
   assert the per-fold section contains both scope values.

### B37.3.3 Backward compat

6. `test_per_fold_cis_footnote_for_non_ensemble_lift_schema_omits_oracle`:
   construct a RollupRow (no `per_fold_oracle_cis` field)
   with `per_fold_cis` populated; assert the rendered
   footnote contains `"| main |"` (scope column still
   present) but no oracle rows.

### B37.3.4 Existing B25 tests

The 5 per-renderer header-string assertions (B25 tests
#11-#15) need to include the new `scope` column. Same
cascade pattern as B30 widening; updated in this commit.

Tests #6 and #7 (cell-position pins) need 10-element cell
lists (was 9 after B30, now 10 with scope column inserted
between identifier and fold).

Test #10 (`_correct_eight_fields`): same 8 FoldCI fields
PLUS new scope cell. Update the field assertions to also
include the scope literal.

### B37.3.5 Expected test delta

Baseline: 1097.
- 6 new tests.
- Existing tests: cascaded via header + cell-position
  updates; counts unchanged.
- Total: 1097 + 6 = 1103.

## B37.4 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B37-Risk-1 | Inserting `scope` column shifts every cell-position assertion in B25. | Medium | Same cascade pattern handled at B30 widening; explicit list in B37.3.4. |
| R-B37-Risk-2 | Non-EnsembleLift schemas don't carry `per_fold_oracle_cis`; `getattr` fallback must default to None then list. | Low | Existing helper pattern handles missing fields via `getattr(row, name, "") or ""`; same here with `or []`. |
| R-B37-Risk-3 | `scope` column always shows "main" for non-EnsembleLift schemas; redundant column space. | Low | The single redundant cell per row is the cost of a uniform helper signature; alternative would be 2 helpers + 5 dispatch sites which is worse. |

## Deferred

(None added.)
