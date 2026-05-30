# B25 delta: per-fold CIs renderer surface (D-B22.1)

## Requirements

R-B25-1 (closes D-B22.1): every `*_markdown_with_ci` report
surfaces a `Per-fold CIs` footnote when ANY rollup row in
that report has a non-None and non-empty
`per_fold_cis: list[FoldCI]`. The footnote lists, for each
fold in each row, the rollup-row identifier columns followed
by EXACTLY these 6 columns in order: `fold`, `metric_mean`,
`metric_ci_lo`, `metric_ci_hi`, `ci_method`,
`ci_fallback_reason`. Rows with `per_fold_cis is None` or
`per_fold_cis == []` are omitted. The same trigger rule and
helper apply to all 5 renderers (raw_loss, pairwise,
training_time, hpo_uplift, ensemble_lift) for parity. The
`FoldCI.n_seeds` and `FoldCI.n_entities` audit fields are
NOT rendered (they remain parquet-shard-audit only; see
deferral D-B25.3). Mirrors the B24 BCa health footnote shape.

## Non-requirements

- v1 does NOT add any new schema field. The
  `per_fold_cis: list[FoldCI] | None` field already exists
  on all 5 RollupRow classes (`bootstrap_manifest.py:139,
  :312, :363, :507, :646`).
- v1 does NOT add a new aggregator code path. The per-fold
  computation already runs via
  `_bootstrap_aggregate.compute_per_fold_cis` at all 5
  per-renderer aggregator call sites.
- v1 does NOT add oracle per-fold rendering (per
  `EnsembleLiftRollupRow.per_fold_cis` covers the MAIN delta
  only; an oracle sibling `per_fold_oracle_cis` is the
  D-B22.2 / B26 scope).
- v1 does NOT add a per-seed-within-fold CI rendering (that
  is D-B22.3 / B27 scope).

## B25.0 Background

### B25.0.1 What B22 delivered

B22 introduced the `bootstrap_per_fold_cis_enabled`
`ExperimentSpec` flag, the `compute_per_fold_cis` helper at
`benchmarks/report/_bootstrap_aggregate.py:115-220`, and a
`per_fold_cis: list[FoldCI] | None` audit field on every
rollup row schema. Each of the 5 aggregators
(`bootstrap_rollup.py`, `bootstrap_pairwise.py`,
`bootstrap_training_time.py`, `bootstrap_hpo_uplift.py`,
`bootstrap_ensemble_lift.py`) computes the list when the
spec flag is enabled and writes it to the parquet shard.

### B25.0.2 What B22 left deferred (D-B22.1)

The field is parquet-shard-audit only. No renderer reads it;
no markdown surface exists. Downstream consumers use
`load_*_rollup` to access the values. The deferral preserved
the data path while postponing the UX surface.

### B25.0.3 Existing patterns to reuse

- `_bootstrap_render.py:76-122` (B24) houses
  `render_bca_health_footnote(rollup_with_fallback, *,
  group_columns, header_labels)`. The new helper follows
  the same signature shape with deterministic sort,
  caller-side pre-filter, and empty-input early return.
- The per-renderer landing-line table in B24.1.2 applies
  unchanged: the new per-fold footnote lands AFTER the BCa
  health footnote block in each of the 5 `_render_with_ci`
  functions (kept after the BCa health block so the section
  ordering in the report is stable: skipped, BCa health,
  per-fold CIs).
- `FoldCI` class at `bootstrap_manifest.py:64-90` carries
  `fold_index, n_seeds, n_entities, metric_mean (nullable),
  metric_ci_lo (nullable), metric_ci_hi (nullable),
  ci_method, ci_fallback_reason (nullable)`. The renderer
  reads all 8 fields.

## B25.1 R-B25-1 design

### B25.1.1 New helper signature

In `benchmarks/report/_bootstrap_render.py`:

```python
def render_per_fold_cis_footnote(
    rollup_with_per_fold: Sequence[Any],
    *,
    group_columns: Sequence[str] = ("dataset_name", "model_name"),
    header_labels: Sequence[str] = ("Dataset", "Model"),
) -> str:
    """B25 / D-B22.1: render the 'Per-fold CIs' footnote.

    Caller pre-filter contract: pass only rows whose
    `per_fold_cis` is non-None AND non-empty (i.e.,
    `r.per_fold_cis is not None and len(r.per_fold_cis) > 0`).
    The helper's own empty-Sequence early return at the outer
    `rollup_with_per_fold` level returns `""`; the per-row
    empty check is the caller's responsibility.

    Each row must carry `per_fold_cis: list[FoldCI]`. The
    helper reads ONLY 6 FoldCI fields per fold:
    `fold_index, metric_mean, metric_ci_lo, metric_ci_hi,
    ci_method, ci_fallback_reason`. The `n_seeds` and
    `n_entities` audit fields are intentionally NOT
    rendered (parquet-shard-audit only; D-B25.3).

    All 5 v1 RollupRow classes carry the `per_fold_cis`
    field; a future row type lacking it would silently
    render with no fold rows via
    `getattr(row, "per_fold_cis", []) or []`.

    Rows in `rollup_with_per_fold` are sorted by
    `group_columns[0]`. Folds within each row are sorted
    defensively by `fold_index` ascending: although
    `compute_per_fold_cis` emits ascending order today, the
    helper does not rely on caller ordering so reproducible
    report bytes survive any future change in the
    aggregator's emit order.

    Nullable metric cells render as `-` (literal hyphen) so
    a reader distinguishes "fold ran but BCa fallback left
    no CI" from "fold absent". `ci_fallback_reason=None`
    renders as `-`. The 120-char truncation on
    `ci_fallback_reason` matches `render_bca_health_footnote`
    and `render_rollup_skipped_footnote`.

    Raises:
        ValueError: when `group_columns` and `header_labels`
            differ in length (matches
            `render_rollup_skipped_footnote` and
            `render_bca_health_footnote`).
    """
```

The helper is pure (no I/O, no module state). Caller
pre-filter pattern mirrors the B24 helper.

### B25.1.2 Per-renderer call sites

For each of the 5 `_markdown_with_ci` renderers, add the
following AFTER the new B24 `render_bca_health_footnote`
block and BEFORE the final `"\n".join(parts)`. Per-renderer
landing line:

| Renderer | Insert after |
|---|---|
| `raw_loss.py` | the B24 `render_bca_health_footnote` block at `:464` |
| `ensemble.py` (pairwise) | the B24 block at `:387` |
| `training_time.py` | the B24 block at `:394` |
| `hpo_uplift.py` | the B24 block at `:759` |
| `ensemble_lift.py` | the B24 block at `:479` |

Block to insert:

```python
# B25 / D-B22.1: per-fold CIs footnote.
rollup_with_per_fold = [
    r for r in rollup if r.per_fold_cis is not None and len(r.per_fold_cis) > 0
]
if rollup_with_per_fold:
    parts.append(
        render_per_fold_cis_footnote(
            rollup_with_per_fold,
            group_columns=(...),  # per-renderer
            header_labels=(...),  # per-renderer
        )
    )
```

Per-renderer column tuples match the B24 table exactly:

| Renderer | group_columns | header_labels |
|---|---|---|
| `raw_loss.py` | `("dataset_name", "model_name")` | `("Dataset", "Model")` |
| `ensemble.py` (pairwise) | `("dataset_name", "model_a", "model_b")` | `("Dataset", "Model A", "Model B")` |
| `training_time.py` | `("dataset_name", "model_name", "hardware_tier")` | `("Dataset", "Model", "Hardware tier")` |
| `hpo_uplift.py` | `("dataset_name", "model_name")` | `("Dataset", "Model")` |
| `ensemble_lift.py` | `("dataset_name",)` | `("Dataset",)` |

### B25.1.3 Footnote markdown layout

```
### Per-fold CIs

| Dataset | Model | fold | metric_mean | metric_ci_lo | metric_ci_hi | ci_method | ci_fallback_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| credit_default | tft | 0 | 0.2150 | 0.2000 | 0.2300 | bca | - |
| credit_default | tft | 1 | 0.2200 | 0.2050 | 0.2350 | bca | a_overshoot |
| credit_default | lightgbm | 0 | 0.1900 | 0.1800 | 0.2000 | percentile | - |
```

Float cells use `%.4f` matching the existing `format_ci_cell`
4-decimal convention. Null cells render as `-`. The
`ci_fallback_reason` literals (`"p0_at_edge"`,
`"a_overshoot"`) come from
`benchmarks/metrics/bootstrap.py:86-98`.

Section ordering in the report (`_render_with_ci` parts
list) becomes, where each "(where applicable)" item only
appears in renderers that emit it today:

1. Header
2. Per-dataset CI table
3. Wilcoxon / Friedman blocks (where applicable: ensemble_lift only)
4. Partial-coverage footnote (B16; where applicable: ensemble_lift and hpo_uplift only)
5. Oracle partial-coverage footnote (B23 / B24 columns; where applicable: ensemble_lift only)
6. Rollup-skipped footnote (B14 / B16; all 5 renderers)
7. **Bootstrap CI method footnote (B24; all 5 renderers)**
8. **Per-fold CIs footnote (B25, NEW; all 5 renderers)**

The per-fold section lands LAST in the parts list, after
whichever of the above sections are present in a given
renderer. Per-fold data is the most granular and least
relevant to a top-level reader.

## B25.2 Implementation outline

1. **Shared helper**: add `render_per_fold_cis_footnote` to
   `benchmarks/report/_bootstrap_render.py` per B25.1.1.
   Add to `__all__`. Update module docstring to mention the
   new helper.
2. **Call sites**: add the pre-filter + helper call to each
   of the 5 `_render_with_ci` functions per B25.1.2. Add
   the helper to each renderer's import block (alphabetical
   order: `render_bca_health_footnote` before
   `render_per_fold_cis_footnote` before
   `render_rollup_skipped_footnote`).
3. **Tests**: add `tests/benchmarks/test_b25_per_fold_cis_footnote.py`
   per B25.3.
4. **Verify**: ruff + pyright + scoped pytest pass.

## B25.3 Tests

New test module: `tests/benchmarks/test_b25_per_fold_cis_footnote.py`.

Baseline test count (commit `6ef4fcd` on main): 973
(verified via
`.venv/bin/pytest tests/benchmarks/ --collect-only -q`).

### B25.3.1 Shared helper (R-B25-1 core)

1. `test_per_fold_cis_footnote_section_heading`: pass a
   non-empty rollup list; assert the rendered markdown
   contains `"### Per-fold CIs"`.
2. `test_per_fold_cis_footnote_unequal_lengths_raises`:
   call the helper directly with mismatched
   `group_columns` and `header_labels`; assert
   `pytest.raises(ValueError, match=r"equal length")`.
3. `test_per_fold_cis_footnote_long_reason_exact_boundary`:
   pass a row with a `ci_fallback_reason` of exactly 121
   chars; assert `reason[:117] + "..."` IS in the output
   AND the full 121-char string is NOT. Mirrors the B24
   helper boundary pin.
4. `test_per_fold_cis_footnote_empty_input_returns_empty_string`:
   call `render_per_fold_cis_footnote([])`; assert the
   return is exactly `""`.
5. `test_per_fold_cis_footnote_120_char_reason_not_truncated`:
   exactly 120 chars passes through unmodified; pins the
   `> 120` boundary.
6. `test_per_fold_cis_footnote_null_metric_renders_dash`:
   FoldCI with `metric_mean=None, metric_ci_lo=None,
   metric_ci_hi=None`; assert the row in the rendered table
   shows `"| - |"` in each of the three metric cells.
7. `test_per_fold_cis_footnote_float_format_4_decimals`:
   FoldCI with metric_mean=0.21567; assert `"| 0.2157 |"`
   appears in the rendered row (ordinary round-up via the
   `%.4f` convention; matches the existing `format_ci_cell`
   precision).
8. `test_per_fold_cis_footnote_sorts_rows_by_group_columns_first_key`:
   two rows with `dataset_name` in reverse alpha order;
   assert the alpha-earlier dataset appears first.
9. `test_per_fold_cis_footnote_sorts_fold_rows_by_fold_index_ascending_when_input_unsorted`
   (arch-R1-C1 + qa-R1-I1 closure): one rollup row with
   three FoldCI entries in OUT-OF-ORDER input
   `fold_index=[2, 0, 1]`; assert the rendered fold rows
   appear in ascending order 0, 1, 2. The helper sorts
   defensively (does not trust caller input) so report
   bytes remain stable if a future change in
   `compute_per_fold_cis` alters the emit order. Kills a
   mutation that omits the helper's defensive sort.
10. `test_per_fold_cis_footnote_cell_data_reads_correct_six_fields`
    (qa-R1-C2 closure): construct a FoldCI with
    `fold_index=7, metric_mean=0.215, metric_ci_lo=0.200,
    metric_ci_hi=0.230, ci_method="bca",
    ci_fallback_reason="p0_at_edge"` (NON-None fallback so
    the cell does NOT render as `-`); assert each of the 6
    literal values appears in the rendered row. Kills a
    getattr-wrong-attr bug on any of the 6 rendered fields.
10a. `test_per_fold_cis_footnote_does_not_surface_n_seeds_or_n_entities`
    (qa-R1-C2 + qa-R1-I2 closure): construct a FoldCI with
    non-zero `n_seeds=99` and `n_entities=88`; assert
    `"n_seeds"` and `"n_entities"` are NOT in the rendered
    output AND the literals `"99"` and `"88"` do NOT appear
    in the per-fold section. Pins R-B25-1's "exactly 6
    columns" contract; a future regression that surfaced
    the audit fields would fail. Implementer note (qa-R2-N1
    closure): use a `dataset_name` and other group-column
    values that do NOT contain the literals `"99"` or
    `"88"` (e.g., `dataset_name="fake_binary"`,
    `model_name="m1"`) so the negative assertions remain
    non-vacuous.
10b. `test_per_fold_cis_footnote_ci_fallback_reason_source_binding`
    (qa-R1-C1 closure): two rollup-row fixtures differing
    only in the FoldCI's `ci_fallback_reason` (one
    `"p0_at_edge"`, one `"a_overshoot"`); render each;
    assert the rendered footnote row in each output
    contains the respective literal AND does NOT contain
    the other. Kills a wiring bug that reads from a fixed
    string or the wrong attribute (analogous to B24's
    qa-R1-I3 test #20 closure).

### B25.3.2 Per-renderer present-fallback emission (R-B25-1 parity)

11. `test_raw_loss_renderer_emits_per_fold_cis_when_present`:
    rollup with one row's `per_fold_cis` populated; render
    via `render_leaderboard_markdown_with_ci`; assert exact
    header `"| Dataset | Model | fold | metric_mean |
    metric_ci_lo | metric_ci_hi | ci_method |
    ci_fallback_reason |"`.
12. `test_pairwise_renderer_emits_per_fold_cis_when_present`:
    same for pairwise; assert exact header
    `"| Dataset | Model A | Model B | fold | metric_mean |
    metric_ci_lo | metric_ci_hi | ci_method |
    ci_fallback_reason |"`.
13. `test_training_time_renderer_emits_per_fold_cis_when_present`:
    same for training_time; assert exact header
    `"| Dataset | Model | Hardware tier | fold | ... |"`.
14. `test_hpo_uplift_renderer_emits_per_fold_cis_when_present`:
    same for hpo_uplift; assert exact header
    `"| Dataset | Model | fold | ... |"`.
15. `test_ensemble_lift_renderer_emits_per_fold_cis_when_present`:
    same for ensemble_lift; assert exact header
    `"| Dataset | fold | ... |"`.

### B25.3.3 Per-renderer silent-when-no-per-fold (R-B25-1)

16-20. `test_<renderer>_renderer_silent_when_per_fold_absent`:
    one per renderer; rollup with every row's
    `per_fold_cis=None`; assert `"### Per-fold CIs"` is NOT
    in the rendered output. Catches a pre-filter logic
    inversion.

### B25.3.4 Mixed-rollup (parametrized over 5 renderers)

21. `test_renderer_emits_only_per_fold_rows_in_mixed_rollup`:
    parametrized over the 5 renderers. Two-row rollup, one
    with `per_fold_cis` populated and one with None; assert
    the populated row's identifier appears in the Per-fold
    CIs section and the None row's identifier does not.

### B25.3.5 Empty list edge case

22. `test_renderer_treats_empty_per_fold_cis_list_as_absent`:
    pass `per_fold_cis=[]` (empty list, not None); assert
    the footnote does NOT fire. The caller pre-filter
    treats empty list as no-data per B25.1.2 (`len(r.per_fold_cis) > 0`).

### B25.3.6 Section ordering (qa-R1-I3 closure)

22a. `test_per_fold_cis_footnote_lands_after_bca_health_footnote`:
    parametrized over the 5 renderers. Construct a rollup
    where one row has BOTH `bootstrap_ci_fallback_reason`
    non-None (triggers B24 BCa health footnote) AND
    `per_fold_cis` non-empty (triggers B25 per-fold footnote);
    render; assert
    `md.find("### Bootstrap CI method") < md.find("### Per-fold CIs")`.
    Catches a mis-ordered insertion that places per-fold
    before BCa health, which would still pass tests
    #11-#15 (header presence).

### B25.3.7 Byte-pin regression (no new test)

The B17 byte-pin renderer tests must continue to pass. The
B17 fixtures all default `per_fold_cis=None` (verified via
`grep -n "per_fold_cis" tests/benchmarks/test_b17_byte_identity_pins.py`
returning 4 sites: `:166, :225, :298, :382`, each an
`assert "per_fold_cis" not in md`) so the new Per-fold CIs
footnote does NOT fire. The B17 absent-substring
assertions on `"per_fold_cis"` test the raw FIELD NAME (no
underscore-name leak); the new section heading is
`"### Per-fold CIs"` (space, capitalized), so the
field-name assertion remains true. No B17 test edit needed
for B25. This is a regression-pin clarification, not a new
test function (does NOT increment the B25 named-test count).

### B25.3.8 Expected test delta

Baseline (post-B24 main `6ef4fcd`): 973 tests collected.

- Existing tests: 973 -> 973 unchanged.
- B25-new: 25 named tests (was 22 in R1; R1 closures added
  test #10a `n_seeds_or_n_entities absent`, test #10b
  `ci_fallback_reason_source_binding`, test #22a
  `lands_after_bca_health_footnote`).
- Parametrize extras: test #21 adds 4 (5 renderers - 1
  named); test #22a adds 4 (5 renderers - 1 named). Total
  parametrize extras: 8.
- Total named: 973 + 25 = 998.
- Total collected: 973 + 25 + 8 = 1006.

## B25.4 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B25-Risk-1 | The new helper signature drifts from `render_bca_health_footnote` in subtle ways (sort, truncation, hyphen placeholder). | Low | Helper modeled directly on the B24 helper. Test #5 (boundary pin) + test #6 (hyphen) + test #8 (sort) explicitly mirror the B24 tests. |
| R-B25-Risk-2 | Adding the `Per-fold CIs` footnote changes the byte content of every report that has any per-fold-enabled row. | Low | B17 byte-pin fixtures all default `per_fold_cis=None` (verified via grep). The byte-pin assertions use `search` + absent-substring (per R-B23-Risk-2 closure), so the new footnote text appearing is still tolerated. |
| R-B25-Risk-3 | The fold row count could explode in large reports (`n_datasets * n_models * n_folds` rows). | Low | The footnote is gated on `bootstrap_per_fold_cis_enabled=True` at the experiment-spec level (`config.py:225`, default `False`). Reports without per-fold data render nothing. Reports with per-fold data accept the size as the documented audit channel. |
| R-B25-Risk-4 | Empty-list (`per_fold_cis=[]`) vs None semantic confusion. | Low | The caller pre-filter at B25.1.2 treats empty list as no-data, matching the helper's "fallback to no-row" behavior. Test #22 pins this contract. |

## Addressed

R1 design swarm on commit `e07b908`: architecture-reviewer
(1C / 4I / 2N REQUEST_CHANGES), qa-test-coverage (2C / 3I /
2N REQUEST_CHANGES), style-reviewer (0C / 0I / 0N APPROVE).
Deduplicated total: 3 CRITICAL, 7 IMPROVEMENT, 4 NITPICK.
Closures:

- **arch-R1-C1 + qa-R1-I1** (test #9 self-contradicts: says
  "reverse order" then "construct pre-sorted"): test #9
  rewritten with input `fold_index=[2, 0, 1]` (deliberately
  out-of-order) and assertion that the rendered output is
  in ascending order 0, 1, 2. Helper docstring updated to
  state "Folds within each row are sorted defensively by
  `fold_index` ascending" so the helper does not rely on
  caller ordering.
- **qa-R1-C1** (test #10 lacks source-binding pin for
  `ci_fallback_reason`; hardwired `"-"` bug survives if
  fixture uses None): test #10 fixture now mandates
  `ci_fallback_reason="p0_at_edge"` (non-None) so the
  rendered cell is the literal token. Added new test #10b
  (`ci_fallback_reason_source_binding`) constructing two
  fixtures differing only in `ci_fallback_reason` and
  asserting the rendered outputs differ accordingly.
  Mirrors B24's qa-R1-I3 test #20 pattern.
- **qa-R1-C2** (spec contradiction: R-B25-1 lists 6 columns
  but helper docstring + test #10 say "all 8 fields"):
  R-B25-1 reaffirmed as the authority (6 rendered columns:
  fold, metric_mean, metric_ci_lo, metric_ci_hi, ci_method,
  ci_fallback_reason). Helper docstring rewritten to read
  "6 fields" and explicitly state n_seeds + n_entities are
  NOT rendered. Test #10 renamed to
  `cell_data_reads_correct_six_fields`. Added new test
  #10a (`does_not_surface_n_seeds_or_n_entities`) pinning
  the audit-field non-exposure contract. Added deferral
  D-B25.3 for the audit-field surface.
- **arch-R1-I1** (landing-line table cites
  `render_rollup_skipped_footnote` sites, not the B24-block
  sites): table updated to cite the actual B24-block lines
  (`:464, :387, :394, :759, :479`).
- **arch-R1-I2** (section ordering list conflated
  per-renderer applicability): items 3-5 now carry
  "(where applicable: ...)" qualifiers and the closing
  paragraph states "lands LAST after whichever of the above
  sections are present".
- **arch-R1-I3 + qa-R1-N2** (B17 absent-substring cites 3
  line numbers, missing `:225`): B25.3.7 enumerates all 4
  B17 sites (`:166, :225, :298, :382`) from a live grep
  command.
- **arch-R1-I4** (helper docstring missing `Raises:` line):
  added `Raises: ValueError` section to the docstring
  matching B24's precedent.
- **qa-R1-I2** (no test pins n_seeds + n_entities are NOT
  rendered): added test #10a (see qa-R1-C2 closure above).
- **qa-R1-I3** (per-renderer tests #11-#15 don't assert
  section ordering vs BCa): added parametrized test #22a
  (`lands_after_bca_health_footnote`) pinning the
  `BCa-health < per-fold` order across all 5 renderers.
- **arch-R1-N1** (test #7 banker rounding mention
  misleading): rewritten as "ordinary round-up via the
  `%.4f` convention".
- **arch-R1-N2** (helper docstring conflates outer empty
  vs per-row empty contracts): rewritten with separate
  caller-pre-filter contract and outer empty-Sequence
  return.
- **qa-R1-N1** (B25.3.6 item #23 numbered alongside test
  functions): B25.3.7 renamed (was B25.3.6) with explicit
  "does NOT increment the B25 named-test count" note.

Test count after R1 closures: 25 named (was 22; +#10a,
+#10b, +#22a); 1006 collected (973 baseline + 25 named +
8 parametrize extras from #21 and #22a both across 5
renderers).

## Deferred

- **D-B25.3**: surface `FoldCI.n_seeds` and
  `FoldCI.n_entities` (the per-fold sample-size audit
  fields) in the renderer. v1 of B25 keeps the rendered
  table to the 6 CI-relevant columns to bound table width;
  a future "Per-fold sample sizes" sibling footnote or a
  widened table could surface the audit fields if a
  consumer asks for them.
- **D-B25.1**: render per-fold CIs as an EXPANDABLE
  sub-table (markdown `<details>` block) rather than a flat
  footnote. v1 ships the flat footnote because rendering
  a `<details>` block in a markdown viewer that strips HTML
  collapses the data. A future renderer-format negotiation
  pass (e.g., separate HTML output) could surface this as
  expandable.
- **D-B25.2**: surface per-fold CIs as percentile-banded
  visualizations (e.g., fold-vs-CI box plot inline). v1
  keeps to flat tabular data; visual rendering is a
  separate audit channel out of scope for the markdown
  reporter.
