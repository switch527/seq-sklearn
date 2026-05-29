# B17 design delta: coordinated `primary_loss_*` to `primary_metric_*` rename (D-B16.5)

**Scope**: D-B16.5 lifts the misleading field-name carryover across
the five RollupRow schemas (`RollupRow`, `PairwiseRollupRow`,
`TrainingTimeRollupRow`, `HPOUpliftRollupRow`,
`EnsembleLiftRollupRow`) and the renderer and aggregator modules
that read or construct them. The rename is pure: no behavior
changes, no new fields, no new sentinel routing. The shipped
semantic is already correct (every schema carries a sibling
`primary_metric: str` field naming what the value actually is:
`"log_loss"`, `"rmse"`, `"complementarity_score"`,
`"wall_seconds"`, `"delta"`, `"delta_loss"`); the field-name
`primary_loss_*` was a copy-paste artifact from B13's first
schema (which DOES carry a loss).

## Requirements

The grading rubric for every reviewer finding traces back to one
of these.

- **R-B17-1** All five rollup schemas use `primary_metric_mean`,
  `primary_metric_ci_lo`, `primary_metric_ci_hi` as the
  canonical field names. The pre-existing `primary_metric: str`
  audit field stays unchanged.
- **R-B17-2** All five renderers (`raw_loss.py`, `ensemble.py`,
  `training_time.py`, `hpo_uplift.py`, `ensemble_lift.py`)
  consume the new names. The aggregator modules
  (`bootstrap_rollup.py`, `bootstrap_pairwise.py`,
  `bootstrap_training_time.py`, `bootstrap_hpo_uplift.py`,
  `bootstrap_ensemble_lift.py`) construct rows with the new
  kwargs. Existing rendered Markdown output is byte-identical
  before and after the rename (a regression pin proves this
  on at least one renderer per family).
- **R-B17-3** The five typed parquet shards round-trip through
  `write_*_rollup` to `load_*_rollup` with the new field names.
  No backward-compatibility reader for the old names: the
  shards are bench-run artifacts, never long-lived, and no
  external consumer reads them. Confirmed by a scoped invariant
  (see R-B17-5): zero non-test references to the renamed names
  appear on any `*RollupRow` class field name after the rename.
  The unrelated `LeaderboardEntry.primary_loss_mean` /
  `.primary_loss_std` at
  `benchmarks/report/raw_loss.py:117-118` is OUT OF SCOPE for
  this rename: it is the B5 std-variant leaderboard row that
  aggregates the underlying primary-loss column at the
  manifest level, NOT a bootstrap-CI rollup row.
- **R-B17-4** The shared B14-extracted helpers
  (`format_ci_cell`, `render_rollup_skipped_footnote`,
  `folds_per_group`) keep their signatures unchanged. The
  rename touches only the schema fields and the call sites
  that read them; the shared helpers receive their inputs by
  position, not by field name.
- **R-B17-5** A scoped invariant locks the rename in place going
  forward. A parametrized test over the five named
  `*RollupRow` classes inspects `model_fields` and asserts no
  field name contains `"primary_loss"` other than the
  intentional `primary_loss_column` audit field. The test is
  committed at the bottom of
  `tests/benchmarks/test_bootstrap_manifest.py` so it lives
  with the schema tests and is collected by the default
  `pytest tests/benchmarks/` run. A separate source-tree grep
  test (committed at the bottom of the new
  `tests/benchmarks/test_b17_byte_identity_pins.py`) reads
  every `.py` file under `benchmarks/` and `tests/` and
  asserts zero occurrences of the three renamed field names
  anywhere except in the whitelisted file
  `benchmarks/report/raw_loss.py` (whitelisted as a whole
  file because the unrelated `LeaderboardEntry` schema and
  its sort keys, formatters, and docstrings reference the
  same names; B5 rollup-row regressions inside this file
  are independently covered by Guard A on `*RollupRow`
  schemas plus pyright on `extra="forbid"` attribute reads
  at step 1 of the implementation outline).
- **R-B17-6** The rename is shipped in ONE commit. No staged
  rollout, no deprecation aliases, no `Field(alias=...)`
  trick. The risk of touching the ~299 references in one
  commit (~59 production + ~240 tests on the three renamed
  names) is bounded by ruff + pyright + the existing
  833-test suite plus the scoped grep test (any missed call
  site fails type check, the grep test, or both).

## B17.0 What the rename actually changes

Old (5 schemas):

```python
primary_loss_mean: float | None = None
primary_loss_ci_lo: float | None = None
primary_loss_ci_hi: float | None = None
```

New (5 schemas):

```python
primary_metric_mean: float | None = None
primary_metric_ci_lo: float | None = None
primary_metric_ci_hi: float | None = None
```

The pre-existing `primary_metric: str` field stays where it
is and keeps its values (`"log_loss"`, `"rmse"`,
`"complementarity_score"`, `"wall_seconds"`, `"delta"`,
`"delta_loss"`). The audit field's existence is what makes
the rename safe: a reader can always tell what metric a row's
`primary_metric_mean` actually is.

**Naming note (arch-R1-N1 acknowledgement)**: the rename puts
`primary_metric: str` (a string label) and
`primary_metric_mean: float | None` (a float value) side by
side on every schema. The parallel is intentional and load-
bearing: the underscore-suffix family
(`_mean`, `_ci_lo`, `_ci_hi`) groups the float value with its
two bracketing CI bounds, and the bare `primary_metric` field
labels the whole group. Alternatives like `metric_value_mean`
or `bootstrap_value_mean` break the parallel without adding
clarity for a reader who has read any other rollup row.

The `primary_loss_column: str` field on `HPOUpliftRollupRow`
and `EnsembleLiftRollupRow` is NOT renamed. It carries the
column name in the upstream B5 manifest from which the per-cell
loss was read for the bootstrap (`log_loss`, `rmse`, etc.) and
is genuinely a loss-column reference. Keeping it under
`primary_loss_column` preserves the distinction "metric you
bootstrapped" (`primary_metric`) vs "loss column you read from"
(`primary_loss_column`).

## B17.1 Renderer-side rewrites

Five renderer modules read the renamed fields today:

1. `benchmarks/report/raw_loss.py`: reads the rollup row's
   bootstrap fields to compose the CI cell in the leaderboard
   table.
2. `benchmarks/report/ensemble.py`: reads them to compose the
   CI cell in the pairwise table.
3. `benchmarks/report/training_time.py`: reads them to compose
   the CI cell in the training-time table.
4. `benchmarks/report/hpo_uplift.py`: reads them in both the
   `_render_dataset_block_with_ci` path and the partial-
   coverage footnote path.
5. `benchmarks/report/ensemble_lift.py`: reads them in
   `_render_complete_table_with_ci` and
   `_render_partial_coverage_footnote`.

The shared B14 helper `_bootstrap_render.py` takes positional
`mean`, `ci_lo`, `ci_hi` arguments (verified by reading
`format_ci_cell` and `render_rollup_skipped_footnote`); no
field-name knowledge lives there. R-B17-4 holds without change.

## B17.2 Aggregator-side rewrites

Five aggregator modules construct rollup rows with kwarg names
that match the schema:

1. `benchmarks/report/bootstrap_rollup.py` (B5 aggregator,
   NOT a renderer despite the filename's `rollup` term).
2. `benchmarks/report/bootstrap_pairwise.py` (B6).
3. `benchmarks/report/bootstrap_training_time.py` (B7).
4. `benchmarks/report/bootstrap_hpo_uplift.py` (B15).
5. `benchmarks/report/bootstrap_ensemble_lift.py` (B16).

Each gets the kwarg-name swap. Total: 5 modules x ~6 lines of
construction-site rewrites each.

## B17.3 Test-side rewrites

The full reference inventory across `benchmarks/` and `tests/`
covers ~299 occurrences on the three renamed names (~59
production-side, ~240 test-side). Each test reference falls
into one of three categories:

- **Field-access** (`row.primary_loss_mean` etc.): simple
  identifier swap.
- **Schema-construction kwargs** (`HPOUpliftRollupRow(
  primary_loss_mean=0.2, ...)`): simple kwarg swap.
- **`model_dump()` dict-key assertions** (`assert dump[
  "primary_loss_mean"] == ...`): simple dict-key swap.

The rename is straightforward enough that a single
`sed`-driven sweep is the right tool, followed by ruff +
pyright + the full suite as the safety net. The B17.4
regression pins then prove the rendered Markdown bytes are
unchanged.

## B17.4 Regression pins (B17-specific)

The byte-identity claim of R-B17-2 needs one explicit pin per
renderer family. The B5 family is ALREADY pinned by the
pre-existing
`tests/benchmarks/test_bootstrap_render_regression.py:199-238`
byte-identical Markdown test against
`render_leaderboard_markdown_with_ci`; the rename will update
that test's fixture kwargs so the existing pin keeps firing
under the new names. B17 adds four NEW pins for the families
that have no equivalent: B6 (pairwise), B7 (training-time),
B15 (HPO-uplift), B16 (ensemble-lift):

- `test_render_pairwise_byte_identity_post_rename` (B6).
- `test_render_training_time_byte_identity_post_rename` (B7).
- `test_render_hpo_uplift_byte_identity_post_rename` (B15).
- `test_render_ensemble_lift_byte_identity_post_rename` (B16).

Each is one test that builds a deterministic rollup + result
fixture, calls the renderer, and asserts the output matches
an explicit multi-line golden Markdown string. The golden
strings are captured from the PRE-rename main branch (commit
`2b70345`, the B16 merge) via a one-shot capture script at
`scripts/capture_b17_golden_strings.py` (committed as part
of this PR; the script itself is NOT a test). The capture
script enforces deterministic fixture properties:

- **No NaN values** on any rollup-row CI field. NaN renders
  via the `(no CI)` branch in `format_ci_cell`, which would
  defeat the byte-pin if a single fixture row's float drift
  between captures took the NaN path.
- **Integer-valued floats** for `n_seeds`, `n_folds`,
  `n_cells_paired`, `n_cells_evaluated`, `n_skipped_cells`.
- **Fixed row order** before sort: the script passes rows
  in a sorted-by-`dataset_name` order so the renderer's
  internal sort is the only ordering applied.
- **Pandas + numpy version pin**: the script reads the
  installed pandas + numpy versions at capture time and
  asserts they match the runtime versions when the golden
  string is consumed; a mismatch fails the test with a
  clear message rather than producing a silent byte diff.
- **Branch coverage on each pin**: each pin's fixture
  includes AT LEAST ONE rollup row with `n_cells_paired
  < n_seeds * n_folds` so the renderer's
  `_render_complete_table_with_ci` (or equivalent) takes
  the `partial=True` path through `format_ci_cell`,
  appending the trailing `*` asterisk to the CI cell. This
  branch is the one that reads `primary_loss_mean`,
  `primary_loss_ci_lo`, `primary_loss_ci_hi` on the rollup
  row and feeds them to the shared helper; a rename miss
  here would render a silent `(no CI)` while the pin's
  happy-path bytes match. AND each pin's fixture includes
  AT LEAST ONE rollup row with `n_skipped_cells > 0` so
  the partial-coverage footnote (`_render_partial_coverage_
  footnote` in `ensemble_lift.py:412-421` and the
  equivalent in `hpo_uplift.py:823-844`) is exercised; the
  footnote reads `dataset_name`, `n_skipped_cells`, and
  `n_cells_paired` on the rollup row (NOT the renamed
  fields directly), so it serves as a width-coverage check
  rather than a rename-specific pin. AND the B5
  `LeaderboardEntry` and the B11 `PerDatasetLift`-driven
  incomplete-block rows stay on SEPARATE fixture rows so
  the branches are reachable.

The pins live in a new file
`tests/benchmarks/test_b17_byte_identity_pins.py` so a future
maintainer can see them as a unit. The module docstring
explicitly names the file as a rename-verification artifact,
not a regression test for ongoing behavior, so a future B17.x
edit knows the file is one-time-scoped.

The pre-existing B5 pin keeps its location at
`tests/benchmarks/test_bootstrap_render_regression.py:199-238`
and is updated in place during the sed sweep; it is NOT
duplicated into the new file.

## B17.5 Invariant guards (R-B17-5)

Two guards together close the rename:

**Guard A: schema-field invariant.** A new parametrized test
in `tests/benchmarks/test_bootstrap_manifest.py` reads
`model_fields` on each of the five `*RollupRow` classes and
asserts no field name contains the substring `"primary_loss"`
other than `primary_loss_column` (on `HPOUpliftRollupRow` and
`EnsembleLiftRollupRow`, where the distinction is meaningful).
A future schema addition that re-introduces a `primary_loss_*`
field on any rollup row fails this test at construction time:

```python
@pytest.mark.parametrize(
    "schema",
    [RollupRow, PairwiseRollupRow, TrainingTimeRollupRow,
     HPOUpliftRollupRow, EnsembleLiftRollupRow],
)
def test_rollup_row_has_no_stray_primary_loss_field(
    schema: type[BaseModel],
) -> None:
    """B17 R-B17-5 invariant: no rollup row may declare a
    `primary_loss_*` field other than the dedicated
    `primary_loss_column` audit field. The bootstrap math
    uses `primary_metric_*`; mixing the two leads back to
    the B13 confusion this delta closes."""
    stray = [
        name
        for name in schema.model_fields
        if "primary_loss" in name and name != "primary_loss_column"
    ]
    assert stray == [], (
        f"{schema.__name__} has stray primary_loss_* field(s): "
        f"{stray}; use primary_metric_* per D-B16.5/B17"
    )
```

The parametrize list is EXPLICIT (five named classes), not a
dynamic `__subclasses__()` lookup, so a future schema that is
not yet imported at test collection time would be visible
only after being added to the parametrize list (the
maintainer who adds the schema also adds the parametrize
entry).

**Guard B: source-tree grep invariant.** A test in the new
`tests/benchmarks/test_b17_byte_identity_pins.py` reads every
`.py` file under `benchmarks/` and `tests/` and asserts zero
occurrences of `primary_loss_mean`, `primary_loss_ci_lo`,
`primary_loss_ci_hi` anywhere EXCEPT
`benchmarks/report/raw_loss.py`, which is whitelisted as a
whole file. The whole-file whitelist is the right scope:
`raw_loss.py`'s legitimate references to
`LeaderboardEntry.primary_loss_mean` and
`LeaderboardEntry.primary_loss_std` appear in roughly nine
sites across the file (the `LeaderboardEntry` class
definition, the `entries.sort(...)` key, two formatter
helpers, the `_render_dataset_block` body, and the std-
leaderboard docstring). A narrower AST-class-span whitelist
would miss those legitimate sites and fire on them, blocking
the rename PR for no reason. The B5 std-leaderboard surface
is independently covered by `*RollupRow` Guard A (which is
where any actual rename regression would land) AND by
ruff + pyright on attribute reads. Within the whitelisted
file, the maintainer is trusted to keep the
`LeaderboardEntry.primary_loss_*` reads consistent with the
unrelated `LeaderboardEntry` schema; nothing in B17 touches
that surface.

This guard converts the manual `grep` step the original draft
delegated to a PR description into an enforced CI check for
every file OTHER than `raw_loss.py`. The renamed `*RollupRow`
reads inside `raw_loss.py` (lines 498-500, the
`_render_dataset_block_with_ci` body that constructs the CI
cell from a `RollupRow`) are caught by pyright when the
schema field rename lands at step 1.

## B17.6 D-B16.7 unlock

The B16 Stage 3 R1 deferral of `n_pair_grid` (D-B16.7) is
unlocked by this rename: any schema-additive edit on
`EnsembleLiftRollupRow` can ship in the same coordinated
migration window once the rename is committed. B17 leaves
the additive question for the D-B16.7 design delta to handle
(this delta keeps strict scope on the rename and does not
add `n_pair_grid` or any other new field).

## Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B17-Risk-1 | A renderer call site is missed and reads the renamed field as `None`, silently shipping a `(no CI)` cell instead of failing loudly. | Medium | ruff + pyright catch the attribute-access failure (`extra="forbid"` Pydantic configs make all reads type-checked). The 4 new byte-identity pins + the pre-existing B5 pin catch the rendered-bytes regression. |
| R-B17-Risk-2 | A test references the old name in a `model_dump()` dict-key assertion and the test passes vacuously because the dict no longer carries the key. | Medium | Guard B (source-tree grep test) is the enforced safety net. A `grep -rn "primary_loss_mean\|primary_loss_ci_lo\|primary_loss_ci_hi" benchmarks/ tests/` must return only the `LeaderboardEntry` whitelist hits in `raw_loss.py`; the test computes this assertion mechanically. |
| R-B17-Risk-3 | An external consumer reads a `bootstrap_*_rollup.parquet` shard and expects the old column names. | Low | The rollup shards are bench-run artifacts; no external consumer ships with this library. A repo-wide grep of `notebooks/`, `scripts/*.sh`, `docs/examples_gallery/*.ipynb`, and root `.py` files (architecture-reviewer confirmed in design R1) returns zero hits. The PR description names the rename so any in-flight user-side parquet reader knows to update. |
| R-B17-Risk-4 | The byte-identity pins capture the wrong golden bytes (e.g., a trailing newline drift, NaN format, sort-order drift, pandas version drift). | Low | The capture script at `scripts/capture_b17_golden_strings.py` enforces deterministic-fixture properties (no NaN, integer-valued counts, fixed row order, pandas + numpy version pin). The script runs against commit `2b70345` and writes golden strings into the test file; both invocations of `_render_*` should produce identical bytes. |
| R-B17-Risk-5 | An in-flight branch references the old field names; the rename merges to main before the branch rebases. | Low | Before commit, run `git for-each-ref --format='%(refname:short)' refs/heads/` to enumerate open branches; if any branch shows `primary_loss` hits, communicate the rename via the team's standard channel and offer rebase guidance. The PR title names "D-B16.5 rename" prominently. |

## B17.7 Test surface

- `tests/benchmarks/test_b17_byte_identity_pins.py` (NEW; 4
  byte-identity pins for B6/B7/B15/B16 + 1 source-tree grep
  guard = 5 tests).
- `tests/benchmarks/test_bootstrap_manifest.py` (extended; 1
  parametrized invariant guard = 5 collected items covering
  the 5 schemas).
- `tests/benchmarks/test_bootstrap_render_regression.py`
  (extended in place; pre-existing B5 pin's fixture kwargs
  renamed to the new names; the pin keeps firing as the
  B5-family byte-identity guarantee).
- All other existing test files renamed-in-place (sed sweep
  on `primary_loss_{mean,ci_lo,ci_hi}` to
  `primary_metric_{mean,ci_lo,ci_hi}`).

Expected test delta after the rename:
- Existing tests: 833 to 833 (renamed identifiers; same
  assertions, same outcomes; HEAD baseline at the B16
  merge tip is 833, not 830).
- B17-new: 4 byte-identity pins + 1 source-tree grep guard
  + 5 parametrize cells = 10.
- Total: 833 + 10 = 843 expected post-rename.

## B17.8 Implementation outline

The rename ships in ONE commit (per R-B17-6) but the
mechanical work is broken into 6 sweeps for review-ability:

1. **Schema sweep**: `benchmarks/bootstrap_manifest.py` only
   (5 schemas, 15 field lines). Verify ruff + pyright pass on
   this file alone before moving on; the renderer and
   aggregator side will fail pyright at this point, which is
   the intended canary.
2. **Aggregator sweep**: the 5 modules under
   `benchmarks/report/bootstrap_*.py`. Each constructs rows
   with the schema fields; update the kwargs.
3. **Renderer sweep**: the 5 modules under
   `benchmarks/report/{raw_loss,ensemble,training_time,
   hpo_uplift,ensemble_lift}.py`. Each reads
   `row.primary_loss_*` to populate cells; update the
   field-access calls.
4. **Test sweep**: the ~14 test files. Field-access + kwarg
   + `model_dump()` dict-key swaps. The pre-existing B5
   byte-identity pin's fixture kwargs are renamed in place.
5. **B17-new tests**: add the 4 new byte-identity pins +
   the 1 source-tree grep guard + the 1 parametrized
   invariant guard. Commit `scripts/capture_b17_golden_strings.py`
   so the golden strings are reproducible from `2b70345`.
6. **Verify**: the source-tree grep guard returns zero hits
   (modulo the `LeaderboardEntry` whitelist) anywhere in
   `benchmarks/` or `tests/`; ruff + pyright clean; 843
   tests pass.

## Addressed

R1 swarm: architecture-reviewer (1C / 4I / 2N
REQUEST_CHANGES), qa-test-coverage (1C / 3I / 1N
REQUEST_CHANGES), style-reviewer (15 em dashes in prose
flagged as CRITICAL category). Total deduped: 3 distinct
CRITICAL, 6 IMPROVEMENT, 2 NITPICK.

CRITICALs addressed:

- **arch-C1 / qa-C1** (R-B17-3 invariant unsatisfiable + the
  Risk-R-B17-2 grep is broken because `LeaderboardEntry` in
  `raw_loss.py` has unrelated `primary_loss_mean` and
  `primary_loss_std` fields): R-B17-3 rewritten to scope the
  invariant to `*RollupRow` classes. Guard B (source-tree
  grep) at B17.5 now whitelists `benchmarks/report/raw_loss.py`
  as a whole file. The residual risk inside that file is
  covered by Guard A on the `*RollupRow` schemas plus
  pyright on `extra="forbid"` attribute reads. (The
  R1-draft used an AST-class-span whitelist on
  `LeaderboardEntry`; R2 pivoted to the whole-file
  whitelist after surfacing ~9 legitimate
  `LeaderboardEntry`-related references outside the class
  body that would have been blocked.)
- **qa-C1** (test-count math: baseline is 833, not 830):
  every count in the doc updated. B17.7 Test surface and
  B17.8 step 6 both quote 833 baseline / 843 post-rename.
- **style-C1** (15 em dashes throughout the doc): all
  replaced with colons, semicolons, or restructured. The
  prose now uses prose conjunctions; the numbered-list and
  bullet patterns use `text: description` form.

IMPROVEMENTs addressed:

- **arch-I1** (`bootstrap_rollup.py` mis-classified as a
  renderer): moved to B17.2 aggregator sweep with the
  explicit "NOT a renderer despite the filename's `rollup`
  term" note.
- **arch-I2** (B5 byte-identity pin duplicates the
  pre-existing pin in `test_bootstrap_render_regression.py`):
  the B5 pin is now described as updated-in-place. The 4
  new pins cover the B6/B7/B15/B16 families; the post-
  rename total drops from 840 to 843 with the corrected
  math (existing 833 + 4 new pins + 1 source-tree grep
  guard + 5 parametrize cells = 843).
- **arch-I3** (reference count ~76 stale): updated to ~299
  with the production-side and test-side split.
- **arch-I4** (golden-string capture under-specified): B17.4
  now names `scripts/capture_b17_golden_strings.py`,
  enumerates the deterministic-fixture requirements (no
  NaN, integer-valued counts, fixed row order, pandas +
  numpy version pin), and mandates branch-coverage on the
  partial-coverage footnote in each pin's fixture.
- **qa-I1** (pins cover only the primary CI-cell path;
  footnote branches accessing `primary_loss_*` are not
  pinned): each pin's fixture now MUST include at least
  one rollup row with `n_skipped_cells > 0`. Mandated in
  the B17.4 deterministic-fixture properties.
- **qa-I2** (invariant guard location ambiguous):
  `tests/benchmarks/test_bootstrap_manifest.py` is now the
  mandatory location for Guard A. The "or its own file"
  hedge is removed.
- **qa-I3** (manual grep needs to be an enforced test):
  Guard B at B17.5 is the new test. Reads every `.py` file
  under `benchmarks/` and `tests/`, asserts zero hits on
  the three renamed names outside the `LeaderboardEntry`
  whitelist.

NITPICKs addressed:

- **arch-N1** (`primary_metric_mean` next to `primary_metric:
  str` is mildly confusing): explicit note in B17.0 about
  the intentional parallel + the rejected alternatives.
- **arch-N2** (schema-version mention misleading because no
  `schema_version` field exists): B17.6 rewritten to say
  "any schema-additive edit can ride alongside this rename
  in a single migration window" without implying a
  versioning field.
- **qa-N1** ("or its own file" phrasing): closed by qa-I2.

R1 added a new risk (R-B17-Risk-5: in-flight branch with old
names), now in the Risks table.

### R2 swarm closure

R2 confirming swarm: architecture-reviewer (1C / 1I / 2N
REQUEST_CHANGES), qa-test-coverage (0C / 0I / 1N APPROVE),
style-reviewer (0C / 0I / 2N APPROVE). Total: 1 CRITICAL,
1 IMPROVEMENT, 5 NITPICK. Closures:

- **arch-R2-C1** (Guard B's AST-class-span whitelist on
  `LeaderboardEntry` would miss legitimate references at
  `raw_loss.py` lines 106, 134, 135, 161, 176, 185, 189,
  254, 255 and block the rename PR): Guard B at B17.5 is
  now a whole-file whitelist on `raw_loss.py`. The
  rationale (B5 std-leaderboard surface is covered by
  Guard A + pyright independently) is captured in the
  Guard B text. The narrower class-span whitelist was a
  false economy.
- **arch-R2-I1** (B17.4 branch-coverage motivation named
  `_render_partial_coverage_footnote` as the branch that
  reads the renamed fields, but that footnote actually
  reads `dataset_name`, `n_skipped_cells`, and
  `n_cells_paired`): the B17.4 branch-coverage requirement
  now correctly names the `partial=True` flag through
  `_render_complete_table_with_ci` (or equivalent) as the
  branch that reads the renamed fields. The
  partial-coverage footnote is kept as a separate
  width-coverage check.
- **arch-R2-N1** (Guard B's whitelist-hit count not
  asserted): NOT addressed. Adding a hit-count assertion
  would couple the test to the precise `LeaderboardEntry`
  field layout, which is unrelated to this rename. A
  future field addition to `LeaderboardEntry` re-using a
  forbidden name is a separate concern; if it happens, it
  will be visible in the diff and easy to call out at
  review time. The current Guard B's job is the
  source-tree sweep on the renamed names outside the
  whitelisted file.
- **arch-R2-N2** (already absorbed by arch-R1-N1 closure;
  no new prose needed).
- **qa-R2-N1** (Guard B AST-class-span hit-count not
  asserted): same as arch-R2-N1; deferred for the same
  reason.
- **style-R2-N1** ("robust" at line 301): the word was
  removed when arch-R2-C1 was closed (the AST-class-span
  paragraph that contained it is gone).
- **style-R2-N2** ("unlock" / "is unlocked by this
  rename" at B17.6 section header): the section was
  rewritten in arch-R2-N2 closure prose so the noun was
  the natural fit; both uses are literal not metaphorical.
  NOT changed.

### R3 swarm closure

R3 confirming swarm: architecture-reviewer (0C / 1I / 1N
REQUEST_CHANGES), qa-test-coverage (0C / 0I / 1N APPROVE),
style-reviewer (0C / 0I / 0N APPROVE). Total: 0 CRITICAL,
1 IMPROVEMENT, 2 NITPICK. Closures:

- **arch-R3-I1** (intra-doc contradiction: the R1 closure log
  for arch-C1 / qa-C1 still described Guard B as "reads the
  file via `ast.parse` to confirm hits land only inside the
  `LeaderboardEntry` class definition" while B17.5 and the
  R2 closure paragraph correctly describe the whole-file
  whitelist): the R1-closure entry now matches the R2
  pivot. The AST-class-span mechanism is historical and
  flagged as such in parens.
- **arch-R3-N1** (R-B17-5 requirement prose at lines 62-68
  still described the whitelist as "explicitly-whitelisted
  `LeaderboardEntry` class on `raw_loss.py`"): tightened to
  match the whole-file scope in B17.5 and the R2 closure
  paragraph.
- **qa-R3-N1** (Guard B hit-count not asserted inside
  whitelist file, carried from R2-N1): deferred. A future
  `LeaderboardEntry` field reusing a forbidden name is
  unrelated to this rename and would be visible in any
  future diff that touches `LeaderboardEntry`.

## Deferred

None at R3.
