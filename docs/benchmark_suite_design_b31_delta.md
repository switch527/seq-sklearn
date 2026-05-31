# B31 delta: permanent-defer sweep (D-B25.1 + D-B25.2 + D-B29.1 + D-B30.1)

## Requirements

R-B31-1 (sweep closure): reclassify 4 open deferrals as
PERMANENTLY-DEFERRED with explicit rationale. Each is
either an explicit "not-chosen alternative" or a
"no-available-fields" item with no concrete work to do.
Doc-only sweep; zero code changes.

The 4 deferrals:
- **D-B25.1**: render per-fold CIs as an EXPANDABLE
  `<details>` block. Markdown viewers that strip HTML
  collapse the data; v1 ships flat footnote (B25). This is
  a renderer-FORMAT decision, not a missing feature: HTML
  output is out of scope for the markdown reporter.
- **D-B25.2**: surface per-fold CIs as percentile-banded
  visualizations. v1 markdown reporter is text-only; image
  generation is a separate output channel out of scope.
- **D-B29.1**: structural bounds on PairwiseRollupRow and
  TrainingTimeRollupRow. These schemas don't carry
  `n_rows` / `n_entities` / `n_folds`; no clean structural
  invariant is available without adding fields. Schema
  fields are stable; this is permanent.
- **D-B30.1**: emit per-fold sample sizes as a SIBLING
  footnote instead of widening the existing table. B30
  picked option a (widening); option b would render the
  same data in a different layout with no information
  delta. v1 chose option a.

## Non-requirements

- v1 does NOT delete the deferrals. The entries remain in
  each source doc but get a `PERMANENTLY-DEFERRED` prefix
  so future maintainers can grep for open vs permanent.
- v1 does NOT change any code or test.

## B31.0 Background

Across phases B25-B30, the running deferral list accumulated
4 entries that were explicitly framed as "not-chosen
alternative" (D-B25.1, D-B25.2, D-B30.1) or
"no-available-fields" (D-B29.1). Each one was added with the
expectation that a future audit might reopen the decision.
After 8 phases of cleanup the architectural picture is
stable: the markdown reporter is text-only by design;
PairwiseRollupRow + TrainingTimeRollupRow have stable
schemas with no proposed field additions; the widening +
sibling-footnote layout choices for the per-fold per-fold
data have no information delta. Reclassifying as
PERMANENTLY-DEFERRED makes the deferral inventory
honest: 12 open deferrals on the post-B30 main are really
8 open + 4 permanent.

## B31.1 Sweep design

For each of the 4 deferral entries, prefix the bullet body
with `**PERMANENTLY-DEFERRED:** ` and append a one-sentence
rationale anchored on architecture/scope. Format:

```
- **D-B25.1**: **PERMANENTLY-DEFERRED:** render per-fold CIs
  as an EXPANDABLE `<details>` block. [original body...]
  Reclassified B31 as a renderer-format alternative out of
  scope for the v1 markdown reporter; revisit only if a
  separate HTML output channel is added.
```

The original body text stays intact for archeology; the
new prefix + sentence makes the classification explicit.

## B31.2 Implementation outline

1. Edit `docs/benchmark_suite_design_b25_delta.md` D-B25.1
   + D-B25.2 entries with the prefix + rationale.
2. Edit `docs/benchmark_suite_design_b29_delta.md` D-B29.1
   entry with the prefix + rationale.
3. Edit `docs/benchmark_suite_design_b30_delta.md` D-B30.1
   entry with the prefix + rationale.
4. No code or test changes.
5. Verify: ruff (no source change), pyright (no source
   change), pytest (no test change, just confirm 1084
   still passes after the doc edit).

## B31.3 Tests

No new tests. The 1084-test baseline must continue to pass
(the doc edits are inert).

## B31.4 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B31-Risk-1 | A future maintainer reads PERMANENTLY-DEFERRED as "do not reopen ever" and misses a legitimate trigger to revisit. | Low | Each rationale names the trigger condition (e.g., "revisit only if separate HTML output channel is added"). |
| R-B31-Risk-2 | The deferral inventory drifts again over future phases. | Low | The PERMANENTLY-DEFERRED prefix is grep-stable; future doc reviews can use `grep -r "PERMANENTLY-DEFERRED" docs/` to enumerate. |

## Deferred

(None added.)
