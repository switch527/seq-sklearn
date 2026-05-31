# B36 delta: final YAGNI sweep (D-B22.3) — ALL DONE

## Requirements

R-B36-1 (closes D-B22.3): reclassify D-B22.3 (per-fold CIs
across SEEDS / fold-mean uncertainty) as
PERMANENTLY-DEFERRED-YAGNI. This is the LAST open deferral
across the benchmark suite. After B36, every D-Bxx.y entry
is either:
- closed (built and merged in B23-B35), or
- PERMANENTLY-DEFERRED (scope-out, B31), or
- PERMANENTLY-DEFERRED-YAGNI (no current consumer, B32 +
  B33 + B36).

## Non-requirements

- v1 does NOT delete the deferral. The entry remains in
  `docs/benchmark_suite_design_b22_delta.md` with the
  `PERMANENTLY-DEFERRED-YAGNI:` prefix.
- v1 does NOT change any code or test.

## B36.0 Background

D-B22.3 proposes a complementary bootstrap view:
"currently per-fold means bootstrapping the test-rows OF
that fold; a complementary view would bootstrap across the
seeds WITHIN a fold, yielding a fold-mean uncertainty".

The cost analysis:
- New `compute_per_fold_seed_cis` helper in
  `_bootstrap_aggregate.py` (parallel to
  `compute_per_fold_cis` but groups by fold + bootstraps
  per-seed-means).
- New `BOOTSTRAP_PER_SEED_SEED_OFFSET` constant.
- New `per_fold_seed_cis: list[FoldCI] | None` field on
  ALL 5 RollupRow schemas.
- 5 aggregator dispatch sites + corresponding tests.
- Total: ~5 schema fields + 1 helper + 5 dispatch sites +
  ~15 new tests.

The use case is cross-seed reproducibility analysis, which
is statistically meaningful but uncommon. No current
consumer has asked for it. The deferral text itself frames
it as "complementary view ... separate audit field", i.e.
acknowledges no v1 gap. Per CLAUDE.md "Don't add features,
refactor, or introduce abstractions beyond what the task
requires", building this would be speculative.

## B36.1 Sweep design

Single doc edit: prefix the D-B22.3 entry in
`docs/benchmark_suite_design_b22_delta.md` with
`**PERMANENTLY-DEFERRED-YAGNI:**` + rationale, matching
the B32 + B33 sweep pattern.

## B36.2 Implementation outline

1. Edit `docs/benchmark_suite_design_b22_delta.md` D-B22.3
   entry with the prefix + rationale.
2. No code or test changes.
3. Verify: 1097 tests still pass.

## B36.3 ALL DONE inventory

After B36 merges, the open-deferral inventory is empty:

| Phase | Closed | Permanently Deferred |
|---|---|---|
| B20 (orig) | D-B20.1, D-B20.2, D-B20.3 (in B23) | — |
| B21 | D-B21.1 (B24), D-B21.4 (B34), D-B21.5 (B30) | D-B21.2 (YAGNI B32), D-B21.3 (YAGNI B32) |
| B22 | D-B22.1 (B25), D-B22.2 (B35), D-B22.5 (B27) | D-B22.3 (YAGNI B36), D-B22.4 (YAGNI B32) |
| B23 | D-B23.1 (B24), D-B23.2 (B26), D-B23.3 (B26) | — |
| B24 | D-B24.3 (B26) | D-B24.1 (YAGNI B33), D-B24.2 (YAGNI B32) |
| B25 | — | D-B25.1 (scope B31), D-B25.2 (scope B31), D-B25.3 (B30) |
| B26 | D-B26.1 (B28), D-B26.2 (B27), D-B26.3 (B27) | — |
| B27 | D-B27.1 (B30), D-B27.2 (B28) | — |
| B28 | D-B28.1 (B29), D-B28.2 (B29 partial) | — |
| B29 | — | D-B29.1 (scope B31) |
| B30 | — | D-B30.1 (scope B31) |
| B35 | — | D-B35.1 (renderer integration, deferred until consumer) |

Total: 19 built/merged + 13 permanently-deferred = 32
deferrals resolved.

## Deferred

(None added.)
