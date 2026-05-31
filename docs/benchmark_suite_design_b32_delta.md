# B32 delta: YAGNI permanent-defer sweep (D-B21.2 + D-B21.3 + D-B22.4 + D-B24.2)

## Requirements

R-B32-1 (sweep closure): reclassify 4 open deferrals as
PERMANENTLY-DEFERRED-YAGNI. Each is a feature gap with no
current consumer; per CLAUDE.md "Don't add features,
refactor, or introduce abstractions beyond what the task
requires", these should be left as design notes for future
maintainers rather than open work items.

The 4 deferrals:
- **D-B21.2**: configurable `ci_method` per-experiment via
  `ExperimentSpec.bootstrap_ci_method`. v1 hard-defaults to
  BCa via `BOOTSTRAP_DEFAULT_CI_METHOD`. Per-experiment
  override has no current consumer; adding the
  `ExperimentSpec` field + 5 aggregator dispatch sites for
  a reviewer-debugging convenience is YAGNI.
- **D-B21.3**: ABC (approximate bootstrap confidence) as a
  third CI method. Literature backing exists but no current
  consumer has asked for ABC over BCa. Adding a third
  variant to the primitive expands API surface without
  delivering value.
- **D-B22.4**: per-kind override on
  `bootstrap_per_fold_cis_enabled` (5 booleans where 1
  currently suffices). The original deferral text already
  flags this as "5 redundant booleans with no per-kind
  variation at the v1 helper level"; classic YAGNI.
- **D-B24.2**: separate `bootstrap_oracle_ci_method` field
  on `EnsembleLiftRollupRow` so oracle + main paths can be
  configured independently. Tied to D-B21.2 (both require
  per-experiment ci_method config); same YAGNI reasoning.

## Non-requirements

- v1 does NOT delete any deferral. Each gets a
  `PERMANENTLY-DEFERRED-YAGNI:` prefix + rationale; the
  original body stays for archeology.
- v1 does NOT change any code or test.

## B32.0 Background

After 9 phases of cleanup (B23-B31), 8 open deferrals
remained on main. Reviewing each through the CLAUDE.md
anti-feature lens reveals 4 are feature-gaps with no
current consumer. Building them would violate the
project's stated rule against speculative features. The
honest move is to acknowledge them as design notes
explaining what could be built if a future consumer asks,
not as open work items.

The remaining 4 (D-B21.4 refactor, D-B22.2 oracle per-fold,
D-B22.3 per-seed bootstrap, D-B24.1 oracle fallback
rendering) ARE real work items addressed in subsequent
phases (B33-B36).

## B32.1 Sweep design

Same pattern as B31: prefix each deferral bullet with
`**PERMANENTLY-DEFERRED-YAGNI:**` and append a
one-sentence rationale anchored on "no current consumer".

The PERMANENTLY-DEFERRED-YAGNI prefix is grep-distinct
from B31's PERMANENTLY-DEFERRED so a future audit can
distinguish "scope-out" (B31) from "YAGNI" (B32).

## B32.2 Implementation outline

1. Edit `docs/benchmark_suite_design_b21_delta.md` D-B21.2
   + D-B21.3 entries.
2. Edit `docs/benchmark_suite_design_b22_delta.md` D-B22.4
   entry.
3. Edit `docs/benchmark_suite_design_b24_delta.md` D-B24.2
   entry.
4. No code or test changes.
5. Verify: 1084 tests still pass (doc edits inert).

## B32.3 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B32-Risk-1 | A future user appears with a real need for one of the deferred features. | Low | Each rationale names the trigger ("if a consumer asks for per-experiment ci_method dispatch, reopen D-B21.2"). |
| R-B32-Risk-2 | YAGNI classification conflates "out of scope" with "low priority". | Low | YAGNI is the right framing per the codebase's stated rule; if the future need materializes, the deferral converts back to an open item. |

## Deferred

(None added.)
