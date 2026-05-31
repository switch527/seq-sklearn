# Benchmark suite phase log (B12–B39)

After the initial benchmark suite design (`benchmark_suite_design.md`,
B1–B11) shipped, the suite went through 27 incremental phases. Each
phase had its own design doc + iterative Claude swarm review + merge.
Once a phase landed on `main`, the per-phase design doc became
historical residue — the durable record of *what* was built lives in
the code + tests, and the durable record of *why* lives in the merge
commit messages.

This log keeps the historical index in one place; the full per-phase
design docs are recoverable from git history at the merge SHAs below.

## Foundational adds

| Phase | Title | Merge |
|---|---|---|
| **B12** | classical-TSC adapter family | `c3e23a0` |
| **B13** | entity-block bootstrap CIs (B5.4) | `e58a4d3` |

## Per-aggregator CI integration (B13.x family closures)

| Phase | Closes | Title | Merge |
|---|---|---|---|
| **B14** | D-B13.1 + D-B13.2 | pairwise + training-time CIs | `bd0f322` |
| **B15** | D-B13.3 | HPO-uplift Δ-statistic paired CI | `f906e24` |
| **B16** | D-B13.4 | ensemble-lift Δloss paired CI | `2b70345` |
| **B17** | D-B16.5 | `primary_loss_*` → `primary_metric_*` rename | `4ff8983` |
| **B18** | D-B16.6 | shared CLI-wrapper factory | `5821691` |
| **B19** | D-B16.7 | `n_pair_grid` on `EnsembleLiftRollupRow` | `7f6d550` |

## Methodological upgrades

| Phase | Closes | Title | Merge |
|---|---|---|---|
| **B20** | D-B16.1 | bootstrap CI on the per-sample-best oracle Δ | `3358651` |
| **B21** | D-B16.2 / D-B13.5 | BCa CI on the entity-block bootstrap | `b2ba135` |
| **B22** | D-B16.3 / D-B13.6 | per-fold CIs on all 5 rollup aggregators | `426af5d` |

## Renderer surfaces

| Phase | Closes | Title | Merge |
|---|---|---|---|
| **B23** | D-B20.1/2/3 | B20 NITs bundle (footnote + raise tokens + validator) | `e0d2214` |
| **B24** | D-B21.1 + D-B23.1 | BCa health footnote bundle | `6ef4fcd` |
| **B25** | D-B22.1 | per-fold CIs renderer surface | `c915015` |

## Schema and validator cleanups

| Phase | Closes | Title | Merge |
|---|---|---|---|
| **B26** | D-B23.2 + D-B23.3 + D-B24.3 | cleanup bundle | `99f5128` |
| **B27** | D-B22.5 + D-B26.2 + D-B26.3 | small cleanups bundle | `37cbfc0` |
| **B28** | D-B27.2 + D-B26.1 | schema-validator cleanups | `f42f01c` |
| **B29** | D-B28.1 + D-B28.2 partial | extend cell-count bounds | `a5d307d` |
| **B30** | D-B21.5 + D-B27.1 + D-B25.3 | tiny cleanups bundle | `fa68c28` |

## Permanent-deferral sweeps

| Phase | Closes | Title | Merge |
|---|---|---|---|
| **B31** | D-B25.1/2 + D-B29.1 + D-B30.1 | permanent-defer sweep (4 deferrals) | `ac4ae1a` |
| **B32** | D-B21.2/3 + D-B22.4 + D-B24.2 | YAGNI permanent-defer sweep (4 deferrals) | `d8a79ef` |
| **B33** | D-B24.1 | YAGNI defer (no design doc; one-line decision) | `4bd79bb` |
| **B36** | D-B22.3 | final YAGNI sweep (ALL DONE marker) | `391fad5` |

## Late refactors and feature closures

| Phase | Closes | Title | Merge |
|---|---|---|---|
| **B34** | D-B21.4 | `BootstrapResult` dataclass refactor | `bf02b5e` |
| **B35** | D-B22.2 | oracle per-fold CIs | `89a6fcf` |
| **B37** | D-B35.1 | per-fold oracle CIs renderer | `06838f6` |
| **B38** | D-B13.7 | sufficient-stats bootstrap fast path | `1a5736f` |
| **B39** | D-B12.6 | TSC categorical one-hot handling | `17fb3fc` |

## Deferral status after B39

The original B13-followup and B12-followup deferrals listed in
`benchmark_suite_implementation_plan.md` are now all either:

- closed by a later phase (see the closure column above), OR
- explicitly `PERMANENTLY-DEFERRED` / `PERMANENTLY-DEFERRED-YAGNI`
  with a one-sentence trigger condition recorded in the master plan
  (grep `PERMANENTLY-DEFERRED` to enumerate), OR
- an explicit v1 scope decision (the B12.1-12.5 heavyweight TSC
  followups, framed as v2-roadmap items).

The single live followup added during this stretch is **D-B38.1**:
wire the sufficient-stats fast path into the 5 aggregators behind
a per-experiment flag. Gated on a consumer surfacing the perf
ceiling.
