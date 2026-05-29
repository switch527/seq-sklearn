# B18 design delta: shared CLI-wrapper factory across the 5 `_run_bootstrap_*_rollup` wrappers (D-B16.6)

**Scope**: D-B16.6 collapses the ~5x duplicated body of the
five CLI wrappers in `benchmarks/run.py`
(`_run_bootstrap_rollup`, `_run_bootstrap_pairwise_rollup`,
`_run_bootstrap_training_time_rollup`,
`_run_bootstrap_hpo_uplift_rollup`,
`_run_bootstrap_ensemble_lift_rollup`) into a single
parametrized factory call. Each wrapper shrinks from a ~65-line
body to a thin spec-passing call. The factory carries the
single source of truth for the four-gate cascade (opt-out,
run-manifest presence, run-manifest load, aggregator-success
unlink-stale-sentinel + log; aggregator-failure delete-partial
+ write-sentinel + log).

The behavior is unchanged. The wrappers' public dispatch
points in `_dispatch_kinds` keep their names so the call sites
do not move. The 49 existing wrapper tests across
`test_run_bootstrap_rollup_wrapper.py` (7),
`test_run_bootstrap_pairwise_wrapper.py` (9),
`test_run_bootstrap_training_time_wrapper.py` (9),
`test_run_bootstrap_hpo_uplift_wrapper.py` (11), and
`test_run_bootstrap_ensemble_lift_wrapper.py` (13) all
continue to pass byte-equivalent without modification. The
existing test counts were verified by
`grep -c "^def test_" tests/benchmarks/test_run_bootstrap_*_wrapper.py`.

## Requirements

The grading rubric for every reviewer finding traces back to
one of these.

- **R-B18-1** All five wrappers route through a single factory
  call. Each wrapper's body is one statement (the factory
  invocation with a per-family spec + the per-family
  aggregator). The factory is the single source of truth for
  the four-gate cascade.
- **R-B18-2** No behavior change. The 49 existing wrapper
  tests pass byte-equivalent. The 5 wrapper functions keep
  their names, signatures, and module-public visibility so
  the existing tests' monkeypatch references resolve
  unchanged. Critically, `monkeypatch.setattr(_run_module,
  "aggregate_bootstrap_*_rollup", _boom)` MUST continue to
  intercept the aggregator call inside the factory (closes
  arch-R1-C1 / qa-R1-C3 monkeypatch-seam concern).
- **R-B18-3** The factory is type-safe: per-family static
  parameters (label, opt-out predicate, rollup path helper,
  sentinel path helper) ride on a typed
  `BootstrapWrapperSpec` dataclass. The per-call
  `aggregator` callable is passed separately so name lookup
  happens at call time against the module namespace,
  preserving the existing monkeypatch seam.
- **R-B18-4** The factory is testable in isolation. A new
  test module `tests/benchmarks/test_bootstrap_wrapper_factory.py`
  exercises the factory against a stub spec that records each
  callable invocation. Each branch of the four-gate cascade
  has at least one dedicated test (including the 3 narrow
  except types in Gate 3 and the two stale-sentinel-presence
  states in Gate 4a + the partial-file-presence states in
  Gate 4b).
- **R-B18-5** The factory does not leak per-family details
  into its signature. The five inputs (spec.label,
  spec.is_enabled, spec.rollup_path, spec.sentinel_path, and
  the per-call aggregator) are the entire varying surface;
  the factory accepts no family-discriminator string and no
  per-family branch.
- **R-B18-6** The factory + the `BootstrapWrapperSpec` live
  in a new module `benchmarks/_bootstrap_wrapper.py` so the
  factory tests can import without dragging in the five
  aggregators. The 5 `_BX_SPEC` constants and the 5 thin
  wrappers stay in `run.py` where they belong.
- **R-B18-7** The refactor ships in ONE commit. No staged
  rollout, no behavioral feature flag. Risk is bounded by
  the 49 wrapper tests + ruff + pyright + the new
  factory tests.

## B18.0 What the refactor actually changes

Before (5 functions, each ~65 lines):

```python
def _run_bootstrap_rollup(
    config: BenchmarkConfig,
    *,
    env: RunEnvironment,
    output_root: Path,
) -> None:
    if not is_rollup_enabled(config):
        logger.info("bootstrap_rollup: skipped (...)")
        return
    if not run_manifest_path(output_root).exists():
        logger.warning("bootstrap_rollup: skipped (...)", output_root)
        return
    try:
        manifest = load_run_manifest(output_root)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        logger.warning("bootstrap_rollup: skipped (...)", ...)
        return
    try:
        rows = aggregate_bootstrap_rollup(
            config, output_root=output_root, env=env, manifest=manifest
        )
        stale = aggregator_failed_sentinel_path(output_root)
        if stale.exists():
            stale.unlink(missing_ok=True)
        logger.info("bootstrap_rollup: %d rollup rows written to %s", ...)
    except RawRollupError as exc:
        logger.warning("bootstrap_rollup: aggregator failed (...)", exc)
        path = rollup_path(output_root)
        if path.exists():
            path.unlink(missing_ok=True)
        sentinel = aggregator_failed_sentinel_path(output_root)
        sentinel.write_text(type(exc).__name__, encoding="utf-8")
```

After (5 functions, each ~8 lines):

```python
def _run_bootstrap_rollup(
    config: BenchmarkConfig,
    *,
    env: RunEnvironment,
    output_root: Path,
) -> None:
    """B13 bootstrap-CI rollup step. Runs AFTER raw_loss completes."""
    _run_bootstrap_rollup_via_factory(
        config,
        env=env,
        output_root=output_root,
        spec=_B5_SPEC,
        aggregator=aggregate_bootstrap_rollup,
    )
```

The five `_BX_SPEC` constants are module-level
`BootstrapWrapperSpec` instances that pre-bind the four
static per-family parameters. The per-call aggregator
argument is the load-bearing fix that preserves the
existing monkeypatch seam.

## B18.1 Why the aggregator rides on the per-call argument, not the spec

(arch-R1-C1 / qa-R1-C3 closure pre-empt)

The 14 existing wrapper tests across the 5 wrapper-test files
use `monkeypatch.setattr(_run_module, "aggregate_bootstrap_*_rollup",
_boom)` to inject a `RawRollupError`-raising stub at the
module-attribute level (verified at
`tests/benchmarks/test_run_bootstrap_rollup_wrapper.py:114-120`,
where the comment explicitly documents "the import happens at
module load time and the local name is what the wrapper
calls"). Python resolves bare-name lookups (`aggregate_bootstrap_rollup`)
inside a function body against the module's global namespace
AT CALL TIME, so the test's monkeypatch intercepts the call
correctly today.

If the aggregator were captured into a frozen
`BootstrapWrapperSpec` constant at module-load time
(`_B5_SPEC = BootstrapWrapperSpec(aggregator=
aggregate_bootstrap_rollup, ...)`), the spec would hold a
hard reference to the function object resolved at module-load.
A later `monkeypatch.setattr` against `_run_module` would
rebind the module attribute but NOT the spec's stored
reference; the factory would still call the original
aggregator and all 14 RawRollupError tests would pass
silently for the wrong reason. The seam break would be
invisible at the test-output level but the production Gate 4b
code path would have zero live test coverage post-refactor.

B18 ships the aggregator as a per-call argument to the
factory. The five thin wrappers pass
`aggregator=aggregate_bootstrap_<family>_rollup` at call
time, which resolves the name against the module namespace
on every invocation; `monkeypatch.setattr` keeps working.

## B18.2 The BootstrapWrapperSpec dataclass

```python
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path

from benchmarks.config import BenchmarkConfig


@dataclass(frozen=True)
class BootstrapWrapperSpec:
    """Per-family static parameters for the shared CLI wrapper.

    `label` is the log-message prefix and the family
    identifier in error footnotes. `is_enabled` is the
    family's opt-out predicate from
    `benchmarks/report/bootstrap_*.py`. `rollup_path` and
    `sentinel_path` are the file-locator helpers from
    `benchmarks/bootstrap_manifest.py`.

    Per the arch-R1-I1 closure, this is a frozen dataclass
    (matching the `HPORegistration` precedent in
    `benchmarks/hpo/_base.py`) rather than a pydantic
    BaseModel with `arbitrary_types_allowed=True`. The four
    callable fields and one string have no validation work
    to do; pydantic's machinery buys nothing here.

    The aggregator is intentionally NOT a field on this
    dataclass. It rides on the factory's per-call argument
    so the existing wrapper tests' module-level
    monkeypatches still intercept the call (B18.1).
    """

    label: str
    is_enabled: Callable[[BenchmarkConfig], bool]
    rollup_path: Callable[[Path], Path]
    sentinel_path: Callable[[Path], Path]
```

## B18.3 The factory function

`benchmarks/_bootstrap_wrapper.py` (NEW module, per
arch-R1-I3 + arch-R1-I4 closure: extract the factory + spec
to a helper module so the factory tests can import without
dragging in the 5 aggregators):

```python
"""Shared CLI-wrapper factory for the 5 bootstrap rollup
wrappers in `benchmarks/run.py` (D-B16.6 / B18)."""

import logging
from collections.abc import Callable
from typing import Protocol

from pydantic import ValidationError

from benchmarks.config import BenchmarkConfig
from benchmarks.experiments.raw_loss import RunEnvironment
from benchmarks.report.bootstrap_rollup import RawRollupError
from benchmarks.run_manifest import RunManifest, load_run_manifest, run_manifest_path

logger = logging.getLogger(__name__)


class _Aggregator(Protocol):
    """Aggregator-callable signature shared by all 5 families.

    Verified consistent across the 5 aggregators at
    `benchmarks/report/bootstrap_rollup.py:374-380`,
    `bootstrap_pairwise.py:233-239`,
    `bootstrap_training_time.py:211-217`,
    `bootstrap_hpo_uplift.py:316-322`,
    `bootstrap_ensemble_lift.py:277-283`."""

    def __call__(
        self,
        config: BenchmarkConfig,
        *,
        output_root: Path,
        env: RunEnvironment,
        manifest: RunManifest,
    ) -> list[object]: ...


def run_bootstrap_rollup_via_factory(
    config: BenchmarkConfig,
    *,
    env: RunEnvironment,
    output_root: Path,
    spec: BootstrapWrapperSpec,
    aggregator: _Aggregator,
) -> None:
    """Shared four-gate cascade for bootstrap-CI wrappers.

    Per-family static parameters ride on `spec`; the
    aggregator callable is passed per-call so the existing
    wrapper tests' module-level monkeypatches still
    intercept the aggregator (B18.1).

    Behavior is byte-equivalent to the pre-B18 wrappers.
    """
    label = spec.label
    if not spec.is_enabled(config):
        logger.info(
            "%s: skipped (the corresponding ExperimentSpec opt-in flag is False)",
            label,
        )
        return
    if not run_manifest_path(output_root).exists():
        logger.warning(
            "%s: skipped (run_manifest.json absent at %s)",
            label,
            output_root,
        )
        return
    try:
        manifest = load_run_manifest(output_root)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        logger.warning(
            "%s: skipped (failed to load run_manifest.json: %s: %s)",
            label,
            type(exc).__name__,
            exc,
        )
        return
    try:
        rows = aggregator(
            config, output_root=output_root, env=env, manifest=manifest
        )
        stale = spec.sentinel_path(output_root)
        if stale.exists():
            stale.unlink(missing_ok=True)
        logger.info(
            "%s: %d rollup rows written to %s",
            label,
            len(rows),
            spec.rollup_path(output_root),
        )
    except RawRollupError as exc:
        logger.warning(
            "%s: aggregator failed (%s); deleting any partial "
            "rollup shard. The report will fall back to the std "
            "variant.",
            label,
            exc,
        )
        path = spec.rollup_path(output_root)
        if path.exists():
            path.unlink(missing_ok=True)
        sentinel = spec.sentinel_path(output_root)
        sentinel.write_text(type(exc).__name__, encoding="utf-8")
```

## B18.4 Log-message normalization

The pre-B18 wrappers each carry slightly different log
prefixes ("bootstrap_rollup", "bootstrap_pairwise_rollup",
etc.). The factory keeps the same prefixes by reading them
off `spec.label`. A quick audit shows ZERO existing test
assertions on the Gate-1 log content (
`grep -nE "no .* ExperimentSpec has|skipped \(no " tests/benchmarks/`
returns zero hits; the `caplog` fixture is not used in any
of the 5 wrapper test files). The 49 existing wrapper tests
assert only on rollup-file presence / absence and the
sentinel-file presence / absence, never on log message
content. R-B18-2 holds for the test-observable surface.

The opt-out log change from per-family suffix to the generic
`"the corresponding ExperimentSpec opt-in flag is False"`
is a deliberate normalization; the per-family information
already lives on the family-specific `is_enabled` predicate
name in the call site stack trace.

## B18.5 Per-family spec constants + thin wrappers

Five module-level constants in `run.py`:

```python
from benchmarks._bootstrap_wrapper import (
    BootstrapWrapperSpec,
    run_bootstrap_rollup_via_factory,
)

_B5_SPEC = BootstrapWrapperSpec(
    label="bootstrap_rollup",
    is_enabled=is_rollup_enabled,
    rollup_path=rollup_path,
    sentinel_path=aggregator_failed_sentinel_path,
)

_B6_SPEC = BootstrapWrapperSpec(
    label="bootstrap_pairwise_rollup",
    is_enabled=is_pairwise_rollup_enabled,
    rollup_path=pairwise_rollup_path,
    sentinel_path=pairwise_aggregator_failed_sentinel_path,
)

_B7_SPEC = BootstrapWrapperSpec(
    label="bootstrap_training_time_rollup",
    is_enabled=is_training_time_rollup_enabled,
    rollup_path=training_time_rollup_path,
    sentinel_path=training_time_aggregator_failed_sentinel_path,
)

_B15_SPEC = BootstrapWrapperSpec(
    label="bootstrap_hpo_uplift_rollup",
    is_enabled=is_hpo_uplift_rollup_enabled,
    rollup_path=hpo_uplift_rollup_path,
    sentinel_path=hpo_uplift_aggregator_failed_sentinel_path,
)

_B16_SPEC = BootstrapWrapperSpec(
    label="bootstrap_ensemble_lift_rollup",
    is_enabled=is_ensemble_lift_rollup_enabled,
    rollup_path=ensemble_lift_rollup_path,
    sentinel_path=ensemble_lift_aggregator_failed_sentinel_path,
)
```

The five thin wrappers each pass their spec + their aggregator
to the factory:

```python
def _run_bootstrap_rollup(
    config: BenchmarkConfig, *,
    env: RunEnvironment, output_root: Path,
) -> None:
    """B13 bootstrap-CI rollup step. Runs AFTER raw_loss completes."""
    run_bootstrap_rollup_via_factory(
        config, env=env, output_root=output_root,
        spec=_B5_SPEC,
        aggregator=aggregate_bootstrap_rollup,
    )
```

The `is_*_enabled` predicates and `aggregate_bootstrap_*`
functions are imported at `run.py:34-83` ahead of the spec
constants (R-B18-Risk-4 mitigation anchor).

## B18.6 Test surface

Existing tests:
- 49 wrapper tests across the 5 wrapper test files (counted
  by `grep -c "^def test_" tests/benchmarks/test_run_bootstrap_*_wrapper.py`:
  rollup=7, pairwise=9, training_time=9, hpo_uplift=11,
  ensemble_lift=13). All 49 pass byte-equivalent.

NEW: `tests/benchmarks/test_bootstrap_wrapper_factory.py`
with 9 tests covering every gate-branch:

1. `test_factory_skips_on_opt_out_predicate_false` (Gate 1).
2. `test_factory_skips_when_run_manifest_absent` (Gate 2).
3. `test_factory_skips_when_load_run_manifest_raises_file_not_found_error`
   (Gate 3 arm 1; closes qa-R1-C1).
4. `test_factory_skips_when_load_run_manifest_raises_value_error`
   (Gate 3 arm 2; closes qa-R1-C1).
5. `test_factory_skips_when_load_run_manifest_raises_validation_error`
   (Gate 3 arm 3; closes qa-R1-C1).
6. `test_factory_happy_path_with_stale_sentinel_unlinks_and_writes_rollup`
   (Gate 4a stale-present arm).
7. `test_factory_happy_path_with_no_stale_sentinel_does_not_raise`
   (Gate 4a stale-absent arm; closes qa-R1-C2).
8. `test_factory_raw_rollup_error_with_partial_file_deletes_and_writes_sentinel`
   (Gate 4b partial-present arm).
9. `test_factory_raw_rollup_error_with_no_partial_file_writes_sentinel_only`
   (Gate 4b partial-absent arm; closes qa-R1-I2).

Plus a wire-up parametrize (qa-R1-I1 closure):
10. `test_all_five_wrappers_route_through_factory` parametrized
    over the 5 `(wrapper_function, spec_constant,
    expected_aggregator_name)` triples. Monkeypatches
    `run_bootstrap_rollup_via_factory` to record `(spec,
    aggregator)` pairs; invokes each wrapper; asserts the
    recorded spec is the expected `_BX_SPEC` and the
    recorded `aggregator.__name__` matches.

Plus log-content sanity (qa-R1-N1 closure):
11. `test_factory_gate_1_log_message_includes_spec_label`
    captures `caplog` and asserts the Gate-1 log message
    contains `spec.label`; locks the normalized log shape.

Expected test delta after the refactor:
- Existing tests: 846 to 846 (no change).
- B18-new: 9 gate tests + 1 wire-up parametrize (5 cells) +
  1 log-content = 15 collected items.
- Total: 846 + 15 = 861 expected post-refactor.

## B18.7 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B18-Risk-1 | A family's aggregator does not actually accept the four-arg signature `(config, *, output_root, env, manifest)`. | High | Pre-refactor verification: signatures at `bootstrap_rollup.py:374-380`, `bootstrap_pairwise.py:233-239`, `bootstrap_training_time.py:211-217`, `bootstrap_hpo_uplift.py:316-322`, `bootstrap_ensemble_lift.py:277-283` all match exactly. The `_Aggregator` Protocol locks this contract at the type-checker level. |
| R-B18-Risk-2 | A test asserts on the per-family Gate-1 log suffix and breaks under normalization. | Low | `grep -nE "no .* ExperimentSpec has\|skipped \(no " tests/benchmarks/` returns zero hits. The 49 wrapper tests do not use `caplog` or any other log-content fixture. R-B18-2 holds. |
| R-B18-Risk-3 | A module-load-time `BootstrapWrapperSpec` constant fails to import because a spec callable is not yet bound. | Low | The 4 `is_*_enabled` predicates + 4 path helpers + 1 string passed to each spec are all imported at `run.py:34-83`, well before the spec-construction block. The aggregator callables are NOT bound on the spec (B18.1); they are looked up at wrapper-call time. No circular-import risk introduced. |
| R-B18-Risk-4 | The factory's `_Aggregator` Protocol is too loose and a future aggregator that drops a kwarg passes type-check yet crashes at runtime. | Low | The Protocol's `__call__` signature pins the exact kwargs (`output_root`, `env`, `manifest`); pyright will reject a callable that drops one. Runtime tests (9 gate tests + 49 existing wrapper tests) catch any mismatch. |
| R-B18-Risk-5 | A future B19 phase adds a 6th wrapper but bypasses the factory by writing inline duplication. | Low | The wire-up test (`test_all_five_wrappers_route_through_factory`) only covers the 5 existing wrappers. The factory + spec dataclass live in their own module (`_bootstrap_wrapper.py`) so a future wrapper that re-implements the cascade is visible in the run.py diff. Document the pattern at the top of the bootstrap-wrappers section in run.py with a `NOTE: future bootstrap-rollup wrappers should call `run_bootstrap_rollup_via_factory`...` line. |
| R-B18-Risk-6 | The monkeypatch seam break described in B18.1 ships accidentally because the aggregator is captured on the spec. | Critical | The factory tests' wire-up parametrize asserts each wrapper calls the factory with its aggregator AS A PER-CALL ARGUMENT, not via the spec. The B18.5 spec constants do NOT carry the aggregator field; the dataclass at B18.2 does not declare it. The existing 49 wrapper tests' monkeypatch interception is the integration-level guarantee. |

## B18.8 Implementation outline

1. **New module**: create `benchmarks/_bootstrap_wrapper.py`
   with the `BootstrapWrapperSpec` dataclass, the
   `_Aggregator` Protocol, and the
   `run_bootstrap_rollup_via_factory` function. Verify ruff
   + pyright pass on the new file in isolation.
2. **Wire run.py**: add 5 `_BX_SPEC` constants to `run.py`
   importing `BootstrapWrapperSpec` and the factory from the
   new module. Verify the imports resolve cleanly.
3. **Thin-out**: rewrite each of the 5 wrappers as a thin
   factory call. Keep the per-family docstrings as the
   family contract reference.
4. **Verify existing tests**: ruff + pyright + the 5 wrapper
   test files (49 tests) pass byte-equivalent. This is
   R-B18-2's load-bearing verification step.
5. **B18-new tests**: add `tests/benchmarks/test_bootstrap_wrapper_factory.py`
   with the 9 gate tests + 5-cell wire-up parametrize + 1
   log-content test = 15 collected items.
6. **Final verify**: the broader 846-test suite passes; 861
   total post-refactor. Run `grep -nE
   "if not is_rollup_enabled|if not run_manifest_path"
   benchmarks/run.py` and confirm zero hits in the wrapper
   bodies (the cascade lives only in the factory).

## Addressed

R1 swarm: architecture-reviewer (1C / 4I / 2N
REQUEST_CHANGES), qa-test-coverage (3C / 3I / 1N
REQUEST_CHANGES), style-reviewer (0C / 0I / 1N APPROVE).
Deduplicated total: 3 CRITICAL, 5 IMPROVEMENT, 3 NITPICK
(plus the accepted house-style bold-density NIT). Closures:

- **arch-R1-C1 / qa-R1-C3** (the monkeypatch seam break: if
  the aggregator is captured on a frozen `BootstrapWrapperSpec`
  at module-load time, the existing 14 wrapper-test
  monkeypatches against `_run_module.aggregate_bootstrap_*_rollup`
  are silenced and Gate 4b loses live coverage): the spec
  no longer carries the aggregator field. The factory takes
  the aggregator as a per-call keyword argument; each thin
  wrapper passes its family's aggregator at call time so
  Python's late-binding name lookup against the module
  namespace preserves the existing monkeypatch interception.
  B18.1 explicitly documents the seam reason inline so a
  future maintainer reading the design understands why the
  aggregator does NOT ride on the spec.
- **qa-R1-C1** (Gate 3's 3-arm except tuple covered by a
  single corrupt-JSON test): B18.6 splits into three
  named tests (`_file_not_found_error`,
  `_value_error`, `_validation_error`).
- **qa-R1-C2** (Gate 4a stale-sentinel-absent branch
  untested at the factory level): B18.6 adds
  `test_factory_happy_path_with_no_stale_sentinel_does_not_raise`.
- **arch-R1-I1** (pydantic `BaseModel` with
  `arbitrary_types_allowed=True` is overkill for a 5-field
  record where 4 fields are callables): switched to
  `@dataclass(frozen=True)` per the `HPORegistration`
  precedent at `benchmarks/hpo/_base.py`. B18.2 documents
  the choice inline.
- **arch-R1-I2** (`Callable[..., list[Any]]` discards the
  keyword-only-arg shape): replaced with the explicit
  `_Aggregator` Protocol at B18.3 pinning the exact kwargs.
- **arch-R1-I3** (placement decision deferred without a
  threshold): pre-empted by arch-R1-I4 closure. The factory
  + dataclass + Protocol live in their own module
  `benchmarks/_bootstrap_wrapper.py` from day one. No
  threshold is needed because the extraction happens now.
- **arch-R1-I4** (the proposed module-co-location made the
  factory tests drag in the 5 aggregators): the factory +
  spec + Protocol now live in `benchmarks/_bootstrap_wrapper.py`.
  The factory tests import only from this module; the 5
  aggregator imports stay isolated in `run.py`. R-B18-6 is
  the requirement that codifies this split.
- **qa-R1-I1** (no test asserts each wrapper actually calls
  the factory): B18.6 adds the
  `test_all_five_wrappers_route_through_factory` parametrize
  over the 5 `(wrapper, spec, expected_aggregator_name)`
  triples.
- **qa-R1-I2** (Gate 4b's no-partial-file branch untested):
  B18.6 adds
  `test_factory_raw_rollup_error_with_no_partial_file_writes_sentinel_only`.
- **qa-R1-I3** (test count math: 47 vs actual 49): fixed
  throughout. The 7+9+9+11+13 = 49 count is verified by
  `grep -c "^def test_" tests/benchmarks/test_run_bootstrap_*_wrapper.py`.
  The expected post-refactor total is 846 + 15 = 861.
- **arch-R1-N1** (R-B18-Risk-4 understated the no-circular-
  import analysis): the Risks table now states "imported at
  `run.py:34-83`, well before the spec-construction block".
- **arch-R1-N2** ("47 existing wrapper tests" claim
  unsourced): the introduction now cites the grep command
  AND the per-file breakdown (7+9+9+11+13 = 49).
- **qa-R1-N1** (no test asserts the new Gate-1 log message):
  B18.6 adds `test_factory_gate_1_log_message_includes_spec_label`
  using `caplog`.
- **style-R1-N1** (bold density above the 200-word
  threshold): NOT changed. All 14 bold occurrences are
  structural labels (one `**Scope**`, six `**R-B18-N**`,
  seven implementation step labels); the style reviewer
  accepted the pattern as house style.

## Deferred

None at R1.
