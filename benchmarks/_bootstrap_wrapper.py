"""Shared CLI-wrapper factory for the 5 bootstrap rollup
wrappers in `benchmarks/run.py` (D-B16.6 / B18).

The five families (B5 raw_loss, B6 pairwise, B7 training_time,
B15 hpo_uplift, B16 ensemble_lift) each have a CLI wrapper
that runs the same four-gate cascade:

1. Opt-out predicate False -> skip + info log.
2. `run_manifest.json` absent -> skip + warning log.
3. `load_run_manifest` raises FileNotFoundError, ValueError,
   or pydantic ValidationError -> skip + warning log.
4. Aggregator call:
   - on success: unlink stale failure sentinel + info log.
   - on `RawRollupError`: delete partial rollup shard +
     write failure sentinel + warning log.

The factory is the single source of truth for the cascade.
Per-family static parameters (label, opt-out predicate,
rollup-path helper, failure-sentinel-path helper) ride on a
`BootstrapWrapperSpec` frozen dataclass. The per-family
aggregator callable is passed as a per-call keyword argument
(see the B18.1 docstring on `BootstrapWrapperSpec`).
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from benchmarks.config import BenchmarkConfig
from benchmarks.experiments.raw_loss import RunEnvironment
from benchmarks.report.bootstrap_rollup import RawRollupError
from benchmarks.run_manifest import RunManifest, load_run_manifest, run_manifest_path

logger = logging.getLogger(__name__)

__all__ = [
    "Aggregator",
    "BootstrapWrapperSpec",
    "run_bootstrap_rollup_via_factory",
]


@dataclass(frozen=True)
class BootstrapWrapperSpec:
    """Per-family static parameters for the shared CLI wrapper.

    `label` is the log-message prefix (and family identifier in
    error footnotes). `is_enabled` is the family's opt-out
    predicate from `benchmarks/report/bootstrap_*.py`.
    `rollup_path` and `sentinel_path` are the file-locator
    helpers from `benchmarks/bootstrap_manifest.py`.

    Per the arch-R1-I1 closure (B18 design), this is a frozen
    dataclass rather than a pydantic BaseModel with
    `arbitrary_types_allowed=True`. The four callable fields
    and one string have no validation work to do; pydantic's
    machinery buys nothing here.

    The aggregator is intentionally NOT a field on this
    dataclass. It rides on the factory's per-call argument so
    the existing wrapper tests' module-level monkeypatches
    against `benchmarks.run.aggregate_bootstrap_*_rollup`
    still intercept the aggregator call. Capturing the
    aggregator on a frozen dataclass at module-load time would
    silently break the existing 14 RawRollupError test
    monkeypatches (the spec would hold a hard reference to
    the function object resolved at import time; later
    `monkeypatch.setattr` against the module attribute would
    not affect the spec's stored reference).
    """

    label: str
    is_enabled: Callable[[BenchmarkConfig], bool]
    rollup_path: Callable[[Path], Path]
    sentinel_path: Callable[[Path], Path]


class Aggregator(Protocol):
    """Aggregator-callable signature shared by all 5 families.

    Verified consistent at:
    - `benchmarks/report/bootstrap_rollup.py:374-380`
    - `benchmarks/report/bootstrap_pairwise.py:233-239`
    - `benchmarks/report/bootstrap_training_time.py:211-217`
    - `benchmarks/report/bootstrap_hpo_uplift.py:316-322`
    - `benchmarks/report/bootstrap_ensemble_lift.py:277-283`
    """

    def __call__(
        self,
        config: BenchmarkConfig,
        *,
        output_root: Path,
        env: RunEnvironment,
        manifest: RunManifest,
    ) -> list[Any]: ...


def run_bootstrap_rollup_via_factory(
    config: BenchmarkConfig,
    *,
    env: RunEnvironment,
    output_root: Path,
    spec: BootstrapWrapperSpec,
    aggregator: Aggregator,
) -> None:
    """Shared four-gate cascade for bootstrap-CI wrappers.

    Per-family static parameters ride on `spec`. The
    `aggregator` callable is passed per-call so the existing
    wrapper tests' module-level monkeypatches still intercept
    the aggregator (see the `BootstrapWrapperSpec` docstring).

    Behavior is byte-equivalent to the pre-B18 wrappers in
    `benchmarks/run.py`.
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
        rows = aggregator(config, output_root=output_root, env=env, manifest=manifest)
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
