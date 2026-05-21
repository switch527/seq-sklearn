"""Phase B7 training-time experiment driver (B6.3).

B6.3 reports per-(dataset, model) fit wall-clock + memory peaks at
fixed config. The data is already captured by the B5 raw-loss
driver (`measure_fit` writes `wall_seconds`, `peak_rss_bytes`,
`peak_cuda_bytes` onto every `ResultRow`), so B7 is a report-only
pass over the existing manifest: no new shard layout, no extra
adapter calls.

`run_training_time` reads the B5 manifest under `output_root`,
aggregates per (dataset, model, hardware_tier), writes
`output_root/training_time.md`, and returns a structured summary
the CLI logs. Resumability is inherited from B5 (no per-cell work
to resume here); a fresh invocation re-renders the report from
the current manifest snapshot.

B7-followup deferrals (documented in the impl plan):

- Scaling-curve plot: design B6.3 names "reported against dataset
  size so the scaling curve is visible". The B5 manifest does not
  carry a per-row dataset-size column today; the B7-followup
  back-ports the count after the schema is extended.
- Repeated-fit timing: B5 fits exactly once per cell. A B7-
  followup with N-rep fit timing (for stable wall-clock) is
  available behind a profile flag.
- Manifest-empty edge: `run_training_time` requires the B5
  manifest to exist; it does NOT auto-run `raw_loss` first. The
  CLI is expected to pass the same `output_root` an earlier
  `raw_loss` invocation produced.
"""

import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments.raw_loss import RunEnvironment
from benchmarks.manifest import load_run
from benchmarks.report.training_time import (
    TrainingTimeSummary,
    aggregate_training_time,
    render_training_time_markdown,
)

logger = logging.getLogger(__name__)


_TRAINING_TIME_REPORT_FILENAME = "training_time.md"


class TrainingTimeExperimentResult(BaseModel):
    """Summary returned by `run_training_time`.

    Counts the (dataset, model, hardware_tier) groups discovered in
    the B5 manifest and how many of them had at least one OK cell.
    The full per-group table lives in
    `output_root/training_time.md`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    groups_evaluated: int  # groups with at least one OK cell
    groups_fully_skipped: int  # groups where every cell was skipped
    report_path: str  # absolute path of the rendered Markdown


def _resolve_training_time_seeds(
    experiments: list[ExperimentSpec],
) -> tuple[int, ...]:
    """Pull seeds from the `training_time` experiment spec(s).

    Not used at the per-cell level (the data is already in the B5
    manifest), but the helper enforces the contract that the config
    declares the experiment kind. Mirrors `_resolve_seeds` in the
    raw_loss + ensemble drivers.

    Raises:
        ValueError: no `training_time` experiment is configured.
    """
    seeds: set[int] = set()
    for spec in experiments:
        if spec.kind == "training_time":
            seeds.update(spec.seeds)
    if not seeds:
        raise ValueError(
            "run_training_time: BenchmarkConfig declares no "
            "training_time experiment; add ExperimentSpec(kind="
            "'training_time') to the config"
        )
    return tuple(sorted(seeds))


def run_training_time(
    config: BenchmarkConfig,
    *,
    output_root: Path,
    env: RunEnvironment,
) -> TrainingTimeExperimentResult:
    """Render the per-(dataset, model, hardware_tier) training-time
    report from an existing B5 manifest under `output_root`.

    Raises:
        ValueError: no `training_time` experiment in the config, or
            the B5 manifest is empty / missing.
    """
    _resolve_training_time_seeds(list(config.experiments))
    manifest = load_run(output_root)
    if manifest.empty:
        raise ValueError(
            f"run_training_time: B5 manifest at {output_root} is "
            f"empty; run `--experiment=raw_loss` first"
        )

    summaries: list[TrainingTimeSummary] = aggregate_training_time(manifest)
    groups_evaluated = sum(1 for s in summaries if s.n_cells_evaluated > 0)
    groups_fully_skipped = sum(1 for s in summaries if s.n_cells_evaluated == 0)

    md = render_training_time_markdown(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / _TRAINING_TIME_REPORT_FILENAME
    report_path.write_text(md, encoding="utf-8")

    summary = TrainingTimeExperimentResult(
        run_id=env.run_id,
        groups_evaluated=groups_evaluated,
        groups_fully_skipped=groups_fully_skipped,
        report_path=str(report_path),
    )
    logger.info("run_training_time: complete: %s", summary.model_dump())
    return summary


__all__ = [
    "TrainingTimeExperimentResult",
    "run_training_time",
]
