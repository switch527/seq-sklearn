"""Leaderboard / report renderers (Phase B5+).

Phase B5 ships the raw-loss leaderboard renderer
(`benchmarks/report/raw_loss.py`). Phase B6 ships the pairwise
ensemble-complementarity renderer (`benchmarks/report/ensemble.py`).
Phase B7 ships the training-time renderer
(`benchmarks/report/training_time.py`). Phase B8 adds the HPO-uplift
renderer; each reads the same manifest produced by the experiment
driver.
"""

from benchmarks.report.ensemble import (
    PairwiseSummary,
    aggregate_pairs,
    render_pairwise_markdown,
)
from benchmarks.report.hpo_uplift import (
    HPOUpliftReport,
    UpliftRow,
    aggregate_hpo_uplift,
    render_hpo_uplift_markdown,
)
from benchmarks.report.raw_loss import (
    LeaderboardEntry,
    rank_by_primary_loss,
    render_leaderboard_markdown,
)
from benchmarks.report.training_time import (
    TrainingTimeSummary,
    aggregate_training_time,
    render_training_time_markdown,
)

__all__ = [
    "HPOUpliftReport",
    "LeaderboardEntry",
    "PairwiseSummary",
    "TrainingTimeSummary",
    "UpliftRow",
    "aggregate_hpo_uplift",
    "aggregate_pairs",
    "aggregate_training_time",
    "rank_by_primary_loss",
    "render_hpo_uplift_markdown",
    "render_leaderboard_markdown",
    "render_pairwise_markdown",
    "render_training_time_markdown",
]
