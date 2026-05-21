"""Leaderboard / report renderers (Phase B5+).

Phase B5 ships the raw-loss leaderboard renderer
(`benchmarks/report/raw_loss.py`). Phase B6 ships the pairwise
ensemble-complementarity renderer (`benchmarks/report/ensemble.py`).
Phases B7-B8 add the training-time and HPO-uplift renderers; each
reads the same manifest produced by the experiment driver.
"""

from benchmarks.report.ensemble import (
    PairwiseSummary,
    aggregate_pairs,
    render_pairwise_markdown,
)
from benchmarks.report.raw_loss import (
    LeaderboardEntry,
    rank_by_primary_loss,
    render_leaderboard_markdown,
)

__all__ = [
    "LeaderboardEntry",
    "PairwiseSummary",
    "aggregate_pairs",
    "rank_by_primary_loss",
    "render_leaderboard_markdown",
    "render_pairwise_markdown",
]
