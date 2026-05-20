"""Leaderboard / report renderers (Phase B5+).

Phase B5 ships the raw-loss leaderboard renderer
(`benchmarks/report/raw_loss.py`). Phases B6-B8 add the ensemble,
training-time, and HPO-uplift renderers; each reads the same
manifest produced by the experiment driver.
"""

from benchmarks.report.raw_loss import (
    LeaderboardEntry,
    rank_by_primary_loss,
    render_leaderboard_markdown,
)

__all__ = [
    "LeaderboardEntry",
    "rank_by_primary_loss",
    "render_leaderboard_markdown",
]
