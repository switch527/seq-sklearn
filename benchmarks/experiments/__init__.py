"""Experiment drivers (B6 / Phase B5+).

Phase B5 lands the `raw_loss` driver (the standalone B6.1 deliverable).
Phase B6 lands the `ensemble` pairwise-complementarity driver.
Phases B7-B8 add `training_time` and `hpo_uplift`.
"""

from benchmarks.experiments.ensemble import (
    EnsembleExperimentResult,
    PairwiseRow,
    run_ensemble,
)
from benchmarks.experiments.raw_loss import (
    RawLossExperimentResult,
    RunEnvironment,
    build_run_environment,
    run_raw_loss,
)

__all__ = [
    "EnsembleExperimentResult",
    "PairwiseRow",
    "RawLossExperimentResult",
    "RunEnvironment",
    "build_run_environment",
    "run_ensemble",
    "run_raw_loss",
]
