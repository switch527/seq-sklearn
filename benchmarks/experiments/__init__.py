"""Experiment drivers (B6 / Phase B5+).

Phase B5 lands the `raw_loss` driver (the standalone B6.1 deliverable).
Phase B6 lands the `ensemble` pairwise-complementarity driver.
Phase B7 lands the `training_time` report-only driver over the B5
manifest. Phase B8 adds `hpo_uplift`.
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
from benchmarks.experiments.training_time import (
    TrainingTimeExperimentResult,
    run_training_time,
)

__all__ = [
    "EnsembleExperimentResult",
    "PairwiseRow",
    "RawLossExperimentResult",
    "RunEnvironment",
    "TrainingTimeExperimentResult",
    "build_run_environment",
    "run_ensemble",
    "run_raw_loss",
    "run_training_time",
]
