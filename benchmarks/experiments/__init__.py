"""Experiment drivers (B6 / Phase B5+).

Phase B5 lands the `raw_loss` driver (the standalone B6.1 deliverable).
Phase B6 lands the `ensemble` pairwise-complementarity driver.
Phase B7 lands the `training_time` report-only driver over the B5
manifest. Phase B8 adds `hpo_uplift`. Phase B11 adds
`ensemble_lift` (the B6.2.5 deliverable: GBM-only vs GBM+seq
Wilcoxon signed-rank lift).
"""

from benchmarks.experiments.ensemble import (
    EnsembleExperimentResult,
    PairwiseRow,
    run_ensemble,
)
from benchmarks.experiments.ensemble_lift import (
    EnsembleLiftExperimentResult,
    run_ensemble_lift,
)
from benchmarks.experiments.hpo_uplift import (
    HPOUpliftExperimentResult,
    run_hpo_uplift,
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
    "EnsembleLiftExperimentResult",
    "HPOUpliftExperimentResult",
    "PairwiseRow",
    "RawLossExperimentResult",
    "RunEnvironment",
    "TrainingTimeExperimentResult",
    "build_run_environment",
    "run_ensemble",
    "run_ensemble_lift",
    "run_hpo_uplift",
    "run_raw_loss",
    "run_training_time",
]
