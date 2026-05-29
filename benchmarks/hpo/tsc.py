"""Classical-TSC family HPO search space (Phase B12).

Wraps a per-trial Optuna sampler for the three aeon classifiers
registered by ``benchmarks/adapters/tsc.py`` (ROCKET, MultiRocket,
Catch22). Each classifier has its own native parameter names; the
sampler dispatches on the model spec's name and emits the
adapter-friendly kwargs the underlying aeon estimator accepts.

Design alignment:

- B6.4.0 (search-space parity disclosure): every tuned row
  records ``hpo_search_space_size`` from the registered
  ``HPOSpace``. The size below names the maximum dimensionality
  across the three classifiers (Catch22 = 2, MultiRocket = 2,
  ROCKET = 1); the report footnotes the per-classifier breakdown
  so the parity claim is honest. GBM declares 6 per
  ``benchmarks/hpo/gbm.py:_GBM_SEARCH_SPACE_SIZE``; the
  asymmetry is reported rather than hidden.
- The TSC family does not consume any per-trial budget hooks
  beyond the profile-tier envelope at
  ``benchmarks/experiments/hpo_uplift.py:_HPO_BUDGETS_BY_PROFILE``
  (smoke=0 / standard=25 / full=100). No per-family override.
"""

import logging
from typing import Any

import optuna

from benchmarks.config import DatasetSpec, ModelSpec
from benchmarks.hpo._base import HPOSpace, register_hpo_space

__all__ = ["sample_tsc_hyperparameters"]


logger = logging.getLogger(__name__)


def _sample_rocket_hyperparameters(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
    return {
        "num_kernels": int(trial.suggest_int("num_kernels", 500, 20_000, log=True)),
    }


def _sample_multirocket_hyperparameters(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
    return {
        "num_kernels": int(trial.suggest_int("num_kernels", 500, 20_000, log=True)),
        "max_dilations_per_kernel": int(trial.suggest_int("max_dilations_per_kernel", 8, 64)),
    }


def _sample_catch22_hyperparameters(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
    return {
        "outlier_norm": bool(trial.suggest_categorical("outlier_norm", [False, True])),
        "replace_nans": bool(trial.suggest_categorical("replace_nans", [True, False])),
    }


_TSC_SAMPLERS: dict[str, Any] = {
    "rocket_classifier": _sample_rocket_hyperparameters,
    "multirocket_classifier": _sample_multirocket_hyperparameters,
    "catch22_classifier": _sample_catch22_hyperparameters,
}


def sample_tsc_hyperparameters(
    trial: optuna.trial.BaseTrial,
    model_spec: ModelSpec,
    dataset_spec: DatasetSpec,
) -> dict[str, Any]:
    """Sample one trial's hyperparameters for a TSC adapter.

    Dispatches on ``model_spec.name`` to the classifier-specific
    sampler. ``dataset_spec`` is currently unused (the TSC
    sampling space is task-agnostic at v1); kept for protocol
    symmetry with the other family samplers.

    Raises:
        ValueError: ``model_spec.family`` is not ``tsc``.
        KeyError: ``model_spec.name`` is not a registered TSC
            adapter.
    """
    del dataset_spec
    if model_spec.family != "tsc":
        raise ValueError(
            f"sample_tsc_hyperparameters: model_spec.family must be "
            f"'tsc', got {model_spec.family!r}"
        )
    sampler = _TSC_SAMPLERS.get(model_spec.name)
    if sampler is None:
        raise KeyError(
            f"sample_tsc_hyperparameters: model {model_spec.name!r} has "
            f"no sampler entry; registered: {sorted(_TSC_SAMPLERS)}"
        )
    return sampler(trial)


# Per-family search-space dimensionality (B6.4.0 disclosure):
# max across the three classifiers. ROCKET = 1 (num_kernels),
# MultiRocket = 2 (num_kernels + max_dilations_per_kernel), Catch22
# = 2 (outlier_norm + replace_nans). The B6.4.0 report quotes ONE
# int per family + the per-classifier breakdown in the disclosure
# footnote.
_TSC_SEARCH_SPACE_SIZE = 2

_TSC_SPACE = HPOSpace(
    family="tsc",
    search_space_size=_TSC_SEARCH_SPACE_SIZE,
    description=(
        "Classical-TSC family search. Per-classifier "
        "dimensionality: ROCKET = 1 (num_kernels, log-uniform 500 "
        "to 20,000); MultiRocket = 2 (num_kernels + "
        "max_dilations_per_kernel); Catch22 = 2 (outlier_norm + "
        "replace_nans). Reported size = max across classifiers. "
        "Materially smaller than the GBM family's 6 dims; the "
        "asymmetry is documented in the B6.4.0 parity disclosure "
        "footnote rather than hidden."
    ),
)
register_hpo_space(_TSC_SPACE, sample_tsc_hyperparameters)
