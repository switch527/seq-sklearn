"""GBM family HPO search space (Phase B10).

Wraps a per-trial Optuna sampler for the three GBM libraries
registered by `benchmarks/adapters/gbm.py` (LightGBM, XGBoost,
CatBoost). Each library has its own native parameter names, so
the sampler dispatches on the model spec's name and emits the
adapter-friendly kwargs the underlying sklearn-API estimator
accepts.

Design alignment:

- B6.4.0 (search-space parity disclosure): every tuned row
  records `hpo_search_space_size` from the registered `HPOSpace`.
  The size below names the dimensionality of the shared sampling
  surface (n_estimators / learning_rate / depth / two
  regularization params + one library-specific knob). v1 does
  NOT normalize across libraries; the asymmetry is reported
  rather than hidden.
- Adapter-locked-keys carryover: the seq_sklearn adapter pins
  `task_type` / `tabular_config` / `scheduler` / `verbose` via
  the locked-keys guard. The GBM adapter does NOT lock any
  hyperparameter names; the sampler returns library-native
  kwargs that the GBM adapter's `_estimator_factory` consumes
  directly.
"""

import logging
from typing import Any

import optuna

from benchmarks.config import DatasetSpec, ModelSpec
from benchmarks.hpo._base import HPOSpace, register_hpo_space

__all__ = ["sample_gbm_hyperparameters"]


logger = logging.getLogger(__name__)


# Library-specific sampler dispatch. Each function emits the
# kwargs the matching estimator factory in
# `benchmarks/adapters/gbm.py` consumes. The sampled kwargs
# include only the library's STABLE search surface (n_estimators
# / learning_rate / depth + two regularization knobs + one
# library-specific knob) so the search dimensionality is
# comparable across libraries even though the underlying
# parameter names are not.


def _sample_lightgbm_hyperparameters(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
    return {
        "n_estimators": int(trial.suggest_int("n_estimators", 50, 500)),
        "learning_rate": float(trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)),
        "num_leaves": int(trial.suggest_int("num_leaves", 15, 255)),
        "min_child_samples": int(trial.suggest_int("min_child_samples", 5, 100)),
        "reg_alpha": float(trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True)),
        "reg_lambda": float(trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True)),
    }


def _sample_xgboost_hyperparameters(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
    return {
        "n_estimators": int(trial.suggest_int("n_estimators", 50, 500)),
        "learning_rate": float(trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)),
        "max_depth": int(trial.suggest_int("max_depth", 3, 12)),
        "min_child_weight": float(trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True)),
        "reg_alpha": float(trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True)),
        "reg_lambda": float(trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)),
    }


def _sample_catboost_hyperparameters(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
    return {
        "iterations": int(trial.suggest_int("iterations", 50, 500)),
        "learning_rate": float(trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)),
        "depth": int(trial.suggest_int("depth", 3, 10)),
        "l2_leaf_reg": float(trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True)),
        "random_strength": float(trial.suggest_float("random_strength", 1e-3, 10.0, log=True)),
        # B10 R1 arch-CRIT-1 close: align CatBoost's sampled
        # dimensionality with LightGBM + XGBoost so the B6.4.0
        # disclosure `_GBM_SEARCH_SPACE_SIZE = 6` is accurate per
        # cell, not per family. `bagging_temperature` controls the
        # subsampling stochasticity (CatBoost's analogue of the
        # other libraries' `subsample` knob).
        "bagging_temperature": float(trial.suggest_float("bagging_temperature", 0.0, 1.0)),
    }


_LIBRARY_SAMPLERS: dict[str, Any] = {
    "lightgbm_classifier": _sample_lightgbm_hyperparameters,
    "lightgbm_regressor": _sample_lightgbm_hyperparameters,
    "xgboost_classifier": _sample_xgboost_hyperparameters,
    "xgboost_regressor": _sample_xgboost_hyperparameters,
    "catboost_classifier": _sample_catboost_hyperparameters,
    "catboost_regressor": _sample_catboost_hyperparameters,
}


def sample_gbm_hyperparameters(
    trial: optuna.trial.BaseTrial,
    model_spec: ModelSpec,
    dataset_spec: DatasetSpec,
) -> dict[str, Any]:
    """Sample one trial's hyperparameters for a GBM adapter.

    Dispatches on `model_spec.name` to the library-specific
    sampler. `dataset_spec` is currently unused (the GBM
    sampling space is task-agnostic; the adapter's
    `task_types` gate handles the binary-vs-regression split via
    the registered ModelSpec); kept for protocol symmetry with
    `sample_seq_sklearn_hyperparameters`.

    Raises:
        ValueError: `model_spec.family` is not `gbm`.
        KeyError: `model_spec.name` is not a registered GBM
            adapter (a B10-followup adapter hasn't wired its
            sampler entry here yet).
    """
    del dataset_spec
    if model_spec.family != "gbm":
        raise ValueError(
            f"sample_gbm_hyperparameters: model_spec.family must be "
            f"'gbm', got {model_spec.family!r}"
        )
    sampler = _LIBRARY_SAMPLERS.get(model_spec.name)
    if sampler is None:
        raise KeyError(
            f"sample_gbm_hyperparameters: model {model_spec.name!r} has "
            f"no sampler entry; registered: {sorted(_LIBRARY_SAMPLERS)}"
        )
    return sampler(trial)


# Library's STABLE-field search-space dimensionality (B6.4.0
# disclosure). Six search dims per library: n_estimators (or
# `iterations` for CatBoost), learning_rate, a depth-like knob, a
# min-child-* knob, and two regularization terms (L1+L2 for
# LightGBM/XGBoost; `random_strength` + `bagging_temperature` for
# CatBoost). The size matches the disclosed value on every
# library after the R1 CatBoost bump.
_GBM_SEARCH_SPACE_SIZE = 6

_GBM_SPACE = HPOSpace(
    family="gbm",
    search_space_size=_GBM_SEARCH_SPACE_SIZE,
    description=(
        "GBM family search (six dims per library): n_estimators "
        "or iterations (CatBoost), learning_rate (log-uniform), a "
        "depth-like knob (num_leaves for LightGBM, max_depth for "
        "XGBoost, depth for CatBoost), a min-child-* knob, and "
        "two regularization terms (reg_alpha + reg_lambda for "
        "LightGBM/XGBoost; l2_leaf_reg + random_strength for "
        "CatBoost). CatBoost additionally samples "
        "bagging_temperature as its subsampling-stochasticity "
        "analogue of the other libraries' subsample knob. "
        "Library-specific parameter names are sampled directly; "
        "the adapter's hyperparameters dict passes them through "
        "to the sklearn-API estimator."
    ),
)
register_hpo_space(_GBM_SPACE, sample_gbm_hyperparameters)
