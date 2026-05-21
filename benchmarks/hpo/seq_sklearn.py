"""seq_sklearn family HPO search space (Phase B8).

Wraps the library's `seq_sklearn.suggest_params` so the harness
gets a per-trial kwargs dict ready to inject into the registered
adapter via its `hyperparameters=` channel.

The library's `suggest_params` requires a `base` config carrying
the required non-search fields (TFT's `tabular_config` in
particular). This module builds that base from the harness's
`DatasetSpec`, calls `suggest_params`, and converts the sampled
nested sub-configs (`loss`, `sampler`, `optimizer`) to their
sklearn-Params equivalents the adapter expects.

Adapter contract carryover (`benchmarks/adapters/seq_sklearn.py`
`_LOCKED_HYPERPARAMETER_KEYS`): the adapter rejects
`hyperparameters` overlap with `task_type`, `tabular_config`,
`scheduler`, `verbose`. The wrapper strips those before returning
because they are bound from the spec (`task_type` /
`tabular_config`) or wired to the harness default (`scheduler`).
`advanced` is empty in v1; we drop it for symmetry with the
non-search field exclusion in `suggest_params`.

B6.4.0 disclosure: the reported `search_space_size` is the
library's STABLE-field dimensionality (count of independent
suggest_params calls in the default search). The number reports
the dimensionality; the actual cardinality depends on conditional
cells (loss legality per task_type, head divisors per hidden_size)
and is intentionally not normalized cross-family per design.
"""

import logging
from typing import Any, cast

import optuna

from benchmarks.adapters.seq_sklearn import _build_tabular_config
from benchmarks.config import DatasetSpec, ModelSpec
from benchmarks.hpo._base import HPOSpace, register_hpo_space
from seq_sklearn import (
    LossParams,
    OptimizerParams,
    SamplerParams,
    TFTClassifier,
    TFTRegressor,
    suggest_params,
)
from seq_sklearn.config.loss import LossConfig
from seq_sklearn.config.tft import TFTConfig
from seq_sklearn.models._classifier import BaseSequenceClassifier
from seq_sklearn.models._regressor import BaseSequenceRegressor

__all__ = ["sample_seq_sklearn_hyperparameters"]


logger = logging.getLogger(__name__)


# Library's STABLE-field search space size (B6.4.0 disclosure):
# - F5-governed quartet: task_type, loss.strategy, sampler.strategy,
#   calibration_strategy (4)
# - optimizer: learning_rate, weight_decay (2)
# - TFT model shape: hidden_size, attention_heads, dropout,
#   variable_selection_dropout, prediction_readout (5)
# Total: 11.
_SEQ_SKLEARN_SEARCH_SPACE_SIZE = 11


_MODEL_CLASS_BY_NAME: dict[str, type[BaseSequenceClassifier | BaseSequenceRegressor]] = {
    "tft_classifier": TFTClassifier,
    "tft_regressor": TFTRegressor,
}


# Per-task placeholder loss strategy for the `base` config the
# library's `suggest_params` requires. The sampler always overrides
# `loss.strategy` from the F5-legal set for the sampled task_type,
# so these placeholders are read only by the pydantic constructor
# at `base` build time.
_PLACEHOLDER_LOSS_BY_TASK: dict[str, str] = {
    "binary": "cross_entropy",
    "multiclass": "cross_entropy",
    "regression_point": "mse",
    "regression_quantile": "pinball",
}


def _placeholder_base(dataset_spec: DatasetSpec) -> TFTConfig:
    """Build a minimal valid `TFTConfig` to pass as `base=` to
    `suggest_params`.

    Only the required non-search fields (`tabular_config`,
    `task_type`, `loss`) are populated; every other field uses its
    pydantic default. The placeholder values are overridden by
    `suggest_params` before the returned config is consumed.
    """
    tab = _build_tabular_config(dataset_spec).to_pydantic()
    placeholder_loss = _PLACEHOLDER_LOSS_BY_TASK.get(dataset_spec.task_type, "cross_entropy")
    kwargs: dict[str, Any] = {
        "task_type": dataset_spec.task_type,
        "tabular_config": tab,
        "loss": LossConfig(strategy=placeholder_loss),  # type: ignore[arg-type]
    }
    # `regression_quantile` task requires `quantiles` on the parent
    # validator (non-quantile tasks reject it). suggest_params
    # forces the value either way; provide a sensible placeholder
    # so the `base` itself validates.
    if dataset_spec.task_type == "regression_quantile":
        kwargs["quantiles"] = (0.1, 0.5, 0.9)
    return TFTConfig(**kwargs)


def sample_seq_sklearn_hyperparameters(
    trial: optuna.trial.BaseTrial,
    model_spec: ModelSpec,
    dataset_spec: DatasetSpec,
) -> dict[str, Any]:
    """Sample one trial's hyperparameters for a seq_sklearn model.

    Pins `task_type` and `tabular_config` to `dataset_spec` so the
    harness's fixed-task / fixed-tabular-config contract holds; the
    F5-governed loss / sampler / calibration cell is still sampled
    by the library. Returns adapter-ready kwargs (sklearn-Params
    sub-configs, the locked-keys stripped).

    Raises:
        ValueError: `model_spec.family` is not `seq_sklearn`.
        KeyError: `model_spec.name` is not in `_MODEL_CLASS_BY_NAME`
            (a newly-registered seq_sklearn adapter hasn't been
            wired here yet).
    """
    if model_spec.family != "seq_sklearn":
        raise ValueError(
            f"sample_seq_sklearn_hyperparameters: model_spec.family must "
            f"be 'seq_sklearn', got {model_spec.family!r}"
        )
    model_class = _MODEL_CLASS_BY_NAME.get(model_spec.name)
    if model_class is None:
        raise KeyError(
            f"sample_seq_sklearn_hyperparameters: model {model_spec.name!r} "
            f"has no class entry; registered: {sorted(_MODEL_CLASS_BY_NAME)}"
        )
    base = _placeholder_base(dataset_spec)
    sampled = cast(TFTConfig, suggest_params(trial, model_class, base=base))
    kwargs: dict[str, Any] = {
        # F5-sampled sub-configs converted to the sklearn-Params
        # form the adapter forwards to the estimator.
        "loss": LossParams(**sampled.loss.model_dump()),
        "sampler": SamplerParams(**sampled.sampler.model_dump()),
        "optimizer": OptimizerParams(**sampled.optimizer.model_dump()),
        "calibration_strategy": sampled.calibration_strategy,
        # TFT model-shape fields.
        "hidden_size": sampled.hidden_size,
        "attention_heads": sampled.attention_heads,
        "dropout": sampled.dropout,
        "variable_selection_dropout": sampled.variable_selection_dropout,
        "prediction_readout": sampled.prediction_readout,
    }
    if dataset_spec.task_type == "regression_quantile":
        kwargs["quantiles"] = sampled.quantiles
    return kwargs


_SEQ_SKLEARN_SPACE = HPOSpace(
    family="seq_sklearn",
    search_space_size=_SEQ_SKLEARN_SEARCH_SPACE_SIZE,
    description=(
        "library's STABLE-field default search: F5-governed quartet "
        "(task_type, loss.strategy, sampler.strategy, calibration_"
        "strategy) plus optimizer.learning_rate, optimizer.weight_"
        "decay, and the TFT model-shape fields (hidden_size, "
        "attention_heads, dropout, variable_selection_dropout, "
        "prediction_readout). task_type and tabular_config are "
        "pinned by the dataset spec at trial time; scheduler is "
        "bound by the adapter to the harness default."
    ),
)
register_hpo_space(_SEQ_SKLEARN_SPACE, sample_seq_sklearn_hyperparameters)
