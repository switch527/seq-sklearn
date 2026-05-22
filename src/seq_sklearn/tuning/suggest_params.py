"""Optuna search-space sampler (A16).

:func:`suggest_params` draws a config from the per-model default search
space. It is **closed under the F5 validity matrix by construction**:
``task_type`` is sampled first (constrained to the model family's v1
tasks), then ``loss.strategy`` from the legal losses for that task, then
``sampler.strategy`` and ``calibration_strategy`` from that cell's legal
subsets. The returned config therefore always satisfies
:func:`seq_sklearn.config._validity.check_combo`; the parent
:class:`BaseModelConfig` model-validator re-checks it on construction as
a backstop.

The numeric default search space (``search_advanced=False``,
``search_extras=False``) is ALPHA-stable: it may change without a MINOR
bump. Pass an explicit ``base`` and your own sampling for reproducible
behavior across releases.
"""

import logging
from collections.abc import Callable
from typing import Any

import optuna
from pydantic import ValidationError

from seq_sklearn.config._domains import (
    CLASSIFICATION_TASK_TYPES,
    REGRESSION_TASK_TYPES,
    V1_TASK_TYPES,
)
from seq_sklearn.config._validity import legal_strategies_for, legal_task_loss_pairs
from seq_sklearn.config.base import BaseModelConfig
from seq_sklearn.config.tft import TFTConfig
from seq_sklearn.errors import ConfigError
from seq_sklearn.models._classifier import BaseSequenceClassifier
from seq_sklearn.models._regressor import BaseSequenceRegressor
from seq_sklearn.tuning._alpha_keys import alpha_keys_for

__all__ = ["suggest_params"]

logger = logging.getLogger(__name__)

# v1 task types per model family. The classification / regression split
# and the v1 set are both single-sourced from config._domains, so no F5
# taxonomy is re-encoded here. The (task, loss) pairs come from the F5
# table for the legal-loss lookup below.
_V1_PAIRS = legal_task_loss_pairs()
_V1_SET = frozenset(V1_TASK_TYPES)
_V1_CLASSIFICATION_TASKS: tuple[str, ...] = tuple(
    t for t in CLASSIFICATION_TASK_TYPES if t in _V1_SET
)
_V1_REGRESSION_TASKS: tuple[str, ...] = tuple(t for t in REGRESSION_TASK_TYPES if t in _V1_SET)

# Default quantile vector injected when regression_quantile is sampled
# and `base` carries none. Monotone, in (0, 1) per the parent validator.
_DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)

# Fields the sampler always writes itself, so a required-without-default
# config field among these does NOT force the caller to pass `base`.
# Kept in sync with the merged-dict assignments below.
_SAMPLER_POPULATED_FIELDS = frozenset(
    {"task_type", "loss", "sampler", "calibration_strategy", "optimizer", "quantiles"}
)


def _suggest_tft_model_shape(trial: optuna.trial.BaseTrial) -> dict[str, object]:
    """Sample TFTConfig's STABLE model-shape fields.

    ``attention_heads`` is drawn from the divisors of the sampled
    ``hidden_size`` (the TFTConfig validator requires
    ``hidden_size % attention_heads == 0``), so the pair is legal by
    construction. The head parameter is namespaced by ``hidden_size``:
    Optuna forbids a categorical parameter name carrying a different
    choice set across trials, so a conditional space must give each
    branch its own name.
    """
    hidden_size = int(trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256]))
    head_choices = [h for h in (1, 2, 4, 8) if hidden_size % h == 0]
    return {
        "hidden_size": hidden_size,
        "attention_heads": int(
            trial.suggest_categorical(f"attention_heads@h{hidden_size}", head_choices)
        ),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "variable_selection_dropout": trial.suggest_float("variable_selection_dropout", 0.0, 0.5),
        "prediction_readout": trial.suggest_categorical(
            "prediction_readout", ["last_valid", "mean_pool"]
        ),
    }


# Per-config model-shape sampler, mirroring _ADAPTER_MAP_BY_CONFIG. v2 /
# v3 register their concrete config's shape sampler here.
_ModelShapeSampler = Callable[[optuna.trial.BaseTrial], dict[str, object]]
_MODEL_SHAPE_BY_CONFIG: dict[type[BaseModelConfig], _ModelShapeSampler] = {
    TFTConfig: _suggest_tft_model_shape,
}


def _family_tasks(model_class: type) -> tuple[str, ...]:
    """Return the v1 task types legal for ``model_class``'s family."""
    if issubclass(model_class, BaseSequenceClassifier):
        return _V1_CLASSIFICATION_TASKS
    if issubclass(model_class, BaseSequenceRegressor):
        return _V1_REGRESSION_TASKS
    raise ConfigError(
        f"{model_class.__name__} is neither a BaseSequenceClassifier nor a "
        "BaseSequenceRegressor; suggest_params cannot infer its task family."
    )


def suggest_params(
    trial: optuna.trial.BaseTrial,
    model_class: type[BaseSequenceClassifier | BaseSequenceRegressor],
    base: BaseModelConfig | None = None,
    *,
    task_type: str | None = None,
    search_advanced: bool = False,
    search_extras: bool = False,
) -> BaseModelConfig:
    """Sample a config from the per-model default search space.

    The default space (``search_advanced=False``, ``search_extras=False``)
    samples only STABLE fields: the F5-governed quartet
    (``task_type`` / ``loss.strategy`` / ``sampler.strategy`` /
    ``calibration_strategy``), the model family's STABLE model-shape
    fields, and the optimizer's ``learning_rate`` / ``weight_decay``.
    ``search_advanced=True`` additionally samples the model's
    ``<Model>AdvancedConfig`` fields and ``search_extras=True`` samples
    the curated per-family ALPHA-key list; both are empty in v1 by
    design, so the flags are structural no-ops for v1.

    Args:
        trial: an Optuna trial drawn from a live study
            (``study.ask()``); a ``FixedTrial`` would not exercise the
            search space.
        model_class: the concrete estimator class to tune. Its
            ``_config_cls`` selects the search space and the family
            (classifier / regressor) bounds ``task_type``.
        base: a fully-specified config supplying every non-searched
            structural field (for TFT this MUST carry ``tabular_config``,
            which has no default). Required whenever ``model_class``'s
            config has a required field outside the search space.
            ``quantiles`` is taken from ``base`` when the sampled task
            is ``regression_quantile`` (falling back to
            ``(0.1, 0.5, 0.9)`` only if ``base`` carries none); for
            every other task it is forced to ``None`` (the parent
            validator rejects quantiles on non-quantile tasks).
        task_type: pin the sampled task. When set, the sampler does
            NOT call ``trial.suggest_categorical("task_type", ...)``
            and uses the provided value directly. The
            F5-downstream sampling (loss / sampler /
            calibration_strategy) is still conditioned on this task
            so the returned config is closed under F5. The value
            must be one of the model family's v1 tasks; otherwise
            ``ConfigError`` is raised. Use case: a benchmark harness
            tuning against a dataset whose task is fixed.
        search_advanced: also sample BETA ``advanced`` fields (no-op in v1).
        search_extras: also sample curated ALPHA ``extra`` keys (no-op in v1).

    Returns:
        A validated ``model_class._config_cls`` instance, closed under
        the F5 validity matrix by construction.

    Raises:
        ConfigError: if ``model_class`` has no inferable task family;
            if ``base`` is required (a non-search required field exists)
            but was not supplied; if ``task_type`` is set but not legal
            for the model family; or if the assembled config fails
            pydantic validation (the ``ValidationError`` is wrapped per
            F8 so the Optuna guard can prune the trial).
    """
    config_cls: type[BaseModelConfig] = model_class._config_cls
    tasks = _family_tasks(model_class)

    required_non_search = {
        name
        for name, field in config_cls.model_fields.items()
        if field.is_required() and name not in _SAMPLER_POPULATED_FIELDS
    }
    if required_non_search and base is None:
        raise ConfigError(
            f"suggest_params requires `base=` for {model_class.__name__}: "
            f"its config has required non-search field(s) "
            f"{sorted(required_non_search)} (e.g. `tabular_config`) that "
            "the default search space does not sample."
        )

    merged: dict[str, Any] = base.model_dump() if base is not None else {}

    # Conditional categorical params are namespaced by the values they
    # depend on: Optuna rejects a parameter name whose choice set
    # changes across trials, so each branch of the F5 tree gets a
    # distinct name (the legal losses for a task, the legal
    # imbalance / calibration for a cell are each fixed per name).
    if task_type is not None:
        if task_type not in tasks:
            raise ConfigError(
                f"suggest_params: pinned task_type={task_type!r} is not "
                f"legal for {model_class.__name__}; family tasks are "
                f"{tasks}"
            )
    else:
        task_type = str(trial.suggest_categorical("task_type", list(tasks)))
    legal_losses = sorted({loss for (t, loss) in _V1_PAIRS if t == task_type})
    loss_strategy = str(trial.suggest_categorical(f"loss_strategy@{task_type}", legal_losses))
    legal_imbalance, legal_calibration = legal_strategies_for(task_type, loss_strategy)
    cell = f"{task_type}.{loss_strategy}"
    sampler_strategy = trial.suggest_categorical(f"sampler_strategy@{cell}", list(legal_imbalance))
    calibration_strategy = trial.suggest_categorical(
        f"calibration_strategy@{cell}", list(legal_calibration)
    )

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    loss_dict = dict(merged.get("loss") or {})
    loss_dict["strategy"] = loss_strategy
    sampler_dict = dict(merged.get("sampler") or {})
    sampler_dict["strategy"] = sampler_strategy
    optimizer_dict = dict(merged.get("optimizer") or {})
    optimizer_dict["learning_rate"] = learning_rate
    optimizer_dict["weight_decay"] = weight_decay

    merged["task_type"] = task_type
    merged["loss"] = loss_dict
    merged["sampler"] = sampler_dict
    merged["calibration_strategy"] = calibration_strategy
    merged["optimizer"] = optimizer_dict

    if task_type == "regression_quantile":
        existing = merged.get("quantiles")
        merged["quantiles"] = tuple(existing) if existing else _DEFAULT_QUANTILES
    else:
        merged["quantiles"] = None

    shape_sampler = _MODEL_SHAPE_BY_CONFIG.get(config_cls)
    if shape_sampler is not None:
        merged.update(shape_sampler(trial))

    if search_advanced:
        # v1: every <Model>AdvancedConfig is empty (only the BETA `extra`
        # escape hatch). Nothing to sample; kept as the documented
        # extension point for v2 / v3.
        logger.debug("search_advanced is a no-op in v1 (empty advanced configs).")
    if search_extras:
        # v1: alpha_keys_for is () for every config, so there is nothing
        # to sample. The lookup stays wired as the v2 / v3 extension
        # point; the sampling branch lands here when keys exist.
        logger.debug(
            "search_extras: curated ALPHA keys for %s are %s",
            config_cls.__name__,
            alpha_keys_for(config_cls),
        )

    # F8: never leak a raw pydantic.ValidationError. suggest_params is
    # the documented body of the Optuna `objective`, run inside
    # optuna_trial_guard, which only converts ConfigError / TrainingError
    # to a pruned trial. An unwrapped ValidationError would escape the
    # guard and (under study.optimize(catch=())) kill the whole study.
    # The sampler is closed under F5 by construction, so this is
    # defense-in-depth for an invalid `base` override or a future
    # config validator; it matches BaseSequenceEstimator._build_config.
    try:
        return config_cls(**merged)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc
