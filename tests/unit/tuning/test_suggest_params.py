"""suggest_params A16 contract (Phase 8, no training).

The headline test is the 1000-iteration validity sweep against a real
``study.ask()`` (a ``FixedTrial`` would not actually exercise the
search space). The rest pin the family bounding, the
closed-under-F5-by-construction guarantee, the regression_quantile
quantile injection, the attention_heads divisibility invariant, the
required-`base` guard, and the v1 no-op flags.
"""

import optuna
import pytest

from seq_sklearn.config._validity import check_combo
from seq_sklearn.config.base import BaseModelConfig
from seq_sklearn.config.loss import LossConfig
from seq_sklearn.config.tabular import TabularToSequenceConfig
from seq_sklearn.config.tft import TFTConfig
from seq_sklearn.errors import ConfigError
from seq_sklearn.models._base import BaseSequenceEstimator
from seq_sklearn.models._classifier import BaseSequenceClassifier
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier
from seq_sklearn.models.transformer.tft.regressor import TFTRegressor
from seq_sklearn.tuning.suggest_params import _MODEL_SHAPE_BY_CONFIG, suggest_params


def _base() -> TFTConfig:
    return TFTConfig(
        task_type="binary",
        loss=LossConfig(strategy="cross_entropy"),
        tabular_config=TabularToSequenceConfig(
            id_col="id",
            time_col="t",
            time_varying_real_cols=("x0", "x1"),
        ),
    )


def test_suggest_params_sweep_respects_validity_matrix() -> None:
    study = optuna.create_study()
    base = _base()
    for _ in range(1000):
        trial = study.ask()
        config = suggest_params(trial, TFTClassifier, base=base)
        # Closed under F5 by construction: this must never raise.
        check_combo(
            config.task_type,
            config.loss.strategy,
            config.sampler.strategy,
            config.calibration_strategy,
        )
        study.tell(trial, 0.0)


def test_regressor_sweep_respects_validity_matrix() -> None:
    # The classifier sweep above only proves F5 closure for the
    # classification cells. The regressor family walks different cells
    # (regression_point: mse/mae/huber; regression_quantile: pinball),
    # so it needs its own closed-under-F5 sweep.
    study = optuna.create_study()
    base = _base()
    for _ in range(1000):
        trial = study.ask()
        config = suggest_params(trial, TFTRegressor, base=base)
        check_combo(
            config.task_type,
            config.loss.strategy,
            config.sampler.strategy,
            config.calibration_strategy,
        )
        study.tell(trial, 0.0)


def test_classifier_only_samples_classification_tasks() -> None:
    study = optuna.create_study()
    base = _base()
    seen: set[str] = set()
    for _ in range(200):
        trial = study.ask()
        seen.add(suggest_params(trial, TFTClassifier, base=base).task_type)
        study.tell(trial, 0.0)
    assert seen <= {"binary", "multiclass"}


def test_regressor_only_samples_regression_tasks_with_valid_quantiles() -> None:
    study = optuna.create_study()
    base = _base()
    seen: set[str] = set()
    for _ in range(200):
        trial = study.ask()
        cfg = suggest_params(trial, TFTRegressor, base=base)
        seen.add(cfg.task_type)
        if cfg.task_type == "regression_quantile":
            q = cfg.quantiles
            assert q is not None
            assert all(0.0 < v < 1.0 for v in q)
            assert list(q) == sorted(q)
        else:
            assert cfg.quantiles is None
        study.tell(trial, 0.0)
    assert seen <= {"regression_point", "regression_quantile"}


def test_regression_quantile_preserves_base_quantiles() -> None:
    # The truthy arm of the quantile injection: a base that carries its
    # own quantiles must NOT be overwritten with _DEFAULT_QUANTILES when
    # regression_quantile is sampled. (coverage.py does not branch-split
    # the ternary, so this needs an explicit value assertion.)
    base = TFTConfig(
        task_type="regression_quantile",
        loss=LossConfig(strategy="pinball"),
        quantiles=(0.25, 0.75),
        tabular_config=TabularToSequenceConfig(
            id_col="id",
            time_col="t",
            time_varying_real_cols=("x0",),
        ),
    )
    study = optuna.create_study()
    saw_quantile = False
    for _ in range(200):
        trial = study.ask()
        cfg = suggest_params(trial, TFTRegressor, base=base)
        if cfg.task_type == "regression_quantile":
            saw_quantile = True
            assert cfg.quantiles == (0.25, 0.75)  # base preserved
        else:
            assert cfg.quantiles is None
        study.tell(trial, 0.0)
    assert saw_quantile


def test_attention_heads_always_divide_hidden_size() -> None:
    study = optuna.create_study()
    base = _base()
    for _ in range(200):
        trial = study.ask()
        cfg = suggest_params(trial, TFTClassifier, base=base)
        assert isinstance(cfg, TFTConfig)
        assert cfg.hidden_size % cfg.attention_heads == 0
        study.tell(trial, 0.0)


def test_returns_concrete_config_cls() -> None:
    study = optuna.create_study()
    trial = study.ask()
    cfg = suggest_params(trial, TFTClassifier, base=_base())
    assert isinstance(cfg, TFTConfig)
    assert isinstance(cfg, BaseModelConfig)


def test_base_required_when_config_has_required_non_search_field() -> None:
    study = optuna.create_study()
    trial = study.ask()
    with pytest.raises(ConfigError, match="requires `base=`"):
        suggest_params(trial, TFTClassifier, base=None)


def test_unknown_family_raises_configerror() -> None:
    class _Bare(BaseSequenceEstimator):  # neither classifier nor regressor
        pass

    study = optuna.create_study()
    trial = study.ask()
    with pytest.raises(ConfigError, match="neither a BaseSequenceClassifier"):
        suggest_params(trial, _Bare, base=_base())  # type: ignore[type-var]


def test_generic_config_cls_skips_model_shape_sampler() -> None:
    # A classifier whose _config_cls has no registered model-shape
    # sampler (only TFTConfig is registered in v1): suggest_params must
    # still produce a valid F5-closed config, just without TFT-shape
    # fields. Exercises the shape_sampler-is-None path.
    class _GenericClf(BaseSequenceClassifier):
        _config_cls = BaseModelConfig

    study = optuna.create_study()
    trial = study.ask()
    cfg = suggest_params(trial, _GenericClf, base=None)
    assert type(cfg) is BaseModelConfig
    assert not isinstance(cfg, TFTConfig)
    assert not hasattr(cfg, "hidden_size")
    check_combo(
        cfg.task_type,
        cfg.loss.strategy,
        cfg.sampler.strategy,
        cfg.calibration_strategy,
    )


def test_search_advanced_and_extras_are_v1_noops() -> None:
    study = optuna.create_study()
    base = _base()
    trial = study.ask()
    cfg = suggest_params(trial, TFTClassifier, base=base, search_advanced=True, search_extras=True)
    assert isinstance(cfg, TFTConfig)
    # advanced stays at its empty v1 default; extra escape hatch untouched.
    assert cfg.advanced == base.advanced


def test_search_advanced_noop_alone() -> None:
    # Isolate the search_advanced branch from search_extras so a
    # regression in one is not masked by the other.
    study = optuna.create_study()
    base = _base()
    cfg = suggest_params(study.ask(), TFTClassifier, base=base, search_advanced=True)
    assert isinstance(cfg, TFTConfig)
    assert cfg.advanced == base.advanced


def test_search_extras_noop_alone() -> None:
    study = optuna.create_study()
    base = _base()
    cfg = suggest_params(study.ask(), TFTClassifier, base=base, search_extras=True)
    assert isinstance(cfg, TFTConfig)
    assert cfg.advanced == base.advanced


def test_invalid_assembled_config_wrapped_as_configerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # F8 + Gemini final-pass r1-C1: an invalid assembled config must
    # raise ConfigError, not a raw pydantic.ValidationError, so the
    # Optuna guard prunes the trial instead of the bare error killing
    # the study. Force an invalid shape (10 % 3 != 0 fails the
    # TFTConfig heads-divide-hidden validator).
    def _bad_shape(_trial: object) -> dict[str, object]:
        return {
            "hidden_size": 10,
            "attention_heads": 3,
            "dropout": 0.1,
            "variable_selection_dropout": 0.1,
            "prediction_readout": "last_valid",
        }

    monkeypatch.setitem(_MODEL_SHAPE_BY_CONFIG, TFTConfig, _bad_shape)
    study = optuna.create_study()
    with pytest.raises(ConfigError):
        suggest_params(study.ask(), TFTClassifier, base=_base())
