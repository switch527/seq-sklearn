"""Estimator-shell contract tests (Phase 6a; F1 / F1.1 / A4).

Exercises the sklearn-contract surface that needs no training: the
adapter-bridged ``get_params`` / ``set_params`` / ``clone``, the punted
methods, ``__sklearn_tags__`` field values, the task-aware loss-default
injection, the ``ConfigError`` wrap of a bad validity-matrix combo, and
the F2 ``calibration_set`` / ``cal_fraction`` conflict.
"""

import numpy as np
import pandas as pd
import pytest
import sklearn

from seq_sklearn.config._adapters import OptimizerParams, TabularConfigParams
from seq_sklearn.errors import ConfigError
from tests._test_models._dummy_estimator import (
    _DummySequenceClassifier,
    _DummySequenceRegressor,
)


def _clf(**kw: object) -> _DummySequenceClassifier:
    params: dict[str, object] = {"task_type": "binary"}
    params.update(kw)
    return _DummySequenceClassifier(**params)  # type: ignore[arg-type]


def test_get_params_exposes_flat_double_underscore_keys() -> None:
    est = _clf(tabular_config=TabularConfigParams(lookback=6))
    params = est.get_params(deep=True)
    assert params["tabular_config__lookback"] == 6
    assert "optimizer__learning_rate" in params
    assert "loss" in params
    assert params["loss"] is None


def test_set_params_chains_through_adapter() -> None:
    est = _clf()
    est.set_params(optimizer__learning_rate=3e-4, tabular_config__lookback=9)
    assert est.optimizer.learning_rate == 3e-4
    assert est.tabular_config.lookback == 9


def test_clone_is_independent_no_adapter_aliasing() -> None:
    opt = OptimizerParams(learning_rate=5e-4)
    est = _clf(optimizer=opt)
    clone = sklearn.base.clone(est)
    assert clone.optimizer is not est.optimizer
    assert clone.optimizer is not opt
    clone.optimizer.learning_rate = 1.0
    assert est.optimizer.learning_rate == 5e-4


@pytest.mark.parametrize("method", ["partial_fit", "fit_predict", "fit_transform"])
def test_punted_methods_raise_stable_message(method: str) -> None:
    est = _clf()
    with pytest.raises(NotImplementedError, match=f"{method} is not supported in seq-sklearn v1"):
        getattr(est, method)()


def test_sklearn_tags_base_values() -> None:
    # sklearn 1.6+ InputTags has no `dataframe` field; the panel-frame
    # contract is two_d_array=False + allow_nan=False (F1.1 reconciled).
    tags = _clf().__sklearn_tags__()
    assert tags.input_tags.two_d_array is False
    assert tags.input_tags.allow_nan is False
    assert tags.target_tags.required is True
    assert tags.requires_fit is True
    assert tags.non_deterministic is False


def test_sklearn_tags_non_deterministic_flips_on_mixed_precision() -> None:
    assert _clf(precision="bf16-mixed").__sklearn_tags__().non_deterministic is True
    assert _clf(precision="16-mixed").__sklearn_tags__().non_deterministic is True
    assert _clf(precision="32-true").__sklearn_tags__().non_deterministic is False


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("binary", "cross_entropy"),
        ("multiclass", "cross_entropy"),
        ("regression_point", "mse"),
        ("regression_quantile", "pinball"),
    ],
)
def test_loss_default_injection_per_task_type(task: str, expected: str) -> None:
    cls = _DummySequenceRegressor if task.startswith("regression") else _DummySequenceClassifier
    kw: dict[str, object] = {"task_type": task}
    if task == "regression_quantile":
        kw["quantiles"] = (0.1, 0.5, 0.9)
    est = cls(**kw)  # type: ignore[arg-type]
    config = est._build_config()
    assert config.loss.strategy == expected


def test_explicit_loss_adapter_overrides_default() -> None:
    from seq_sklearn.config._adapters import LossParams

    est = _clf(loss=LossParams(strategy="focal"))
    assert est._build_config().loss.strategy == "focal"


def test_build_config_wraps_validation_error_as_config_error() -> None:
    # platt is illegal for multiclass per the F5 validity matrix; the
    # pydantic ValidationError must surface as ConfigError (A4 step 4).
    est = _DummySequenceClassifier(task_type="multiclass", calibration_strategy="platt")
    with pytest.raises(ConfigError):
        est._build_config()


def test_window_time_index_empty_returns_empty() -> None:
    from seq_sklearn.models._base import _window_time_index

    out = _window_time_index(np.empty(0, dtype=int))
    assert out.shape == (0,)


def test_calibration_set_with_positive_cal_fraction_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    est = _clf(cal_fraction=0.1)
    x = pd.DataFrame({"id": [0], "time": [pd.Timestamp("2021-01-01")], "sr": [0.0]})
    with pytest.raises(ConfigError, match="cal_fraction"):
        est.fit(x, np.array([0]), calibration_set=(x, np.array([0])))
