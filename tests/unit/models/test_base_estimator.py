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
import sklearn.exceptions

from seq_sklearn.config.adapters import OptimizerParams, TabularConfigParams
from seq_sklearn.errors import ConfigError, NotFittedError
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


def test_predict_proba_before_fit_raises_notfitted() -> None:
    # Classifier-shell error path: predict_proba pre-fit raises and is
    # catchable as BOTH the seq-sklearn and the sklearn NotFittedError
    # (dual-parent MRO), matching the regressor predict_quantiles path.
    est = _clf()
    x = pd.DataFrame({"id": [0], "time": [pd.Timestamp("2021-01-01")], "sr": [0.0]})
    with pytest.raises(NotFittedError):
        est.predict_proba(x)
    with pytest.raises(sklearn.exceptions.NotFittedError):
        est.predict_proba(x)


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
    from seq_sklearn.config.adapters import LossParams

    est = _clf(loss=LossParams(strategy="focal"))
    assert est._build_config().loss.strategy == "focal"


def test_build_config_wraps_validation_error_as_config_error() -> None:
    # platt is illegal for multiclass per the F5 validity matrix; the
    # pydantic ValidationError must surface as ConfigError (A4 step 4).
    est = _DummySequenceClassifier(task_type="multiclass", calibration_strategy="platt")
    with pytest.raises(ConfigError):
        est._build_config()


def test_window_time_index_shared_helper_and_monotone_guard() -> None:
    # The estimator's calibration-fold seam uses the single shared
    # splits.window_time_index (same impl the Trainer delegates to), and
    # the monotone-entity_id precondition raises (not silently corrupts).
    from seq_sklearn.data.splits import window_time_index
    from seq_sklearn.training.trainer import Trainer

    assert window_time_index(np.empty(0, dtype=int)).shape == (0,)
    np.testing.assert_array_equal(
        window_time_index(np.array([0, 0, 1, 1, 1])), np.array([0, 1, 0, 1, 2])
    )
    # Trainer's static alias delegates to the same shared impl (not just
    # truthy): identical output on the same input.
    np.testing.assert_array_equal(
        Trainer._window_time_index(np.array([0, 0, 1])),
        window_time_index(np.array([0, 0, 1])),
    )
    with pytest.raises(ValueError, match="monotone non-decreasing"):
        window_time_index(np.array([1, 0, 1]))


def test_calibration_set_with_positive_cal_fraction_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    est = _clf(cal_fraction=0.1)
    x = pd.DataFrame({"id": [0], "time": [pd.Timestamp("2021-01-01")], "sr": [0.0]})
    with pytest.raises(ConfigError, match="cal_fraction"):
        est.fit(x, np.array([0]), calibration_set=(x, np.array([0])))


def test_needs_calibration_fold_but_cal_fraction_zero_raises() -> None:
    # threshold_tuning (or a calibrator) with cal_fraction=0 and no
    # calibration_set has no fold to fit on: rejected at the fit
    # boundary naming cal_fraction, not deep in the calibrator (F2).
    est = _clf(threshold_tuning=True, cal_fraction=0.0)
    x = pd.DataFrame({"id": [0], "time": [pd.Timestamp("2021-01-01")], "sr": [0.0]})
    with pytest.raises(ConfigError, match="cal_fraction"):
        est.fit(x, np.array([0]))


def test_calibrator_strategy_with_cal_fraction_zero_raises() -> None:
    # The OTHER needs_fold operand: a configured calibrator (not
    # threshold_tuning) with cal_fraction=0 and no calibration_set must
    # also raise at the fit boundary naming cal_fraction (F2).
    est = _clf(calibration_strategy="platt", threshold_tuning=False, cal_fraction=0.0)
    x = pd.DataFrame({"id": [0], "time": [pd.Timestamp("2021-01-01")], "sr": [0.0]})
    with pytest.raises(ConfigError, match="cal_fraction"):
        est.fit(x, np.array([0]))
