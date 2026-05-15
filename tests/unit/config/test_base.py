"""Tests for the shared training and model configs at architecture A4.

Post-refactor: construction is nested (``loss=LossConfig(...)``,
``sampler=SamplerConfig(...)``, ``optimizer=OptimizerConfig(...)``)
rather than flat. The ``_model`` helper takes the F5 display-label
arguments (``loss_strategy`` / ``imbalance_strategy``) and translates to
the nested shape so every prior invariant is still pinned with identical
intent (the display labels are the F5 bridge vocabulary, allowed in
tests per the requirements F5 bridge table).
"""

import json

import pytest
from pydantic import ValidationError

from seq_sklearn.config.base import BaseModelConfig, BaseTrainingConfig
from seq_sklearn.config.loss import LossConfig
from seq_sklearn.config.optimizer import OptimizerConfig
from seq_sklearn.config.sampler import SamplerConfig
from seq_sklearn.config.scheduler import SchedulerConfig


def _model(
    *,
    task_type: str = "binary",
    loss_strategy: str = "cross_entropy",
    imbalance_strategy: str = "none",
    **overrides: object,
) -> BaseModelConfig:
    return BaseModelConfig(
        task_type=task_type,  # type: ignore[arg-type]
        loss=LossConfig(strategy=loss_strategy),  # type: ignore[arg-type]
        sampler=SamplerConfig(strategy=imbalance_strategy),  # type: ignore[arg-type]
        **overrides,
    )


class TestBaseTrainingConfig:
    def test_default_construction_succeeds(self) -> None:
        cfg = BaseTrainingConfig()
        assert cfg.optimizer.learning_rate == 1e-3
        assert cfg.optimizer.name == "adamw"
        assert cfg.scheduler.name == "cosine_with_warmup"
        assert cfg.batch_size == 64
        assert cfg.precision == "auto"

    def test_frozen_post_construction(self) -> None:
        cfg = BaseTrainingConfig()
        with pytest.raises(ValidationError):
            cfg.batch_size = 1  # type: ignore[misc]

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BaseTrainingConfig(not_a_field=True)  # type: ignore[call-arg]

    def test_negative_learning_rate_rejected(self) -> None:
        """Negative LR is rejected through the nested optimizer sub-config."""
        with pytest.raises(ValidationError):
            BaseTrainingConfig(optimizer=OptimizerConfig(learning_rate=-0.1))


class TestBaseModelConfig:
    def test_minimal_legal_construction(self) -> None:
        cfg = _model()
        assert cfg.task_type == "binary"
        assert cfg.loss.strategy == "cross_entropy"
        assert cfg.sampler.strategy == "none"
        assert cfg.calibration_strategy == "none"

    def test_loss_is_required(self) -> None:
        """LossConfig has no default strategy; loss must be supplied."""
        with pytest.raises(ValidationError):
            BaseModelConfig(task_type="binary")  # type: ignore[call-arg]

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _model(extra_param=1)

    def test_illegal_combo_rejected(self) -> None:
        with pytest.raises(ValidationError, match="class_weighted"):
            _model(loss_strategy="focal", imbalance_strategy="class_weighted")

    def test_v1_1_task_type_rejected_with_message(self) -> None:
        with pytest.raises(ValidationError, match=r"v1\.1"):
            _model(task_type="multilabel")

    @pytest.mark.parametrize("task", ["multilabel", "regression_multioutput"])
    def test_v1_task_type_rejects_multilabel_and_regression_multioutput(self, task: str) -> None:
        """Both v1.1 task types are rejected at the user-facing config surface."""
        loss = "cross_entropy" if task == "multilabel" else "mse"
        with pytest.raises(ValidationError, match=r"v1\.1"):
            _model(task_type=task, loss_strategy=loss)

    def test_val_plus_cal_must_be_under_one(self) -> None:
        with pytest.raises(ValidationError, match="val_fraction"):
            _model(val_fraction=0.6, cal_fraction=0.5)

    def test_quantiles_strictly_increasing_required(self) -> None:
        with pytest.raises(ValidationError, match="strictly increasing"):
            _model(
                task_type="regression_quantile",
                loss_strategy="pinball",
                quantiles=(0.5, 0.1, 0.9),
            )

    def test_quantiles_inside_open_unit_interval_required(self) -> None:
        with pytest.raises(ValidationError, match=r"\(0, 1\)"):
            _model(
                task_type="regression_quantile",
                loss_strategy="pinball",
                quantiles=(0.0, 0.5, 0.9),
            )

    def test_quantiles_at_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _model(
                task_type="regression_quantile",
                loss_strategy="pinball",
                quantiles=(0.1, 0.5, 1.0),
            )

    def test_quantiles_single_element_accepted(self) -> None:
        """One-element tuple has no adjacent pairs; the monotone check
        is vacuously true. Pinning the empty-range edge."""
        cfg = _model(
            task_type="regression_quantile",
            loss_strategy="pinball",
            quantiles=(0.5,),
            calibration_strategy="conformal",
        )
        assert cfg.quantiles == (0.5,)

    def test_val_plus_cal_sum_exactly_one_rejected(self) -> None:
        """The `>= 1.0` boundary at base.py:_check_val_cal_sum; pins the gate."""
        with pytest.raises(ValidationError, match="val_fraction"):
            _model(val_fraction=0.5, cal_fraction=0.5)

    def test_legal_quantile_vector_succeeds(self) -> None:
        cfg = _model(
            task_type="regression_quantile",
            loss_strategy="pinball",
            quantiles=(0.1, 0.5, 0.9),
            calibration_strategy="conformal",
        )
        assert cfg.quantiles == (0.1, 0.5, 0.9)

    def test_frozen_post_construction(self) -> None:
        cfg = _model()
        with pytest.raises(ValidationError):
            cfg.task_type = "multiclass"  # type: ignore[misc]

    def test_nested_base_model_config_model_dump_json_round_trips(self) -> None:
        """The nested four-tier shape survives the on-disk JSON round trip.

        Pins requirements N1 save/load (mode="json"): every nested family
        sub-config carries a non-empty ``extra`` tuple that must survive
        ``model_dump`` -> json -> ``model_validate`` byte-for-byte.
        """
        original = BaseModelConfig(
            task_type="binary",
            loss=LossConfig(strategy="cross_entropy", extra=(("loss_flag", True),)),
            sampler=SamplerConfig(strategy="none", extra=(("samp_flag", 1),)),
            optimizer=OptimizerConfig(name="sgd", learning_rate=0.05, extra=(("opt_flag", "x"),)),
            scheduler=SchedulerConfig(name="one_cycle", extra=(("sched_flag", 2.5),)),
            calibration_strategy="platt",
        )
        payload = json.loads(json.dumps(original.model_dump(mode="json")))
        restored = BaseModelConfig.model_validate(payload)
        assert restored == original
