"""Tests for the shared training and model configs at architecture A4."""

import pytest
from pydantic import ValidationError

from seq_sklearn.config.base import BaseModelConfig, BaseTrainingConfig


class TestBaseTrainingConfig:
    def test_default_construction_succeeds(self) -> None:
        cfg = BaseTrainingConfig()
        assert cfg.learning_rate == 1e-3
        assert cfg.batch_size == 64
        assert cfg.precision == "auto"

    def test_frozen_post_construction(self) -> None:
        cfg = BaseTrainingConfig()
        with pytest.raises(ValidationError):
            cfg.learning_rate = 0.5  # type: ignore[misc]

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BaseTrainingConfig(not_a_field=True)  # type: ignore[call-arg]

    def test_negative_learning_rate_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BaseTrainingConfig(learning_rate=-0.1)


class TestBaseModelConfig:
    def _legal_kwargs(self, **overrides: object) -> dict[str, object]:
        return {
            "task_type": "binary",
            "loss_strategy": "cross_entropy",
            **overrides,
        }

    def test_minimal_legal_construction(self) -> None:
        cfg = BaseModelConfig(**self._legal_kwargs())
        assert cfg.task_type == "binary"
        assert cfg.imbalance_strategy == "none"
        assert cfg.calibration_strategy == "none"

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BaseModelConfig(**self._legal_kwargs(extra_param=1))

    def test_illegal_combo_rejected(self) -> None:
        with pytest.raises(ValidationError, match="class_weighted"):
            BaseModelConfig(
                **self._legal_kwargs(loss_strategy="focal", imbalance_strategy="class_weighted")
            )

    def test_v1_1_task_type_rejected_with_message(self) -> None:
        with pytest.raises(ValidationError, match=r"v1\.1"):
            BaseModelConfig(**self._legal_kwargs(task_type="multilabel"))

    def test_val_plus_cal_must_be_under_one(self) -> None:
        with pytest.raises(ValidationError, match="val_fraction"):
            BaseModelConfig(**self._legal_kwargs(val_fraction=0.6, cal_fraction=0.5))

    def test_quantiles_strictly_increasing_required(self) -> None:
        with pytest.raises(ValidationError, match="strictly increasing"):
            BaseModelConfig(
                **self._legal_kwargs(
                    task_type="regression_quantile",
                    loss_strategy="pinball",
                    quantiles=(0.5, 0.1, 0.9),
                )
            )

    def test_quantiles_inside_open_unit_interval_required(self) -> None:
        with pytest.raises(ValidationError, match=r"\(0, 1\)"):
            BaseModelConfig(
                **self._legal_kwargs(
                    task_type="regression_quantile",
                    loss_strategy="pinball",
                    quantiles=(0.0, 0.5, 0.9),
                )
            )

    def test_quantiles_at_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BaseModelConfig(
                **self._legal_kwargs(
                    task_type="regression_quantile",
                    loss_strategy="pinball",
                    quantiles=(0.1, 0.5, 1.0),
                )
            )

    def test_quantiles_single_element_accepted(self) -> None:
        """One-element tuple has no adjacent pairs; the monotone check
        is vacuously true. Pinning the empty-range edge."""
        cfg = BaseModelConfig(
            **self._legal_kwargs(
                task_type="regression_quantile",
                loss_strategy="pinball",
                quantiles=(0.5,),
                calibration_strategy="conformal",
            )
        )
        assert cfg.quantiles == (0.5,)

    def test_val_plus_cal_sum_exactly_one_rejected(self) -> None:
        """The `>= 1.0` boundary at base.py:_check_val_cal_sum; pins the gate."""
        with pytest.raises(ValidationError, match="val_fraction"):
            BaseModelConfig(**self._legal_kwargs(val_fraction=0.5, cal_fraction=0.5))

    def test_legal_quantile_vector_succeeds(self) -> None:
        cfg = BaseModelConfig(
            **self._legal_kwargs(
                task_type="regression_quantile",
                loss_strategy="pinball",
                quantiles=(0.1, 0.5, 0.9),
                calibration_strategy="conformal",
            )
        )
        assert cfg.quantiles == (0.1, 0.5, 0.9)

    def test_frozen_post_construction(self) -> None:
        cfg = BaseModelConfig(**self._legal_kwargs())
        with pytest.raises(ValidationError):
            cfg.task_type = "multiclass"  # type: ignore[misc]
