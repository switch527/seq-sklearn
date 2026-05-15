"""Shared training and model configs (per architecture A4).

:class:`BaseTrainingConfig` carries optimizer / scheduler / hardware
fields that every model needs at fit time. :class:`BaseModelConfig`
extends with task-type, loss, imbalance, and calibration plus the F5
validity-matrix and quantile-monotonicity cross-field validators.

Both are frozen pydantic v2 models. Mutation is reconciled with sklearn's
contract via the :class:`seq_sklearn.config._params_adapter.TabularConfigParams`
adapter pattern documented in A4 step 3.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from seq_sklearn.config._validity import check_combo

__all__ = ["BaseModelConfig", "BaseTrainingConfig"]


class BaseTrainingConfig(BaseModel):
    """Shared training-loop hyperparameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    batch_size: int = Field(default=64, ge=1)
    max_epochs: int = Field(default=50, ge=1)
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    scheduler: Literal["constant", "cosine_with_warmup", "one_cycle", "reduce_on_plateau"] = (
        "cosine_with_warmup"
    )
    warmup_steps: int = Field(default=100, ge=0)
    gradient_clip_val: float | None = None
    accumulate_grad_batches: int = Field(default=1, ge=1)
    precision: Literal["bf16-mixed", "16-mixed", "32-true", "auto"] = "auto"
    early_stopping_patience: int = Field(default=10, ge=1)
    val_check_interval: float = Field(default=1.0, gt=0.0)
    val_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    cal_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    val_split_strategy: Literal["time_ordered", "random"] = "time_ordered"
    num_workers: int | None = None
    pin_memory: bool | None = None
    seed: int = 42
    verbose: bool = True


class BaseModelConfig(BaseTrainingConfig):
    """Shared model-level hyperparameters.

    The two cross-field validators below run in the order declared:

    1. :meth:`_check_validity_matrix` rejects any
       ``(task_type, loss_strategy, imbalance_strategy, calibration_strategy)``
       cell that is not in the F5 matrix (delegates to
       :func:`seq_sklearn.config._validity.check_combo`).
    2. :meth:`_check_quantiles_monotone` rejects any non-monotone
       ``quantiles`` vector or value outside ``(0, 1)``. Lives on
       :class:`BaseModelConfig` (not on :class:`seq_sklearn.config.tft.TFTConfig`)
       so v2 quantile regressors (PatchTST quantile, TimesNet quantile)
       inherit the validator without duplication.
    """

    task_type: Literal[
        "binary",
        "multiclass",
        "multilabel",
        "regression_point",
        "regression_quantile",
        "regression_multioutput",
    ]
    loss_strategy: Literal["cross_entropy", "focal", "mse", "mae", "huber", "pinball"]
    imbalance_strategy: Literal[
        "none", "class_weighted", "oversample_minority", "undersample_majority"
    ] = "none"
    calibration_strategy: Literal[
        "none",
        "temperature",
        "platt",
        "isotonic",
        "conformal",
        "isotonic_quantile",
    ] = "none"
    threshold_tuning: bool = False
    threshold_metric: Literal["f1", "balanced_accuracy", "youden_j"] = "f1"
    focal_gamma: float = Field(default=2.0, gt=0.0)
    huber_delta: float = Field(default=1.0, gt=0.0)
    quantiles: tuple[float, ...] | None = None
    oversample_ratio: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _check_validity_matrix(self) -> Self:
        check_combo(
            self.task_type,
            self.loss_strategy,
            self.imbalance_strategy,
            self.calibration_strategy,
        )
        return self

    @model_validator(mode="after")
    def _check_quantiles_monotone(self) -> Self:
        if self.quantiles is None:
            return self
        q = self.quantiles
        if any(not (0.0 < v < 1.0) for v in q):
            raise ValueError(f"quantiles must lie in (0, 1); got {list(q)}")
        if any(q[i] >= q[i + 1] for i in range(len(q) - 1)):
            raise ValueError(f"quantiles must be strictly increasing; got {list(q)}")
        return self

    @model_validator(mode="after")
    def _check_val_cal_sum(self) -> Self:
        if self.val_fraction + self.cal_fraction >= 1.0:
            raise ValueError(
                f"val_fraction + cal_fraction must be < 1.0; got "
                f"{self.val_fraction} + {self.cal_fraction} = "
                f"{self.val_fraction + self.cal_fraction}"
            )
        return self
