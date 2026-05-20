"""Shared training and model configs (per architecture A4).

:class:`BaseTrainingConfig` carries training-loop fields plus the nested
:class:`OptimizerConfig` / :class:`SchedulerConfig` family sub-configs.
:class:`BaseModelConfig` extends with task-type, the nested
:class:`LossConfig` / :class:`SamplerConfig`, calibration, and the F5
validity-matrix / quantile-monotonicity / val+cal-sum cross-field
validators.

Both are frozen pydantic v2 models. Mutation is reconciled with
sklearn's contract via the adapter pattern at
:mod:`seq_sklearn.config.adapters` (A4 step 3).
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from seq_sklearn.config._validity import check_combo
from seq_sklearn.config.loss import LossConfig
from seq_sklearn.config.optimizer import OptimizerConfig
from seq_sklearn.config.sampler import SamplerConfig
from seq_sklearn.config.scheduler import SchedulerConfig

__all__ = ["BaseModelConfig", "BaseTrainingConfig"]


class BaseTrainingConfig(BaseModel):
    """Shared training-loop hyperparameters.

    Optimizer and scheduler hyperparameters live on the nested
    :class:`OptimizerConfig` / :class:`SchedulerConfig` sub-configs
    (each with its own ``extra`` escape hatch) rather than as flat
    fields, so the configuration surface can grow without a shape break.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(default=64, ge=1)
    max_epochs: int = Field(default=50, ge=1)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
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

    The three cross-field validators below run in the order declared
    (pydantic v2 dispatches ``model_validator(mode="after")`` in
    declaration order; nested sub-configs are validated before the
    parent's after-validators run):

    1. :meth:`_check_validity_matrix` rejects any
       ``(task_type, loss.strategy, sampler.strategy, calibration_strategy)``
       cell not in the F5 matrix (delegates to
       :func:`seq_sklearn.config._validity.check_combo`, whose parameter
       names keep the F5 display labels ``loss_strategy`` /
       ``imbalance_strategy`` per the requirements F5 bridge table).
    2. :meth:`_check_quantiles_monotone` rejects a non-monotone
       ``quantiles`` vector or any value outside ``(0, 1)``. Lives here
       (not on :class:`seq_sklearn.config.tft.TFTConfig`) so v2 quantile
       regressors inherit it without duplication.
    3. :meth:`_check_val_cal_sum` rejects ``val_fraction + cal_fraction
       >= 1.0``.
    """

    task_type: Literal[
        "binary",
        "multiclass",
        "multilabel",
        "regression_point",
        "regression_quantile",
        "regression_multioutput",
    ]
    loss: LossConfig
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
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
    quantiles: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def _check_validity_matrix(self) -> Self:
        check_combo(
            self.task_type,
            self.loss.strategy,
            self.sampler.strategy,
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
