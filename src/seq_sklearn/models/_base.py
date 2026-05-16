"""``BaseSequenceEstimator``: the sklearn-contract estimator shell (A2 / A4 / A17).

The estimator owns mutable scalar attributes plus six nested
``BaseEstimator`` adapters (A4 step 1/3). The frozen pydantic config is
built inside :meth:`fit` as ``self.config_`` (A4 step 2); the
task-type-aware loss default is injected there when the ``loss`` adapter
is omitted. ``fit`` plumbs :class:`TabularToSequence` -> :class:`Trainer`
(curried-factory pattern, A7), then fits the optional calibrator on the
held-out calibration fold (A9). Save / load is the A17 two-file format.

The class is abstract: a concrete family supplies ``_config_cls`` and
:meth:`_build_backbone_head`; the classifier / regressor mix-ins overlay
``predict*`` and the F1.1 fit-state attributes. The synthetic
``_DummySequenceClassifier`` / ``_DummySequenceRegressor`` exercise the
whole shell before TFT specifics land in Phase 7.
"""

import platform
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import pandas as pd
import torch
from pydantic import ValidationError
from sklearn.base import BaseEstimator
from sklearn.utils import Tags
from torch import Tensor, nn

import seq_sklearn
from seq_sklearn.calibration._protocol import _Calibrator
from seq_sklearn.config._adapters import (
    LossParams,
    OptimizerParams,
    SamplerParams,
    SchedulerParams,
    TabularConfigParams,
    TFTAdvancedParams,
)
from seq_sklearn.config.base import BaseModelConfig
from seq_sklearn.config.tabular import TabularToSequenceConfig
from seq_sklearn.data.splits import compute_three_way_split
from seq_sklearn.data.tabular_to_sequence import TabularToSequence
from seq_sklearn.errors import ConfigError, NotFittedError
from seq_sklearn.serialization import (
    CURRENT_SCHEMA_VERSION,
    load_weights_and_state,
    save_weights_and_state,
)
from seq_sklearn.training._lightning_module import _LightningModule
from seq_sklearn.training.trainer import Trainer

__all__ = ["BaseSequenceEstimator"]

# A4: LossConfig.strategy has no pydantic default (task-dependent); the
# estimator injects the task-appropriate value when `loss` is omitted.
# The v1.1 entries are unreachable in v1 (the F5 validity matrix rejects
# v1.1 task_types before _build_config runs) but present so v1.1 is a
# one-line enablement later.
_DEFAULT_LOSS_FOR_TASK: dict[str, str] = {
    "binary": "cross_entropy",
    "multiclass": "cross_entropy",
    "multilabel": "cross_entropy",
    "regression_point": "mse",
    "regression_quantile": "pinball",
    "regression_multioutput": "mse",
}

_MIXED_PRECISION = ("bf16-mixed", "16-mixed")
_PUNTED_MSG = "{name} is not supported in seq-sklearn v1; see docs/roadmap.md"


def _window_time_index(entity_ids: np.ndarray) -> np.ndarray:
    """Per-entity time ordinals from the contiguous batch blocks (A5).

    ``TabularToSequence.transform`` emits each entity's windows
    contiguously, time-ascending, with ``entity_id`` monotone
    non-decreasing, so the ordinal is
    ``concatenate([arange(n_i) for n_i])``. Mirrors
    ``Trainer._window_time_index`` so the estimator locates the same
    calibration fold the Trainer held out for the identical config.
    """
    if entity_ids.size == 0:
        return np.empty(0, dtype=int)
    _, counts = np.unique(entity_ids, return_counts=True)
    return np.concatenate([np.arange(c) for c in counts])


class BaseSequenceEstimator(BaseEstimator, ABC):
    """Abstract sklearn estimator shell shared by every model family.

    ``__sklearn_tags__`` reads raw instance attributes (never
    ``self.config_``) because ``check_estimator`` invokes it on
    unconfigured instances (A4 step 2).
    """

    # Concrete families set this to their config subclass (TFTConfig in
    # Phase 7); the base / dummy build a plain BaseModelConfig.
    _config_cls: ClassVar[type[BaseModelConfig]] = BaseModelConfig

    # Fit-state attributes (F1.1), declared so pyright tracks them on the
    # __new__-constructed instance in load(). The trailing-underscore
    # names are the sklearn fit-state convention; _module / _loaded_*
    # are private wiring.
    config_: BaseModelConfig
    transformer_: TabularToSequence
    calibrator_: _Calibrator | None
    n_features_in_: int
    feature_names_in_: np.ndarray
    n_outputs_: int
    feature_schema_fingerprint_: str | None
    _module: _LightningModule
    _loaded_backbone: nn.Module
    _loaded_head: nn.Module

    def __init__(
        self,
        *,
        task_type: str,
        tabular_config: TabularConfigParams | None = None,
        optimizer: OptimizerParams | None = None,
        scheduler: SchedulerParams | None = None,
        loss: LossParams | None = None,
        sampler: SamplerParams | None = None,
        advanced: TFTAdvancedParams | None = None,
        calibration_strategy: str = "none",
        threshold_tuning: bool = False,
        threshold_metric: str = "f1",
        quantiles: tuple[float, ...] | None = None,
        batch_size: int = 64,
        max_epochs: int = 50,
        gradient_clip_val: float | None = None,
        accumulate_grad_batches: int = 1,
        precision: str = "auto",
        early_stopping_patience: int = 10,
        val_check_interval: float = 1.0,
        val_fraction: float = 0.1,
        cal_fraction: float = 0.1,
        val_split_strategy: str = "time_ordered",
        num_workers: int | None = None,
        pin_memory: bool | None = None,
        seed: int = 42,
        verbose: bool = True,
        resume_path: str | Path | None = None,
    ) -> None:
        self.task_type = task_type
        # sklearn's contract: __init__ stores params verbatim and never
        # transforms them (sklearn.base.clone re-checks identity and
        # rejects any constructor that "modifies a parameter"). The
        # A4-draft "clone each adapter in __init__" both breaks clone()
        # and is redundant: sklearn.base.clone already deep-clones every
        # nested param before re-construction, so a cloned estimator
        # never aliases the original's adapters. A None nested param is
        # filled with a fresh default object (sklearn allows the
        # None-default-instance idiom; the substituted default is then
        # stable across clone because clone passes the real instance,
        # not None, back in).
        self.tabular_config = (
            tabular_config if tabular_config is not None else TabularConfigParams()
        )
        self.optimizer = optimizer if optimizer is not None else OptimizerParams()
        self.scheduler = scheduler if scheduler is not None else SchedulerParams()
        self.loss = loss
        self.sampler = sampler if sampler is not None else SamplerParams()
        self.advanced = advanced if advanced is not None else TFTAdvancedParams()
        self.calibration_strategy = calibration_strategy
        self.threshold_tuning = threshold_tuning
        self.threshold_metric = threshold_metric
        self.quantiles = quantiles
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.precision = precision
        self.early_stopping_patience = early_stopping_patience
        self.val_check_interval = val_check_interval
        self.val_fraction = val_fraction
        self.cal_fraction = cal_fraction
        self.val_split_strategy = val_split_strategy
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.verbose = verbose
        self.resume_path = resume_path

    # --- abstract family seam --------------------------------------

    @abstractmethod
    def _build_backbone_head(
        self, config: BaseModelConfig, transformer: TabularToSequence
    ) -> tuple[nn.Module, nn.Module]:
        """Return ``(backbone, head)`` for the curried Trainer factory."""

    @abstractmethod
    def _encode_targets(self, y: object) -> np.ndarray:
        """Map raw ``y`` to the numeric target the loss consumes."""

    @abstractmethod
    def _make_calibrator(self) -> _Calibrator | None:
        """Construct the configured calibrator (or ``None`` for 'none')."""

    @abstractmethod
    def _set_target_fit_state(self, y: np.ndarray) -> None:
        """Set the family-specific F1.1 attrs (``classes_`` / ``quantiles_``)."""

    # --- config bridge (A4 step 4) ---------------------------------

    def _build_config(self) -> BaseModelConfig:
        if self.loss is not None:
            loss = self.loss
        else:
            # The map value is always a valid LossConfig strategy literal;
            # LossParams' Literal annotation does not narrow a str.
            loss = LossParams(strategy=_DEFAULT_LOSS_FOR_TASK[self.task_type])  # type: ignore[arg-type]
        try:
            return self._config_cls(
                task_type=self.task_type,  # type: ignore[arg-type]
                loss=loss.to_pydantic(),
                sampler=self.sampler.to_pydantic(),
                optimizer=self.optimizer.to_pydantic(),
                scheduler=self.scheduler.to_pydantic(),
                calibration_strategy=self.calibration_strategy,  # type: ignore[arg-type]
                threshold_tuning=self.threshold_tuning,
                threshold_metric=self.threshold_metric,  # type: ignore[arg-type]
                quantiles=self.quantiles,
                batch_size=self.batch_size,
                max_epochs=self.max_epochs,
                gradient_clip_val=self.gradient_clip_val,
                accumulate_grad_batches=self.accumulate_grad_batches,
                precision=self.precision,  # type: ignore[arg-type]
                early_stopping_patience=self.early_stopping_patience,
                val_check_interval=self.val_check_interval,
                val_fraction=self.val_fraction,
                cal_fraction=self.cal_fraction,
                val_split_strategy=self.val_split_strategy,  # type: ignore[arg-type]
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                seed=self.seed,
                verbose=self.verbose,
            )
        except ValidationError as exc:
            raise ConfigError(str(exc)) from exc

    # --- fit --------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: object,
        *,
        calibration_set: tuple[pd.DataFrame, object] | None = None,
        optuna_trial: object | None = None,
    ) -> "BaseSequenceEstimator":
        """Fit the transformer, model, and optional calibrator (F1 / A7 / A9).

        ``optuna_trial`` is a keyword (never a config field) so the
        pydantic ``extra="forbid"`` schema and ``get_params`` / ``save``
        stay free of it (F1 / F7). ``calibration_set`` and a positive
        ``cal_fraction`` are mutually exclusive (F2): the conflict is
        rejected here at ``fit`` time as well as inside the splitter.
        """
        if calibration_set is not None and self.cal_fraction > 0.0:
            raise ConfigError(
                "calibration_set was provided and cal_fraction > 0; pass "
                "calibration_set with cal_fraction=0, or drop it and let the "
                "three-way split carve the calibration fold (F2)"
            )
        config = self._build_config()
        # Seed before any randomness (backbone weight init in
        # model_factory, the SubsetRandomSampler in Trainer.fit) so two
        # same-seed fits in one process are bit-identical (N1). The N4
        # deterministic-algorithm gate is the Trainer's job (it fires on
        # precision='32-true' + a set seed); the estimator owns only the
        # seed thread (A7 / F5).
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        tts_config = self.tabular_config.to_pydantic()
        transformer = TabularToSequence(tts_config, self.task_type)  # type: ignore[arg-type]
        transformer.fit(X, y)

        y_encoded = self._encode_targets(y)
        self._set_target_fit_state(y_encoded)

        def model_factory() -> tuple[nn.Module, nn.Module]:
            return self._build_backbone_head(config, transformer)

        trainer = Trainer(
            transformer,
            config,
            model_factory,
            resume_path=self.resume_path,
        )
        module = trainer.fit(
            X,
            calibration_set_provided=calibration_set is not None,
            optuna_trial=optuna_trial,  # type: ignore[arg-type]
        )
        module.eval()

        self.config_ = config
        self._module = module
        self.transformer_ = transformer
        declared = [
            *tts_config.static_categorical_cols,
            *tts_config.static_real_cols,
            *tts_config.time_varying_real_cols,
            *tts_config.time_varying_categorical_cols,
        ]
        self.n_features_in_ = len(declared)
        self.feature_names_in_ = np.asarray(declared, dtype=object)
        self.n_outputs_ = 1
        self.feature_schema_fingerprint_ = transformer.feature_schema_fingerprint_

        self.calibrator_ = self._fit_calibrator(transformer, X, calibration_set)
        self._post_fit(transformer, X, calibration_set)
        return self

    def _calibration_fold(
        self,
        transformer: TabularToSequence,
        X: pd.DataFrame,  # noqa: N803
        calibration_set: tuple[pd.DataFrame, object] | None,
    ) -> tuple[Tensor, Tensor]:
        """Raw logits + targets for the held-out calibration fold (A9).

        With an explicit ``calibration_set`` the whole set is the fold.
        Otherwise the same deterministic three-way split the Trainer
        applied is recomputed (identical config + identical
        ``window_time_index`` derivation) so the calibrator fits on the
        rows training held out.
        """
        if calibration_set is not None:
            x_cal, y_cal = calibration_set
            batch = transformer.transform(x_cal)
            return self._raw_outputs(batch), torch.as_tensor(self._encode_targets(y_cal))
        batch = transformer.transform(X)
        entity_ids = batch["entity_id"].cpu().numpy()
        _, _, cal_idx = compute_three_way_split(
            entity_ids,
            _window_time_index(entity_ids),
            val_fraction=self.val_fraction,
            cal_fraction=self.cal_fraction,
            val_split_strategy=self.val_split_strategy,  # type: ignore[arg-type]
            calibration_set_provided=False,
        )
        return self._raw_outputs(batch)[cal_idx], batch["target"][cal_idx]

    def _post_fit(
        self,
        transformer: TabularToSequence,
        X: pd.DataFrame,  # noqa: N803
        calibration_set: tuple[pd.DataFrame, object] | None,
    ) -> None:
        """Family post-calibration hook (binary threshold tuner). No-op here."""

    def _raw_outputs(self, batch: dict[str, Tensor]) -> Tensor:
        """Fitted backbone + head on a transformed batch (eval, no grad)."""
        module = self._module
        with torch.no_grad():
            backbone_out = module.backbone(batch)
            logits = module.head(backbone_out.representation)
        return logits.detach().cpu()

    def _fit_calibrator(
        self,
        transformer: TabularToSequence,
        X: pd.DataFrame,  # noqa: N803
        calibration_set: tuple[pd.DataFrame, object] | None,
    ) -> _Calibrator | None:
        calibrator = self._make_calibrator()
        if calibrator is None:
            return None
        raw, targets = self._calibration_fold(transformer, X, calibration_set)
        calibrator.fit(raw, targets)
        return calibrator

    # --- save / load (A17) -----------------------------------------

    def _collect_tensors(self) -> dict[str, Tensor]:
        tensors: dict[str, Tensor] = {}
        for name, value in self._module.backbone.state_dict().items():
            tensors[f"backbone.{name}"] = value
        for name, value in self._module.head.state_dict().items():
            tensors[f"head.{name}"] = value
        return tensors

    def _collect_state(self) -> dict[str, object]:
        return {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "seq_sklearn_version": seq_sklearn.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
            "precision_resolved": self.config_.precision,
            "feature_schema_fingerprint": self.feature_schema_fingerprint_,
            "created_at": datetime.now(UTC).isoformat(),
            "task_type": self.task_type,
            "config": self.config_.model_dump(mode="json"),
            "tabular_config": self.tabular_config.to_pydantic().model_dump(mode="json"),
            "feature_names_in_": list(self.feature_names_in_),
            "n_outputs_": self.n_outputs_,
            "calibration_strategy": self.calibration_strategy,
            "tabular_to_sequence_state": self.transformer_.serialize(),
            "calibrator": (self.calibrator_.serialize() if self.calibrator_ is not None else None),
            **self._family_state(),
        }

    @abstractmethod
    def _family_state(self) -> dict[str, object]:
        """Family fit-state for ``state.json`` (``classes_`` / ``quantiles_``)."""

    @abstractmethod
    def _restore_family_state(self, state: dict[str, object]) -> None:
        """Inverse of :meth:`_family_state` on a freshly-loaded instance."""

    @abstractmethod
    def _build_calibrator_from(self, blob: dict[str, object]) -> _Calibrator:
        """Reconstruct the configured calibrator from its serialized blob."""

    def save(self, path: str | Path) -> None:
        """Write the A17 two-file directory (``weights.safetensors`` + ``state.json``)."""
        save_weights_and_state(path, self._collect_tensors(), self._collect_state())

    @classmethod
    def load(cls, path: str | Path) -> "BaseSequenceEstimator":
        """Reconstruct an estimator from the A17 directory (no pickle, A17)."""
        weights, state = load_weights_and_state(path)
        config = cls._config_cls.model_validate(state["config"])
        obj = cls.__new__(cls)
        BaseEstimator.__init__(obj)
        obj.task_type = cast("str", state["task_type"])
        obj.calibration_strategy = cast("str", state["calibration_strategy"])
        # Restore the hyperparameter attributes the prediction path and
        # the family head-sizing (`_build_backbone_head`) read; the
        # frozen config is the source of truth (no adapters on a loaded
        # instance).
        obj.quantiles = config.quantiles
        obj.threshold_tuning = config.threshold_tuning
        obj.threshold_metric = config.threshold_metric
        obj.precision = config.precision
        obj.seed = config.seed
        obj.config_ = config
        obj.feature_names_in_ = np.asarray(state["feature_names_in_"], dtype=object)
        obj.n_features_in_ = len(obj.feature_names_in_)
        obj.n_outputs_ = cast("int", state["n_outputs_"])
        obj.feature_schema_fingerprint_ = cast("str | None", state["feature_schema_fingerprint"])
        tts_config = TabularToSequenceConfig.model_validate(state["tabular_config"])
        obj.transformer_ = TabularToSequence.deserialize(
            cast("dict[str, object]", state["tabular_to_sequence_state"]),
            tts_config,
            cast("str", state["task_type"]),  # type: ignore[arg-type]
        )
        obj._restore_family_state(state)
        backbone, head = obj._build_backbone_head(config, obj.transformer_)
        backbone.load_state_dict(
            {k[len("backbone.") :]: v for k, v in weights.items() if k.startswith("backbone.")}
        )
        head.load_state_dict(
            {k[len("head.") :]: v for k, v in weights.items() if k.startswith("head.")}
        )
        backbone.eval()
        head.eval()
        obj._loaded_backbone = backbone
        obj._loaded_head = head
        cal_blob = state["calibrator"]
        obj.calibrator_ = (
            obj._build_calibrator_from(cast("dict[str, object]", cal_blob))
            if cal_blob is not None
            else None
        )
        return obj

    def _predict_module(self) -> tuple[nn.Module, nn.Module]:
        """The (backbone, head) pair to predict with (fitted or loaded)."""
        if hasattr(self, "_loaded_backbone"):
            return self._loaded_backbone, self._loaded_head
        return self._module.backbone, self._module.head

    def _predict_raw(self, X: pd.DataFrame) -> Tensor:  # noqa: N803
        """Transform ``X`` and run the prediction model (eval, no grad)."""
        self._check_fitted()
        batch = self.transformer_.transform(X)
        backbone, head = self._predict_module()
        backbone.eval()
        head.eval()
        with torch.no_grad():
            logits = head(backbone(batch).representation)
        return logits.detach().cpu()

    def _check_fitted(self) -> None:
        if not hasattr(self, "transformer_"):
            raise NotFittedError(f"{type(self).__name__} is not fitted; call fit before predict")

    # --- sklearn tags + punted methods (F1.1) ----------------------

    def __sklearn_tags__(self) -> Tags:
        # Instance method (NOT classmethod) per F1.1; chains super and
        # mutates the Tags dataclass. sklearn 1.6+ `InputTags` has no
        # `dataframe` field (the F1.1 draft pre-dated the final API);
        # the panel-DataFrame contract is expressed as "NOT a plain
        # numpy 2-D-array estimator" (two_d_array=False) plus the F2
        # no-NaN contract. non_deterministic flips True on mixed
        # precision (N5).
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = False
        tags.input_tags.allow_nan = False
        tags.target_tags.required = True
        tags.requires_fit = True
        tags.non_deterministic = self.precision in _MIXED_PRECISION
        return tags

    def partial_fit(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(_PUNTED_MSG.format(name="partial_fit"))

    def fit_predict(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(_PUNTED_MSG.format(name="fit_predict"))

    def fit_transform(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(_PUNTED_MSG.format(name="fit_transform"))
