"""``BaseSequenceEstimator``: the sklearn-contract estimator shell (A2 / A4 / A17).

The estimator owns mutable scalar attributes plus six nested
``BaseEstimator`` adapters (A4 step 1/3). The frozen pydantic config is
built inside :meth:`fit` as ``self.config_`` (A4 step 2); the
task-type-aware loss default is injected there when the ``loss`` adapter
is omitted. ``fit`` plumbs :class:`TabularToSequence` -> :class:`Trainer`
(curried-factory pattern, A7), then fits the optional calibrator on the
held-out calibration fold (A9). Save / load is the A17 two-file format.

The class is abstract: a concrete family supplies ``_config_cls``,
``_config_kwargs`` and :meth:`_build_backbone_head`; the classifier /
regressor mix-ins overlay ``predict*`` and the F1.1 fit-state
attributes.
"""

import platform
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

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
from seq_sklearn.data.splits import (
    below_floor_mask,
    compute_three_way_split,
    window_time_index,
)
from seq_sklearn.data.tabular_to_sequence import TabularToSequence
from seq_sklearn.errors import (
    ConfigError,
    DataContractError,
    NotFittedError,
    OnnxExportError,
)
from seq_sklearn.models._backbone import BackboneOutput
from seq_sklearn.serialization import (
    CURRENT_SCHEMA_VERSION,
    load_weights_and_state,
    save_weights_and_state,
)
from seq_sklearn.training._lightning_module import _LightningModule
from seq_sklearn.training.trainer import Trainer

if TYPE_CHECKING:
    import optuna

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


class BaseSequenceEstimator(BaseEstimator, ABC):
    """Abstract sklearn estimator shell shared by every model family.

    ``__sklearn_tags__`` reads raw instance attributes (never
    ``self.config_``) because ``check_estimator`` invokes it on
    unconfigured instances (A4 step 2).
    """

    # Concrete subclasses override this with their config class (e.g.
    # the TFT family's TFTConfig); the base / dummy use BaseModelConfig.
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

    def _config_kwargs(self) -> dict[str, object]:
        """Family-specific extra kwargs for ``_config_cls`` construction.

        The base / dummy build a plain ``BaseModelConfig`` and add
        nothing. A concrete family whose ``_config_cls`` adds required
        fields (the TFT family's ``TFTConfig`` declares the required
        ``tabular_config`` plus model-shape fields, ``extra='forbid'``)
        overrides this so ``_build_config`` stays a single shared
        implementation instead of a copy-paste per family.
        """
        return {}

    def _build_config(self) -> BaseModelConfig:
        if self.loss is not None:
            loss = self.loss
        else:
            # The map value is always a valid LossConfig strategy literal;
            # LossParams' Literal annotation does not narrow a str.
            loss = LossParams(strategy=_DEFAULT_LOSS_FOR_TASK[self.task_type])  # type: ignore[arg-type]
        shared: dict[str, object] = {
            "task_type": self.task_type,
            "loss": loss.to_pydantic(),
            "sampler": self.sampler.to_pydantic(),
            "optimizer": self.optimizer.to_pydantic(),
            "scheduler": self.scheduler.to_pydantic(),
            "calibration_strategy": self.calibration_strategy,
            "threshold_tuning": self.threshold_tuning,
            "threshold_metric": self.threshold_metric,
            "quantiles": self.quantiles,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "gradient_clip_val": self.gradient_clip_val,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "precision": self.precision,
            "early_stopping_patience": self.early_stopping_patience,
            "val_check_interval": self.val_check_interval,
            "val_fraction": self.val_fraction,
            "cal_fraction": self.cal_fraction,
            "val_split_strategy": self.val_split_strategy,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "seed": self.seed,
            "verbose": self.verbose,
        }
        try:
            return self._config_cls(**shared, **self._config_kwargs())  # type: ignore[arg-type]
        except ValidationError as exc:
            raise ConfigError(str(exc)) from exc

    # --- fit --------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: object,
        *,
        calibration_set: tuple[pd.DataFrame, object] | None = None,
        optuna_trial: "optuna.trial.BaseTrial | None" = None,
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
        needs_fold = self._make_calibrator() is not None or self.threshold_tuning
        if needs_fold and calibration_set is None and self.cal_fraction <= 0.0:
            raise ConfigError(
                f"calibration_strategy={self.calibration_strategy!r} / "
                f"threshold_tuning={self.threshold_tuning} needs a calibration "
                "fold but cal_fraction=0 and no calibration_set was given; set "
                "cal_fraction > 0 or pass calibration_set (F2)"
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
        # Encode targets BEFORE fitting the transformer: TabularToSequence
        # builds a numeric (id, time) -> label map (it casts y with
        # float()), so a classifier's raw string / object labels must be
        # LabelEncoded first. _set_target_fit_state pins classes_ /
        # quantiles_ off the encoded y.
        y_encoded = self._encode_targets(y)
        self._set_target_fit_state(y_encoded)
        tts_config = self.tabular_config.to_pydantic()
        transformer = TabularToSequence(tts_config, self.task_type)  # type: ignore[arg-type]
        transformer.fit(X, y_encoded)

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
            optuna_trial=optuna_trial,
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

        Two branches with deliberately asymmetric below-floor handling:

        - Explicit ``calibration_set``: the whole caller-provided set is
          the fold and the targets are the caller's REAL ``y_cal``
          (via ``_encode_targets``), never ``transform``'s sentinels.
          Below-floor entities are intentionally NOT filtered here: the
          caller owns the composition of an explicit calibration set,
          and there is no sentinel-label hazard because ``y_cal`` is
          real. (Whether to additionally exclude short-history entities
          from a caller-supplied set is a separate input-validation
          concern, deferred; see implementation_plan Deferred.)
        - Recomputed fold (no explicit set): the same deterministic
          three-way split the Trainer applied is recomputed so the
          calibrator fits on the rows training held out. Here the
          targets are ``batch["target"]`` straight from ``transform``,
          so below-floor windows (``min_periods <= n <
          min_periods_predict``) carry sentinel targets (``-1``
          classification / ``NaN`` regression). They MUST be dropped
          (the ``keep`` filter) so the calibrator / threshold tuner
          never fit on a sentinel label.
        """
        if calibration_set is not None:
            x_cal, y_cal = calibration_set
            # Fail fast on a length mismatch (one window per row, so
            # len(raw) == len(x_cal)); without this the mispaired
            # arrays fail late and cryptically inside the calibrator /
            # threshold-tuner fit instead of here.
            n_x, n_y = len(x_cal), np.asarray(y_cal).shape[0]
            if n_x != n_y:
                raise DataContractError(
                    f"explicit calibration_set length mismatch: x_cal has "
                    f"{n_x} rows but y_cal has {n_y}"
                )
            batch = transformer.transform(x_cal)
            # transform() returns raw outputs in sorted (id, time) order;
            # y_cal is in caller order. Reorder the raw outputs to caller
            # order so the calibrator / threshold tuner always sees
            # (caller-order outputs, caller-order y_cal) (F1/F2, Gemini
            # G-C2). Pinned here so BOTH consumers (the calibrator-fit
            # path and the classifier _post_fit threshold path) inherit
            # the fix.
            raw = self._raw_outputs(batch)[batch["input_row_order"]]
            return raw, torch.as_tensor(self._encode_targets(y_cal))
        # Internal-split branch: raw outputs, cal_idx, and batch["target"]
        # are all in the SAME sorted space and indexed by the same
        # sorted-space `keep`, so the pairing is already self-consistent.
        # input_row_order must NOT be applied here: doing so would
        # mis-order a currently-correct pairing.
        batch = transformer.transform(X)
        entity_ids = batch["entity_id"].cpu().numpy()
        _, _, cal_idx = compute_three_way_split(
            entity_ids,
            window_time_index(entity_ids),
            val_fraction=self.val_fraction,
            cal_fraction=self.cal_fraction,
            val_split_strategy=self.val_split_strategy,  # type: ignore[arg-type]
            calibration_set_provided=False,
        )
        keep = cal_idx[~self._below_floor_mask(batch)[cal_idx]]
        if keep.size == 0:
            raise ConfigError(
                "the recomputed calibration fold is empty after dropping "
                "below-floor windows: every calibration-fold entity has "
                "fewer than min_periods_predict="
                f"{self.transformer_.config.min_periods_predict} rows. Lower "
                "min_periods_predict, raise cal_fraction, or pass an explicit "
                "calibration_set (F2)"
            )
        return self._raw_outputs(batch)[keep], batch["target"][keep]

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

    def _hyperparams(self) -> dict[str, object]:
        """The full ``__init__`` surface as JSON primitives.

        ``get_params(deep=True)`` flattens every adapter to scalar
        leaf keys; the adapter OBJECTS themselves are dropped (they
        rebuild from the leaf keys via ``set_params`` on load). Tuples
        become lists and ``Path`` becomes ``str`` through JSON; both
        round-trip back into the estimator without behaviour change.
        Persisting this so a loaded estimator's ``get_params`` /
        ``sklearn.base.clone`` work (F1) rather than AttributeError on
        the un-restored constructor params.
        """
        out: dict[str, object] = {}
        for key, value in self.get_params(deep=True).items():
            if isinstance(value, BaseEstimator):
                continue
            out[key] = str(value) if isinstance(value, Path) else value
        return out

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
            "hyperparams": self._hyperparams(),
            # exclude=tabular_config: a no-op for BaseModelConfig (no
            # such field) and the de-dup for TFTConfig (which embeds it)
            # so the flat `tabular_config` key below is the single
            # authoritative transformer-config dump (A17).
            "config": self.config_.model_dump(mode="json", exclude={"tabular_config"}),
            "tabular_config": self.tabular_config.to_pydantic().model_dump(mode="json"),
            "feature_names_in_": list(self.feature_names_in_),
            "n_outputs_": self.n_outputs_,
            # calibration_strategy is NOT a top-level key: it round-trips
            # through `hyperparams` (an __init__ scalar) and load()
            # restores it via set_params, so a duplicate here would be a
            # dead, migration-confusing field (A17).
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
        # `_collect_state` dumps `config` with exclude={"tabular_config"}
        # (A17: the transformer config is persisted once, as the flat
        # `tabular_config` key, the single authoritative copy). A
        # `_config_cls` that embeds `tabular_config` as a required field
        # (the TFT family's `TFTConfig`) must have it merged back before
        # validation; `BaseModelConfig` has no such field so this is a
        # no-op for the base / dummy path.
        config_state = cast("dict[str, object]", state["config"])
        if "tabular_config" in cls._config_cls.model_fields:
            config_state = {**config_state, "tabular_config": state["tabular_config"]}
        config = cls._config_cls.model_validate(config_state)
        # Reconstruct through the real constructor + set_params so the
        # full hyperparameter surface (every __init__ scalar AND the six
        # rebuilt adapters) is restored: a loaded estimator's
        # get_params(deep=True) / sklearn.base.clone then work (F1).
        hyperparams = cast("dict[str, object]", state["hyperparams"])
        obj = cls(task_type=cast("str", state["task_type"]))
        # `loss` is the one adapter __init__ stores verbatim as None
        # (F5 task-aware default injection needs to know the user did
        # not specify it). Every other adapter defaults to a real
        # instance, so set_params can route their `prefix__leaf` keys.
        # If a loss adapter WAS specified at save, _hyperparams persisted
        # its `loss__*` leaves but dropped the bare `loss` object;
        # set_params would then route `loss__strategy` to None. Restore a
        # default LossParams first so the leaves rebuild onto it.
        if obj.loss is None and any(k.startswith("loss__") for k in hyperparams):
            obj.loss = LossParams()
        obj.set_params(**hyperparams)
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

    def export_onnx(self, path: str | Path, X: pd.DataFrame) -> None:  # noqa: N803
        """Export the fitted model FORWARD to ONNX (F1, BETA).

        The exported graph is backbone + head producing RAW HEAD
        LOGITS ``(B, K)`` only. Calibration, threshold tuning,
        below-floor NaN-fill, and the F1 caller-input-row-order
        restore are sklearn-side numpy post-processing applied by
        ``predict`` / ``predict_proba`` / ``predict_quantiles`` and are
        deliberately NOT in the ONNX graph. To reproduce
        ``predict_proba`` from the ONNX output, apply sigmoid/softmax
        and the fitted calibration in the serving layer.

        The graph trusts its inputs: the eager input-validation guards
        (non-finite, embedding-index bounds, left-pad contract,
        all-padding row) are SKIPPED on the export path via the
        backbone's ``onnx_export`` flag (set ``True`` only on
        ``_OnnxForward``'s private deepcopy of the backbone, never on
        the shared instance), since ONNX has no exception
        mechanism. Validation remains the responsibility of the eager
        ``predict`` path (where the flag is False and every guard
        runs) and the caller.

        Dynamic batch ``B`` is supported; the time dimension is the
        fit-time constant ``lookback``. ``X`` is required (a fitted
        ``TabularToSequence`` schema-valid panel; one window per row
        is emitted, mirroring ``predict``).

        Args:
            path: destination ``.onnx`` file path.
            X: a schema-valid panel; its ``transform`` supplies the
                example inputs (include a short, left-padded entity to
                trace the mask path).

        Raises:
            NotFittedError: called before ``fit``.
            OnnxExportError: the optional ``[onnx]`` extra is absent,
                or ``torch.export`` / ``torch.onnx.export`` could not
                lower the model.
        """
        self._check_fitted()
        import importlib

        try:
            # Dynamic import: presence-check the optional [onnx] extra
            # while preserving the ImportError as __cause__ (a unit
            # test pins the message + __cause__). String import keeps
            # pyright/ruff clean without the extra installed.
            importlib.import_module("onnx")
            importlib.import_module("onnxruntime")
        except ImportError as e:
            raise OnnxExportError(
                "ONNX export requires the optional extra: pip install seq-sklearn[onnx]"
            ) from e

        from torch.nn.attention import SDPBackend, sdpa_kernel

        from seq_sklearn.inference.onnx import _OnnxForward

        backbone, head = self._predict_module()
        batch = self.transformer_.transform(X)
        args = (
            batch["static_categorical"],
            batch["static_real"],
            batch["time_varying_real"],
            batch["time_varying_categorical"],
            batch["padding_mask"],
        )
        # _OnnxForward deepcopies backbone/head and sets onnx_export
        # only on its private clone, so the trace captures the
        # gather-preserving path while the shared backbone the
        # estimator predicts with is never mutated (thread-safe).
        module = _OnnxForward(backbone, head)
        batch_dim = torch.export.Dim("batch")
        dynamic_shapes = ({0: batch_dim},) * 5

        try:
            with sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
                exported = torch.export.export(
                    module, args, dynamic_shapes=dynamic_shapes, strict=False
                )
                torch.onnx.export(
                    exported,
                    args,
                    str(path),
                    dynamo=True,
                    opset_version=20,
                    external_data=False,
                )
        except Exception as e:
            raise OnnxExportError(f"ONNX export failed during lowering: {e}") from e

    def _predict_module(self) -> tuple[nn.Module, nn.Module]:
        """The (backbone, head) pair to predict with (fitted or loaded)."""
        if hasattr(self, "_loaded_backbone"):
            return self._loaded_backbone, self._loaded_head
        return self._module.backbone, self._module.head

    def _forward_backbone(
        self,
        X: pd.DataFrame,  # noqa: N803
    ) -> tuple[BackboneOutput, nn.Module, dict[str, Tensor], np.ndarray]:
        """Transform ``X`` and run the backbone (eval, no grad).

        Returns ``(backbone_output, head, batch, below_floor_mask)``.
        The full :class:`BackboneOutput` (not just ``representation``)
        is returned so the transformer family's
        ``predict_with_attention`` can read the introspection tensors
        from the SAME forward pass ``_predict_raw`` uses, rather than a
        second pass that could disagree.

        The predict-time schema fingerprint must match the fit-time one
        (F4): a column-set / dtype / config drift between fit and
        predict is rejected with :class:`DataContractError` rather than
        silently producing wrong predictions.
        """
        self._check_fitted()
        predict_fingerprint = self.transformer_._build_fingerprint(X)
        if predict_fingerprint != self.feature_schema_fingerprint_:
            raise DataContractError(
                "predict X schema does not match the fit-time schema "
                "(feature_schema_fingerprint mismatch); the declared "
                "columns / dtypes must be identical to fit (F4)"
            )
        batch = self.transformer_.transform(X)
        backbone, head = self._predict_module()
        backbone.eval()
        head.eval()
        with torch.no_grad():
            output = backbone(batch)
        return output, head, batch, self._below_floor_mask(batch)

    def _predict_raw(self, X: pd.DataFrame) -> tuple[Tensor, np.ndarray, np.ndarray]:  # noqa: N803
        """Backbone + head logits, below-floor mask, caller-order restore.

        Returns ``(raw, below, input_row_order)``. ``raw`` and ``below``
        are in transform (sorted) row order; ``input_row_order`` is the
        permutation that indexes an emission-order array back to the
        caller's input ``X`` row order (F1). Callers apply it as the
        last step, AFTER the sorted-space below-floor NaN-fill.
        """
        output, head, batch, below = self._forward_backbone(X)
        with torch.no_grad():
            logits = head(output.representation)
        input_row_order = batch["input_row_order"].cpu().numpy()
        return logits.detach().cpu(), below, input_row_order

    def _below_floor_mask(self, batch: dict[str, Tensor]) -> np.ndarray:
        """Per-output-row mask of entities below ``min_periods_predict``.

        ``TabularToSequence.transform`` emits exactly ``n`` windows for
        an ``n``-row entity, so the per-entity-code window count equals
        the entity's row count; a code whose count is below
        ``min_periods_predict`` is a below-floor entity. Those output
        rows are NaN-filled by the family predict so a short entity
        never gets a silently-wrong finite prediction (requirements
        F NaN-in-output: predict_proba / predict_quantiles return
        NaN of the correct shape, never zero-filled). The aggregated
        breach WARNING is emitted once per call by ``transform`` itself.
        """
        entity_ids = batch["entity_id"].cpu().numpy()
        return below_floor_mask(entity_ids, self.transformer_.config.min_periods_predict)

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
