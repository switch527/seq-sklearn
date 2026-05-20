"""Shared TFT-family estimator wiring (A4 step 3 / A2, Phase 7).

`TFTClassifier` and `TFTRegressor` differ only in the head and the
classifier / regressor family base they compose. The A4-adapter
``__init__``, the ``TFTConfig`` bridge (``_config_cls`` /
``_config_kwargs``), and the fitted-``TabularToSequence`` ->
``TFTBackbone`` wiring are identical, so they live here once as a mixin
both concrete classes compose, instead of being duplicated per file
(the same single-source discipline as
``data.splits.window_time_index`` / ``below_floor_mask``).
"""

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from seq_sklearn.config.adapters import (
    LossParams,
    OptimizerParams,
    SamplerParams,
    SchedulerParams,
    TabularConfigParams,
    TFTAdvancedParams,
)
from seq_sklearn.config.base import BaseModelConfig
from seq_sklearn.config.tft import TFTConfig
from seq_sklearn.data.tabular_to_sequence import TabularToSequence
from seq_sklearn.models.transformer.tft.backbone import TFTBackbone

if TYPE_CHECKING:
    from seq_sklearn.models._base import BaseSequenceEstimator

    # Type-check time only: gives pyright the cooperative
    # `super().__init__` signature + the inherited adapter attributes.
    # At runtime this is a bare `object` mixin; `super()` resolves
    # through the composed class's real MRO to
    # `BaseSequenceEstimator.__init__` regardless of the static base.
    _MixinBase = BaseSequenceEstimator
else:
    _MixinBase = object

__all__ = ["_TFTEstimatorMixin"]


class _TFTEstimatorMixin(_MixinBase):
    """A4 step-3 ``__init__`` + ``TFTConfig`` bridge + ``TFTBackbone`` wiring.

    Bare cooperative mixin. The composed MRO is ``TFT* ->
    _TFTEstimatorMixin -> TransformerSequenceEstimator.<Family> ->
    BaseSequence<Family> -> ... -> BaseSequenceEstimator``; this
    ``__init__`` stores the TFT model-shape params verbatim, then
    delegates the base param surface to
    :meth:`BaseSequenceEstimator.__init__` through ``super()`` (which
    applies the verbatim / None-default-instance idiom). sklearn
    ``get_params`` / ``clone`` introspect this signature, so every
    param is an explicit named keyword-only argument.
    """

    _config_cls: ClassVar[type[BaseModelConfig]] = TFTConfig

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
        hidden_size: int = 128,
        attention_heads: int = 4,
        dropout: float = 0.1,
        variable_selection_dropout: float = 0.1,
        prediction_readout: str = "last_valid",
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
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.variable_selection_dropout = variable_selection_dropout
        self.prediction_readout = prediction_readout
        super().__init__(
            task_type=task_type,
            tabular_config=tabular_config,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            sampler=sampler,
            advanced=advanced,
            calibration_strategy=calibration_strategy,
            threshold_tuning=threshold_tuning,
            threshold_metric=threshold_metric,
            quantiles=quantiles,
            batch_size=batch_size,
            max_epochs=max_epochs,
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches,
            precision=precision,
            early_stopping_patience=early_stopping_patience,
            val_check_interval=val_check_interval,
            val_fraction=val_fraction,
            cal_fraction=cal_fraction,
            val_split_strategy=val_split_strategy,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
            verbose=verbose,
            resume_path=resume_path,
        )

    def _config_kwargs(self) -> dict[str, object]:
        """TFTConfig extras on top of the shared BaseModelConfig surface.

        ``TFTConfig`` requires ``tabular_config`` and adds the
        ``advanced`` sub-config plus the flat model-shape fields;
        ``_build_config`` already assembles every shared
        ``BaseModelConfig`` field, so only the TFT-specific ones go here.
        """
        return {
            "tabular_config": self.tabular_config.to_pydantic(),
            "advanced": self.advanced.to_pydantic(),
            "hidden_size": self.hidden_size,
            "attention_heads": self.attention_heads,
            "dropout": self.dropout,
            "variable_selection_dropout": self.variable_selection_dropout,
            "prediction_readout": self.prediction_readout,
        }

    def _build_tft_backbone(
        self,
        config: BaseModelConfig,
        transformer: TabularToSequence,
    ) -> TFTBackbone:
        """Construct the TFT backbone from the FITTED transformer.

        The categorical cardinalities and real-column counts are read
        off the fitted ``TabularToSequence`` (``cardinalities_`` is
        keyed by column name; the per-side ordering follows the
        transformer config's column tuples), so the embedding table
        widths match the encoder vocab exactly.
        """
        tft_config = cast("TFTConfig", config)
        tcfg = transformer.config
        return TFTBackbone(
            tft_config,
            static_cat_cardinalities=[
                transformer.cardinalities_[c] for c in tcfg.static_categorical_cols
            ],
            tv_cat_cardinalities=[
                transformer.cardinalities_[c] for c in tcfg.time_varying_categorical_cols
            ],
            n_static_real=len(tcfg.static_real_cols),
            n_tv_real=len(tcfg.time_varying_real_cols),
        )
