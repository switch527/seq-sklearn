"""Library-side training wrapper (per architecture A7 / requirements F5).

:class:`Trainer` is the seam between the estimator layer (Phase 6) and
pytorch-lightning. It owns the A7 construction order: it computes the
three-way split through :mod:`seq_sklearn.data.splits`, builds the F5
DataLoaders, curries the optimizer / scheduler factories with
``functools.partial`` so the residual matches the
:class:`_LightningModule` callable type, resolves the concrete precision,
flips the N4 determinism gate, attaches the A7 callback list, wraps the
backbone / head / loss triple in a :class:`_LightningModule`, and runs
``pl_trainer.fit`` (forwarding ``resume_path`` as ``ckpt_path`` per A20
item 5).

Split semantics live in :func:`seq_sklearn.data.splits.compute_three_way_split`;
the loss / optimizer / scheduler dispatch lives in the Phase 4a
factories. This module only composes them. The ``cfg.loss`` ALPHA -> BETA
extras promotion is routed here at the ``_configure_loss`` call site
(the Phase 4b obligation recorded in the Phase 4a ledger); the
``build_sampler`` dispatch is this Trainer's job per A20 item 6.
"""

import logging
import os
from collections.abc import Callable
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from seq_sklearn.config._extras import extract_deprecated_extras
from seq_sklearn.config.base import BaseModelConfig
from seq_sklearn.data.splits import (
    below_floor_mask,
    compute_three_way_split,
    window_time_index,
)
from seq_sklearn.data.tabular_to_sequence import TabularToSequence
from seq_sklearn.errors import ConfigError
from seq_sklearn.hardware import detect
from seq_sklearn.training._determinism import enable_strict_mode
from seq_sklearn.training._lightning_module import _LightningModule
from seq_sklearn.training._precision import resolve_precision
from seq_sklearn.training.callbacks import (
    EventEmitter,
    GradScalerWatchdog,
    RngStateCallback,
)
from seq_sklearn.training.losses import build_loss
from seq_sklearn.training.optimizers import build_optimizer
from seq_sklearn.training.sampling import (
    oversample_minority,
    undersample_majority,
)
from seq_sklearn.training.schedulers import build_scheduler

__all__ = ["Trainer"]

logger = logging.getLogger(__name__)

ModelFactory = Callable[[], tuple[nn.Module, nn.Module]]


class _TensorDictDataset(Dataset[dict[str, Tensor]]):
    """Index the ``TabularToSequence.transform`` batch dict per window.

    The transform emits one big dict whose tensors share the leading
    window axis. The dataset slices every tensor at index ``i`` so the
    default collate reassembles a per-batch dict the
    :class:`_LightningModule` consumes unchanged.
    """

    def __init__(self, batch: dict[str, Tensor]) -> None:
        self._batch = batch
        self._length = next(iter(batch.values())).shape[0]

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {key: value[index] for key, value in self._batch.items()}


class Trainer:
    """Wrap pytorch-lightning around a backbone / head / loss triple (A7).

    Construct with a fitted :class:`TabularToSequence`, a frozen
    :class:`BaseModelConfig`, and a model factory returning the
    ``(backbone, head)`` pair. :meth:`fit` runs the A7 sequence end to
    end. ``resume_path`` is forwarded to ``pl_trainer.fit`` as
    ``ckpt_path`` so model / optimizer / scheduler state and (via
    :class:`RngStateCallback`) RNG state restore from a checkpoint per F5.
    """

    def __init__(
        self,
        transformer: TabularToSequence,
        config: BaseModelConfig,
        model_factory: ModelFactory,
        resume_path: str | Path | None = None,
    ) -> None:
        self.transformer = transformer
        self.config = config
        self.model_factory = model_factory
        self.resume_path = resume_path
        self._val_metric_name = "val_loss"

    def _resolve_precision(self) -> str:
        """Resolve the concrete Lightning precision string (A11 / N5)."""
        return resolve_precision(detect(), self.config.precision)

    def _deterministic(self, precision: str) -> bool:
        """A7 gate: deterministic only on ``32-true`` with a set seed.

        A7 phrases the gate as ``precision == "32-true" and seed is
        set``. ``BaseTrainingConfig.seed`` is typed ``int`` with a
        default of 42 and never ``None``, so "seed is set" is a static
        invariant of the config type and the precision check is the only
        runtime variable. Encoding the seed clause as a redundant runtime
        comparison would be dead, so the invariant is asserted in the
        config type and the gate reduces to the precision check.
        """
        return precision == "32-true"

    def _dataloader_kwargs(self) -> dict[str, object]:
        """Resolve the F5 DataLoader defaults, honoring config overrides."""
        cpu_count = os.cpu_count() or 1
        num_workers = (
            self.config.num_workers if self.config.num_workers is not None else min(4, cpu_count)
        )
        on_cuda = torch.cuda.is_available()
        pin_memory = self.config.pin_memory if self.config.pin_memory is not None else on_cuda
        return {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": num_workers > 0,
        }

    def _train_sampler(self, train_idx: np.ndarray, targets: Tensor) -> SubsetRandomSampler | None:
        """Build the F5 imbalance sampler over the training fold.

        Returns ``None`` for the ``none`` / ``class_weighted`` strategies
        (class weighting moves into the loss, not the sampler). The
        oversample / undersample strategies wrap the Phase 4a index
        builders in a :class:`SubsetRandomSampler` per A20 item 6; the
        builders draw through a seeded ``numpy`` generator so two fits
        with the same seed resample identically (N4).
        """
        strategy = self.config.sampler.strategy
        if strategy in ("none", "class_weighted"):
            return None
        labels = targets[train_idx].cpu().numpy()
        rng = np.random.default_rng(self.config.seed)
        if strategy == "oversample_minority":
            resampled = oversample_minority(
                labels,
                rng,
                oversample_ratio=self.config.sampler.oversample_ratio,
                replacement=self.config.sampler.replacement,
            )
        else:  # strategy == "undersample_majority"
            resampled = undersample_majority(
                labels, rng, replacement=self.config.sampler.replacement
            )
        global_idx = train_idx[resampled]
        return SubsetRandomSampler(global_idx.tolist())

    def _class_weights(self, train_idx: np.ndarray, targets: Tensor) -> Tensor | None:
        """F5 ``class_weighted`` weights derived from the train fold.

        Returns ``None`` for every sampler strategy except
        ``class_weighted`` (A8 architecture.md:1208, the authority for
        the train-fold-only derivation; F5 requirements.md ~786-787 only
        states "frequency-based per-class weights"). The F5 validity
        matrix guarantees ``loss.strategy == "cross_entropy"`` whenever
        this branch is reached, and ``build_loss`` re-asserts it. For
        ``binary`` the result is the scalar ``pos_weight = neg_count /
        pos_count`` consumed by ``BCEWithLogitsLoss``; for ``multiclass``
        it is the per-class inverse-frequency vector consumed by
        ``CrossEntropyLoss``. Frequencies are counted over the targets
        restricted to ``train_idx`` so the held-out folds never leak into
        the weighting, but the multiclass vocabulary is sized from the
        FULL transformed targets so a class entirely held out of the
        train fold still gets a slot (a shorter weight vector would raise
        an opaque shape mismatch on the first forward; A8 ~1209).
        """
        if self.config.sampler.strategy != "class_weighted":
            return None
        fold = targets[train_idx]
        if self.config.task_type == "binary":
            labels = fold.reshape(-1)
            pos = float((labels == 1).sum().item())
            neg = float((labels == 0).sum().item())
            if pos <= 0 or neg <= 0:
                absent = "positive" if pos <= 0 else "negative"
                logger.warning(
                    "class_weighted binary fold has no %s samples; "
                    "pos_weight falls back to 1.0 (no reweighting)",
                    absent,
                )
                return torch.tensor([1.0], dtype=torch.float32)
            return torch.tensor([neg / pos], dtype=torch.float32)
        # multiclass: per-class inverse frequency over the train fold,
        # vocabulary sized from the full transformed targets.
        n_classes = int(targets.reshape(-1).to(torch.int64).max().item()) + 1
        fold_labels = fold.reshape(-1).to(torch.int64)
        counts = torch.bincount(fold_labels, minlength=n_classes).to(torch.float32)
        return counts.sum() / torch.clamp(counts, min=1.0)

    def _configure_loss(self, class_weights: Tensor | None) -> nn.Module:
        """Build the F5 loss, routing ``cfg.loss`` through the extras alias.

        The ALPHA -> BETA ``extra`` promotion for the loss family is the
        Phase 4b call-site obligation (Phase 4a left ``build_loss``
        intentionally string-in). Routing happens here so a promoted
        ``extra`` key reaches its typed field before the strings are
        read. ``class_weights`` is the F5 ``class_weighted`` weight
        tensor (``None`` for every other strategy), threaded from
        :meth:`fit` where the train fold and transformed targets are
        both in scope (A8 ~1204-1209).
        """
        loss_cfg, extra = extract_deprecated_extras(self.config.loss, "loss")
        if extra:
            logger.warning(
                "ignoring unrecognized loss.extra key(s) %s: build_loss is "
                "string-in and consumes only typed loss fields",
                sorted(extra),
            )
        return build_loss(
            self.config.task_type,
            loss_cfg.strategy,
            class_weights=class_weights,
            focal_gamma=loss_cfg.focal_gamma,
            huber_delta=loss_cfg.huber_delta,
            quantiles=self.config.quantiles,
        )

    def _total_steps(self, n_train: int) -> int | None:
        """Accumulation-adjusted optimizer-step count for step schedulers.

        ``None`` for the non-step schedulers (``constant`` /
        ``reduce_on_plateau``); the step-horizon schedulers
        (``one_cycle`` / ``cosine_with_warmup``) derive ``total_steps``
        from ``max_epochs * ceil(batches / accumulate_grad_batches)`` per
        A20 item 1 so the schedule stays correct under gradient
        accumulation.
        """
        if self.config.scheduler.name in ("constant", "reduce_on_plateau"):
            return None
        if n_train == 0:
            raise ConfigError(
                "a step-horizon scheduler was requested but the training "
                "fold is empty; total_steps cannot be derived (A20 item 1)."
            )
        batches_per_epoch = -(-n_train // self.config.batch_size)
        opt_steps_per_epoch = -(-batches_per_epoch // self.config.accumulate_grad_batches)
        return self.config.max_epochs * opt_steps_per_epoch

    def build_pl_trainer(self, precision: str) -> pl.Trainer:
        """Construct the A7 ``pl.Trainer``: precision, gate, callbacks.

        ``enable_strict_mode`` is called when the deterministic gate is on
        so the four N4 process globals are guaranteed beyond Lightning's
        own ``deterministic=True`` (which sets only
        ``use_deterministic_algorithms``). ``logger=False`` suppresses
        Lightning's auto-attached ``TensorBoardLogger``; callers attach
        their own.
        """
        deterministic = self._deterministic(precision)
        if deterministic:
            enable_strict_mode()
        callbacks: list[pl.Callback] = [
            EarlyStopping(
                monitor=self._val_metric_name,
                patience=self.config.early_stopping_patience,
            ),
            ModelCheckpoint(save_last=True, save_top_k=1, monitor=self._val_metric_name),
            GradScalerWatchdog(),
            EventEmitter(),
            RngStateCallback(),
        ]
        return pl.Trainer(
            max_epochs=self.config.max_epochs,
            precision=precision,  # type: ignore[arg-type]
            deterministic=deterministic,
            gradient_clip_val=self.config.gradient_clip_val,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            val_check_interval=self.config.val_check_interval,
            callbacks=callbacks,
            logger=False,
            enable_progress_bar=self.config.verbose,
        )

    def _build_module(
        self,
        n_train: int,
        class_weights: Tensor | None,
        optuna_trial: optuna.trial.BaseTrial | None,
    ) -> _LightningModule:
        """Wire the curried factories and wrap the model triple (A7 step 3/4).

        ``optuna_trial`` threads from :meth:`fit` into the
        :class:`_LightningModule` constructor (A16 ~1990-1994: the trial
        reaches the module via ``fit``, never via the pydantic config).
        """
        backbone, head = self.model_factory()
        loss = self._configure_loss(class_weights)
        optimizer_factory = partial(build_optimizer, config=self.config.optimizer)
        scheduler_factory = partial(
            build_scheduler,
            config=self.config.scheduler,
            monitor=self._val_metric_name,
            total_steps=self._total_steps(n_train),
        )
        return _LightningModule(
            backbone=backbone,  # type: ignore[arg-type]
            head=head,
            loss=loss,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            val_metric_name=self._val_metric_name,
            optuna_trial=optuna_trial,
        )

    def _below_floor_mask(self, entity_ids: np.ndarray) -> np.ndarray:
        """Below-`min_periods_predict` window mask (shared `splits` impl).

        Delegates to :func:`seq_sklearn.data.splits.below_floor_mask`,
        the single source the estimator's calibration-fold / predict
        path also uses, so the Trainer's train / val drop and the
        estimator's drop cannot diverge.
        """
        return below_floor_mask(entity_ids, self.transformer.config.min_periods_predict)

    def fit(
        self,
        x_panel: object,
        *,
        calibration_set_provided: bool = False,
        optuna_trial: optuna.trial.BaseTrial | None = None,
    ) -> _LightningModule:
        """Run the A7 sequence: split, loaders, module, ``pl_trainer.fit``.

        ``x_panel`` is the validated panel the fitted transformer accepts.
        Returns the fitted :class:`_LightningModule` so the estimator
        layer (Phase 6) can build the calibrator on the held-out fold.
        ``resume_path`` threads to ``pl_trainer.fit(ckpt_path=...)`` per
        A20 item 5; :class:`RngStateCallback` restores RNG state on that
        path so a resumed run continues bit-identically. ``optuna_trial``
        threads into the :class:`_LightningModule` constructor (A16
        ~1990-1994: via ``fit``, never the config) for the deferred-prune
        hook.
        """
        batch = self.transformer.transform(x_panel)  # type: ignore[arg-type]
        entity_ids = batch["entity_id"].cpu().numpy()
        window_time_index = self._window_time_index(entity_ids)

        train_idx, val_idx, _cal_idx = compute_three_way_split(
            entity_ids,
            window_time_index,
            val_fraction=self.config.val_fraction,
            cal_fraction=self.config.cal_fraction,
            val_split_strategy=self.config.val_split_strategy,
            calibration_set_provided=calibration_set_provided,
        )

        # Entities with fewer windows than min_periods_predict carry
        # sentinel targets (-1 classification / NaN regression) from
        # TabularToSequence.transform. They must not enter the train /
        # val folds: torch.bincount on -1 raises in _class_weights, and
        # NaN regression targets trip the F9 non-finite-loss abort. The
        # estimator already drops them from the recomputed cal fold;
        # filter them here so the Trainer never fits on a sentinel.
        below = self._below_floor_mask(entity_ids)
        pre_train, pre_val = train_idx.size, val_idx.size
        train_idx = train_idx[~below[train_idx]]
        val_idx = val_idx[~below[val_idx]]
        # Scoped to the below-floor cause only: raise when a fold that
        # was non-empty BEFORE the filter is emptied BY it (symmetric
        # with the estimator's empty-calibration-fold guard). A fold
        # already empty pre-filter (e.g. val_fraction=0) is a distinct,
        # pre-existing situation and is left to the prior behaviour so
        # this message stays accurate to its actual cause.
        if (pre_train > 0 and train_idx.size == 0) or (pre_val > 0 and val_idx.size == 0):
            raise ConfigError(
                "the train / val fold is empty after dropping below-floor "
                f"windows: every entity has fewer than min_periods_predict="
                f"{self.transformer.config.min_periods_predict} rows. Lower "
                "min_periods_predict or supply longer-tenure entities."
            )

        class_weights = self._class_weights(train_idx, batch["target"])

        dataset = _TensorDictDataset(batch)
        loader_kwargs = self._dataloader_kwargs()
        sampler = self._train_sampler(train_idx, batch["target"])
        if sampler is None:
            train_loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                sampler=SubsetRandomSampler(train_idx.tolist()),
                **loader_kwargs,  # type: ignore[arg-type]
            )
        else:
            train_loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                sampler=sampler,
                **loader_kwargs,  # type: ignore[arg-type]
            )
        val_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=SubsetRandomSampler(val_idx.tolist()),
            **loader_kwargs,  # type: ignore[arg-type]
        )

        precision = self._resolve_precision()
        pl_trainer = self.build_pl_trainer(precision)
        module = self._build_module(len(train_idx), class_weights, optuna_trial)
        ckpt_path = str(self.resume_path) if self.resume_path is not None else None
        pl_trainer.fit(module, train_loader, val_loader, ckpt_path=ckpt_path)
        return module

    @staticmethod
    def _window_time_index(entity_ids: np.ndarray) -> np.ndarray:
        """Delegate to :func:`seq_sklearn.data.splits.window_time_index`.

        Single source of truth shared with the estimator's
        calibration-fold seam so the two cannot drift; kept as a thin
        Trainer-private alias for the existing call site / tests.
        """
        return window_time_index(entity_ids)
