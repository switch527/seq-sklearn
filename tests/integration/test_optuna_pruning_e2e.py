"""End-to-end Optuna pruning (Phase 8, A16 / A7 deferred-raise / N1).

Pins the full pruning loop through the real estimator: the
LightningModule reports ``val_loss`` in ``on_validation_epoch_end``,
stashes the prune decision, and raises ``optuna.TrialPruned`` at the
END of ``on_train_epoch_end`` (A7) so the ``train.epoch`` events still
fire for the pruned epoch. Uses the architecture-mandated
``MedianPruner(n_startup_trials=0, n_warmup_steps=0, n_min_trials=1)``;
``n_min_trials=1`` is mandatory or the second trial has no median to
compare against and the test goes flaky. Determinism comes from the
pinned seed plus a healthy-vs-crippled learning-rate split, so trial 0
completes with a low ``val_loss`` and trial 1 reports a worse epoch-0
``val_loss`` than that median and is pruned. CPU-pinned.
"""

import numpy as np
import optuna
import pandas as pd
import pytest
import torch

from seq_sklearn.config._adapters import (
    OptimizerParams,
    SchedulerParams,
    TabularConfigParams,
)
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier
from seq_sklearn.tuning.pruning import optuna_trial_guard

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _panel() -> tuple[SyntheticPanelGenerator, pd.DataFrame, np.ndarray]:
    gen = SyntheticPanelGenerator(
        target_kind="binary",
        num_entities=20,
        periods_per_entity=24,
        signal_strength=0.7,
        lookback=8,
        seed=42,
    )
    panel, y = gen.generate(seed=42)
    return gen, panel, y


def _tab(gen: SyntheticPanelGenerator) -> TabularConfigParams:
    return TabularConfigParams(
        id_col=gen.id_col,
        time_col=gen.time_col,
        static_categorical_cols=tuple(gen.static_categorical_cols),
        static_real_cols=tuple(gen.static_real_cols),
        time_varying_real_cols=tuple(gen.time_varying_real_cols),
        time_varying_categorical_cols=tuple(gen.time_varying_categorical_cols),
        lookback=gen.lookback,
        min_periods=1,
        min_periods_predict=1,
        max_categorical_cardinality=10_000,
    )


def test_optuna_median_pruner_prunes_second_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    gen, x, y = _panel()

    def objective(trial: optuna.trial.Trial) -> float:
        with optuna_trial_guard(trial):
            # Trial 0 learns (its low val_loss becomes the median);
            # trial 1's near-frozen weights report a worse epoch-0
            # val_loss and trip MedianPruner.
            lr = 1e-2 if trial.number == 0 else 1e-12
            est = TFTClassifier(
                task_type="binary",
                tabular_config=_tab(gen),
                optimizer=OptimizerParams(learning_rate=lr),
                scheduler=SchedulerParams(name="constant", warmup_steps=0),
                hidden_size=8,
                attention_heads=2,
                max_epochs=2,
                batch_size=32,
                val_fraction=0.2,
                cal_fraction=0.0,
                precision="32-true",
                verbose=False,
                seed=42,
            )
            est.fit(x, y, optuna_trial=trial)
            return 0.0

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=0, n_min_trials=1),
    )
    study.optimize(objective, n_trials=2, catch=())

    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[1].state == optuna.trial.TrialState.PRUNED
