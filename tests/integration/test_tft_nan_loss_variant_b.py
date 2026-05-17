"""N1 NaN-loss guard, Variant B against the real TFT (Phase 7, F9).

Variant A (Phase 4, dummy estimator: monkey-patched loss) already
landed. Variant B injects ``Inf`` into the backbone weights before the
first step so the F9 abort fires from the NATURAL NaN-propagation route
(Inf weights -> Inf activations -> non-finite loss), not a patched
loss. Three consecutive non-finite steps must raise ``TrainingError``.

Isolation (A14): the estimator is function-local and fresh, and the
corruption is applied through a pytest ``monkeypatch`` of
``_build_backbone_head`` (auto-undone at teardown), so no shared state
leaks; the clean backbone ``state_dict`` is additionally snapshotted
and restored in a ``finally`` as defence in depth.
"""

import numpy as np
import pytest
import torch
from torch import nn

from seq_sklearn.config._adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.errors import TrainingError
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def test_tft_inf_weights_aborts_with_training_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    gen = SyntheticPanelGenerator(
        target_kind="binary",
        num_entities=20,
        periods_per_entity=24,
        signal_strength=0.7,
        lookback=8,
        seed=42,
    )
    x, y = gen.generate(seed=42)
    est = TFTClassifier(
        task_type="binary",
        tabular_config=TabularConfigParams(
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
        ),
        scheduler=SchedulerParams(name="constant", warmup_steps=0),
        hidden_size=16,
        attention_heads=2,
        max_epochs=3,
        batch_size=16,
        val_fraction=0.2,
        cal_fraction=0.0,
        precision="32-true",
        verbose=False,
        seed=42,
    )

    real_build = est._build_backbone_head
    clean_state: dict[str, torch.Tensor] = {}

    def _corrupt(config: object, transformer: object) -> tuple[nn.Module, nn.Module]:
        backbone, head = real_build(config, transformer)  # type: ignore[arg-type]
        clean_state.update({k: v.detach().clone() for k, v in backbone.state_dict().items()})
        with torch.no_grad():
            # Inf one real parameter at step 0; Inf weights propagate to
            # a non-finite loss through the natural forward path.
            next(iter(backbone.parameters())).fill_(float("inf"))
        return backbone, head

    monkeypatch.setattr(est, "_build_backbone_head", _corrupt)
    try:
        with pytest.raises(TrainingError, match="non-finite"):
            est.fit(x, y)
    finally:
        if clean_state and hasattr(est, "_module"):
            est._module.backbone.load_state_dict(clean_state)
    assert np.isfinite(np.asarray(y)).all()  # the panel itself was clean
