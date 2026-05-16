"""Tests for the abstract backbone base (A15)."""

import torch

from seq_sklearn.models._backbone import BackboneOutput, BaseBackbone


class _MinimalBackbone(BaseBackbone):
    """Overrides only forward; inherits the default metric method."""

    def forward(self, batch: dict[str, torch.Tensor]) -> BackboneOutput:
        return BackboneOutput(
            representation=batch["x"],
            padding_mask=batch["mask"],
        )


def test_base_backbone_compute_training_metrics_returns_empty() -> None:
    model = _MinimalBackbone()
    out = model.forward({"x": torch.zeros(2, 4), "mask": torch.zeros(2, 3, dtype=torch.bool)})
    assert model.compute_training_metrics(out) == {}


def test_backbone_output_fields_round_trip() -> None:
    rep = torch.randn(3, 5)
    mask = torch.zeros(3, 7, dtype=torch.bool)
    out = BackboneOutput(representation=rep, padding_mask=mask)
    assert out.representation is rep
    assert out.padding_mask is mask
