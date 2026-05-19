"""ONNX-traceable forward wrapper (Phase 10, architecture A21).

`torch.onnx.export` needs an ``nn.Module`` with positional tensor
inputs and a tensor output. The TFT backbone takes a
``dict[str, Tensor]`` and returns a dataclass; the head maps
``representation -> logits``. :class:`_OnnxForward` adapts that to the
five tensor inputs the backbone forward actually consumes and returns
the raw head logits ``(B, K)``.

The exported graph is the model FORWARD ONLY (backbone + head, raw
logits). Calibration, thresholding, below-floor NaN-fill, and
caller-order restore are sklearn-side numpy post-processing applied by
the family ``predict`` / ``predict_proba`` and are deliberately NOT in
the ONNX graph (the same torch/numpy seam ``_predict_raw`` draws).

The gather-preserving pack-free LSTM path is selected by the
backbone's ``onnx_export`` flag, which
:meth:`BaseSequenceEstimator.export_onnx` toggles transiently around
the export and restores in a ``finally`` (the trained eager forward
is untouched).
"""

import torch
from torch import Tensor, nn

__all__ = ["_OnnxForward"]


class _OnnxForward(nn.Module):
    """Backbone + head as a positional-tensor-in, logits-out module.

    The five inputs are exactly the keys ``TFTBackbone.forward``
    reads off the batch dict; ``entity_id`` is not consumed on the
    forward path and ``target`` / ``input_row_order`` are post-hoc, so
    they are excluded from the traced signature.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.backbone.eval()
        self.head.eval()

    def forward(
        self,
        static_categorical: Tensor,
        static_real: Tensor,
        time_varying_real: Tensor,
        time_varying_categorical: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        batch = {
            "static_categorical": static_categorical,
            "static_real": static_real,
            "time_varying_real": time_varying_real,
            "time_varying_categorical": time_varying_categorical,
            "padding_mask": padding_mask,
        }
        # ALWAYS use the gather-preserving pack-free LSTM path (Phase
        # 10 plan Step 1 GATE CONDITION): the wrapper never depends on
        # whether pack_padded_sequence lowers, so the parity test's
        # `_OnnxForward == _predict_raw[0]` (atol=1e-6) is, on every
        # run, the objective proof that pack-free equals the packed
        # production path. The flag is restored so the shared backbone
        # the estimator predicts with is untouched after the call.
        prev = getattr(self.backbone, "onnx_export", False)
        setattr(self.backbone, "onnx_export", True)  # noqa: B010
        try:
            with torch.no_grad():
                representation = self.backbone(batch).representation
                return self.head(representation)
        finally:
            setattr(self.backbone, "onnx_export", prev)  # noqa: B010
