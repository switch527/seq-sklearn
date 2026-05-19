"""`_OnnxForward` export-clone isolation (Phase 10, S8 Gemini CRITICAL).

`_OnnxForward.__init__` MUST deepcopy the backbone/head and set
``onnx_export`` only on the private clone. Mutating the shared
``_predict_module()`` backbone would race a concurrent ``predict()``
(the sklearn read-only post-fit thread-safety contract; Phase 9 G-C1
class). These tests are mutation-sensitive by construction: reverting
``copy.deepcopy(backbone)`` to a bare ``backbone`` alias in
``_OnnxForward.__init__`` flips every assertion below to red. The
heavyweight parity suite does NOT catch that regression (its
post-export ``_predict_raw`` oracle itself mutates under the alias, so
the atol=1e-6 compare still passes), which is exactly why this lives
here as a standalone pin.
"""

from torch import Tensor, nn

from seq_sklearn.inference.onnx import _OnnxForward


class _BackboneStub(nn.Module):
    """Minimal stand-in for ``TFTBackbone``: an ``onnx_export`` flag
    (default False, as the real backbone sets it) and one parameter so
    ``deepcopy`` and ``eval()`` exercise the real nn.Module paths."""

    def __init__(self) -> None:
        super().__init__()
        self.onnx_export = False
        self.lin = nn.Linear(2, 2)

    def forward(self, batch: dict[str, Tensor]) -> object:
        raise AssertionError("forward not exercised by the isolation pin")


class _HeadStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 1)

    def forward(self, rep: Tensor) -> Tensor:
        return self.lin(rep)


def test_onnxforward_does_not_mutate_shared_backbone() -> None:
    backbone = _BackboneStub()
    head = _HeadStub()
    assert backbone.onnx_export is False

    wrapper = _OnnxForward(backbone, head)

    # The shared instances the fitted estimator predicts with are
    # untouched: flag still False, identity unchanged. Reverting the
    # deepcopy to an alias makes the flag True here -> fails.
    assert backbone.onnx_export is False
    assert wrapper.backbone is not backbone
    assert wrapper.head is not head
    # The private clone is permanently on the pack-free export path.
    assert wrapper.backbone.onnx_export is True


def test_onnxforward_clone_params_are_distinct_objects() -> None:
    """Deepcopy, not a shallow alias: every clone Parameter is a
    distinct object from the shared backbone's. An alias revert makes
    them identical -> fails."""
    backbone = _BackboneStub()
    wrapper = _OnnxForward(backbone, _HeadStub())
    shared = list(backbone.parameters())
    clone = list(wrapper.backbone.parameters())
    assert shared
    assert len(shared) == len(clone)
    assert all(s is not c for s, c in zip(shared, clone, strict=True))
