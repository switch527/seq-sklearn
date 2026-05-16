"""The :class:`_Calibrator` structural contract (per architecture A9).

A9 pins the four-method protocol every concrete calibrator implements.
The contract is internal: estimators hold a ``_Calibrator`` and call it
through these four methods only, so a v3 calibrator family is added by
implementing the protocol without touching estimator code.

Two semantics are load-bearing and pinned by the Phase 5 deliverable
tests:

* **What ``transform`` consumes and returns.** ``fit`` / ``transform``
  take the model's RAW outputs (the protocol names the argument
  ``logits`` after the classifier case, but a regression calibrator
  receives the predicted-quantile matrix). ``transform`` returns
  CALIBRATED PROBABILITIES for classification calibrators and
  CALIBRATED QUANTILE VALUES for regression calibrators, never logits.
  The estimator's ``predict_proba`` / ``predict_quantiles`` consumes
  the return value directly, so the activation (softmax / sigmoid) is
  the calibrator's responsibility, not the caller's. A "no-op" path is
  the estimator selecting ``calibration_strategy="none"`` and never
  constructing a calibrator at all; there is no identity calibrator.
* **``serialize`` is JSON-only.** Per requirements F4 a calibrator's
  state lives in ``state.json``, never in the safetensors weight
  archive. ``serialize`` returns a dict of JSON primitives (``float``,
  ``int``, ``str``, ``bool``, ``None``, and ``list`` / ``dict`` of the
  same); ``deserialize`` is its exact inverse. The round-trip test
  asserts equality AFTER ``json.dumps`` / ``json.loads`` so any
  float-precision loss in JSON encoding surfaces.
"""

from typing import Protocol, runtime_checkable

from torch import Tensor

__all__ = ["_Calibrator"]


@runtime_checkable
class _Calibrator(Protocol):
    """Structural type for every concrete calibrator (A9).

    ``runtime_checkable`` so the estimator-base save / load path can
    ``isinstance``-guard a deserialized object in Phase 6 without
    importing every concrete class.
    """

    def fit(self, logits: Tensor, y_true: Tensor) -> None:
        """Fit calibrator state on the held-out calibration fold.

        ``logits`` is the model's raw output on the calibration fold
        (class logits for classification, the predicted-quantile matrix
        for regression); ``y_true`` is the aligned ground truth. Fitting
        mutates the instance in place and returns nothing.
        """
        ...

    def transform(self, logits: Tensor) -> Tensor:
        """Map raw model output to calibrated probabilities / quantiles.

        Returns calibrated probabilities for classification calibrators
        and calibrated quantile values for regression calibrators (see
        the module docstring); the activation is applied here.
        """
        ...

    def serialize(self) -> dict[str, object]:
        """Return fitted state as a JSON-primitive dict (F4)."""
        ...

    @classmethod
    def deserialize(cls, blob: dict[str, object]) -> "_Calibrator":
        """Reconstruct a fitted calibrator from :meth:`serialize` output."""
        ...
