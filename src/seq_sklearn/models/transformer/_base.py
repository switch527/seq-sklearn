"""Transformer-family estimator base (architecture A2 / A15.1, Phase 6b).

`TransformerSequenceEstimator` is a namespace carrying two mixin
classes. A concrete transformer model composes the matching mixin with
the Phase-6a family base, e.g.::

    class TFTClassifier(TransformerSequenceEstimator.Classifier,
                         BaseSequenceClassifier): ...

The mixin adds ``predict_with_attention`` (BETA, A15.1): one backbone
forward pass (via the Phase-6a ``_forward_backbone`` seam) yields both
the prediction surface AND the transformer introspection tensors, so
the two never disagree across separate passes. It composes WITHOUT
overriding any base ``predict`` / ``predict_proba`` / ``predict_quantiles``
contract. The default return is CPU ``np.ndarray``; a ``device``
argument flips every field to an on-device ``Tensor`` (A20 item 2).
"""

from typing import cast

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from seq_sklearn.inference.attention import AttentionOutput, RegressionAttentionOutput
from seq_sklearn.models._classifier import BaseSequenceClassifier
from seq_sklearn.models._regressor import BaseSequenceRegressor
from seq_sklearn.models.transformer._backbone import TransformerBackboneOutput

__all__ = ["TransformerSequenceEstimator"]


def _emit(t: Tensor, device: torch.device | str | None) -> np.ndarray:
    """CPU ``np.ndarray`` (``device=None``) or detached on-device ``Tensor``.

    Return type is annotated ``np.ndarray`` because that is the default
    and the A15.1 dataclass field type; the ``device`` path returns a
    ``Tensor`` at runtime per the documented BETA contract (cast so the
    dataclass construction stays pyright-clean).
    """
    if device is None:
        return t.detach().cpu().numpy()
    return cast("np.ndarray", t.detach().to(device))


def _emit_arr(a: np.ndarray, device: torch.device | str | None) -> np.ndarray:
    """Same as :func:`_emit` for a value already materialised as ndarray."""
    if device is None:
        return a
    return cast("np.ndarray", torch.as_tensor(a).to(device))


def _reorder_np(a: np.ndarray, order: np.ndarray) -> np.ndarray:
    return a[order]


def _reorder_t(t: Tensor, order: np.ndarray) -> Tensor:
    """Index an emission-order tensor back to caller X row order (F1).

    The index tensor is built on ``t``'s device so a GPU-resident
    introspection tensor reorders without a device-mismatch.
    """
    return t[torch.as_tensor(order, device=t.device)]


class TransformerSequenceEstimator:
    """Namespace for the transformer-family ``Classifier`` / ``Regressor`` mixins."""

    class Classifier:
        """Mixin: composes with :class:`BaseSequenceClassifier` (A2)."""

        def predict_with_attention(
            self,
            X: pd.DataFrame,  # noqa: N803
            *,
            device: torch.device | str | None = None,
        ) -> AttentionOutput:
            """Predictions + interpretable attention surfaces (A15.1, BETA).

            Raises:
                NotFittedError: ``fit`` has not been called (via the
                    shared ``_forward_backbone`` fitted check, F8).
            """
            est = cast("BaseSequenceClassifier", self)
            output, head, batch, below = est._forward_backbone(X)
            tout = cast("TransformerBackboneOutput", output)
            with torch.no_grad():
                # .detach().cpu() before _proba_from_raw: it calls
                # .numpy() internally, which raises on a CUDA tensor
                # (GPU-trained model). Mirrors the base _predict_raw
                # contract and the Regressor mixin below.
                logits_t = head(output.representation).detach().cpu()
            proba = est._proba_from_raw(logits_t)
            proba[below] = np.nan
            idx = est._index_from_proba(proba)
            logits = logits_t.numpy()  # already detached + on CPU
            logits[below] = np.nan
            # Every field above is in transform (sorted) row order with
            # the below-floor NaN-fill already applied in that space.
            # Restore caller X row order ONCE here, per field, AFTER the
            # fill (F1; predictions/proba/logits come from a local
            # forward, not the base _predict_raw seam, so this is the
            # single reorder for the whole dataclass).
            iro = batch["input_row_order"].cpu().numpy()
            return AttentionOutput(
                predictions=_emit_arr(_reorder_np(idx, iro), device),
                probabilities=_emit_arr(_reorder_np(proba, iro), device),
                logits=_emit_arr(_reorder_np(logits, iro), device),
                var_selection_weights=_emit(_reorder_t(tout.var_selection_weights, iro), device),
                static_var_selection_weights=_emit(
                    _reorder_t(tout.static_var_selection_weights, iro), device
                ),
                attention_weights=_emit(_reorder_t(tout.attention_weights, iro), device),
                padding_mask=_emit(_reorder_t(output.padding_mask, iro), device),
                entity_id=_emit(_reorder_t(batch["entity_id"], iro), device),
            )

    class Regressor:
        """Mixin: composes with :class:`BaseSequenceRegressor` (A2)."""

        def predict_with_attention(
            self,
            X: pd.DataFrame,  # noqa: N803
            *,
            device: torch.device | str | None = None,
        ) -> RegressionAttentionOutput:
            """Predictions + interpretable attention surfaces (A15.1, BETA).

            ``predictions`` is the calibrated ``(N,)`` point estimate, or
            the calibrated ``(N, len(quantiles))`` matrix in quantile
            mode (NaN-filled for below-floor entities, matching
            ``predict`` / ``predict_quantiles``). No ``logits`` field by
            design (qa r2-I3).

            Raises:
                NotFittedError: ``fit`` has not been called (F8).
            """
            est = cast("BaseSequenceRegressor", self)
            output, head, batch, below = est._forward_backbone(X)
            tout = cast("TransformerBackboneOutput", output)
            with torch.no_grad():
                raw = head(output.representation).detach().cpu()
            mat = est._calibrate_raw(raw, below)  # (N, Q), below-rows NaN
            if est._is_quantile():
                predictions = mat
                quantiles_used: tuple[float, ...] | None = tuple(
                    np.asarray(est.quantiles_, dtype=np.float64).tolist()
                )
            else:
                predictions = mat.reshape(-1)
                quantiles_used = None
            # predictions is in transform (sorted) order, NaN-filled in
            # that space by _calibrate_raw. Restore caller X row order
            # ONCE per per-row field here, AFTER the fill (F1).
            # quantiles_used is fit-time metadata, NOT per-row: it is
            # shuffle-invariant and left untouched.
            iro = batch["input_row_order"].cpu().numpy()
            return RegressionAttentionOutput(
                predictions=_emit_arr(_reorder_np(predictions, iro), device),
                quantiles_used=quantiles_used,
                var_selection_weights=_emit(_reorder_t(tout.var_selection_weights, iro), device),
                static_var_selection_weights=_emit(
                    _reorder_t(tout.static_var_selection_weights, iro), device
                ),
                attention_weights=_emit(_reorder_t(tout.attention_weights, iro), device),
                padding_mask=_emit(_reorder_t(output.padding_mask, iro), device),
                entity_id=_emit(_reorder_t(batch["entity_id"], iro), device),
            )
