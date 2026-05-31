"""TFTBackbone: the encoder-only Temporal Fusion Transformer (per A6 / A15).

Composes the A6 block pipeline in the order corrected against
the canonical reference (Lim et al., arXiv:1912.09363, and the
Google Research TF1 ``tft_model.py``): static encoders ->
static VSN -> four context
GRNs (c_s, c_h, c_c, c_e) -> past VSN gated by c_s -> LSTM initialized
``lstm(x, (c_h, c_c))`` over packed variable-length sequences ->
post-LSTM GLU + AddNorm against the pre-LSTM VSN output -> c_e
enrichment GRN -> interpretable shared-V self-attention -> post-attention
GLU + AddNorm -> position-wise FFN GRN + GLU + AddNorm -> readout.

The constructor takes the per-column categorical cardinalities and the
real-feature counts. These are exactly the fitted
:class:`~seq_sklearn.data.tabular_to_sequence.TabularToSequence`
attributes (``cardinalities_`` plus the config column counts); the A6 /
F4 spec does not pin the backbone constructor signature, so this surface
is chosen to consume the Phase 2 transformer's fitted state directly.
The embedding table width is ``cardinality + 1`` to match the encoder's
per-column ``<unk>`` slot at index 0.
"""

import logging
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from seq_sklearn.config.tft import TFTConfig
from seq_sklearn.errors import PredictionError
from seq_sklearn.models._backbone import BackboneOutput, BaseBackbone
from seq_sklearn.models._layers import make_embedding, make_linear
from seq_sklearn.models.transformer._backbone import TransformerBackboneOutput
from seq_sklearn.models.transformer._interpretable_attention import (
    InterpretableMultiHeadAttention,
)
from seq_sklearn.models.transformer.tft.blocks import GLU, GRN, VSN, AddNorm

__all__ = ["TFTBackbone"]

logger = logging.getLogger(__name__)


class TFTBackbone(BaseBackbone):
    """Encoder-only TFT producing a pooled representation plus introspection.

    ``static_cat_cardinalities`` / ``tv_cat_cardinalities`` are the raw
    per-column cardinalities (the embedding table adds one row for the
    ``<unk>`` slot). ``n_static_real`` / ``n_tv_real`` are the real-feature
    counts. The backbone needs at least one static and one temporal
    variable so the VSN has something to select; a synthetic
    all-zeros variable is added when a side has no real features and no
    categoricals so the VSN dimensionality is always positive.
    """

    def __init__(
        self,
        config: TFTConfig,
        *,
        static_cat_cardinalities: list[int],
        tv_cat_cardinalities: list[int],
        n_static_real: int,
        n_tv_real: int,
    ) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.hidden_size = hidden
        self.prediction_readout = config.prediction_readout
        # ONNX-export-only: when True, _run_lstm uses the pack-free
        # gather-preserving path (no pack_padded_sequence, which does
        # not lower under torch.onnx.export(dynamo=True)). Default
        # False so the trained eager forward is untouched; _OnnxForward
        # sets it True only on its private deepcopy of this backbone
        # for the export trace, never on the shared instance the
        # fitted estimator predicts with (Phase 10 plan Step 1/2).
        self.onnx_export = False

        self.static_cat_embeddings = nn.ModuleList(
            make_embedding(card + 1, hidden) for card in static_cat_cardinalities
        )
        self.static_real_proj = nn.ModuleList(make_linear(1, hidden) for _ in range(n_static_real))
        self.tv_cat_embeddings = nn.ModuleList(
            make_embedding(card + 1, hidden) for card in tv_cat_cardinalities
        )
        self.tv_real_proj = nn.ModuleList(make_linear(1, hidden) for _ in range(n_tv_real))

        n_static_vars = len(static_cat_cardinalities) + n_static_real
        n_tv_vars = len(tv_cat_cardinalities) + n_tv_real
        # The VSN must select over at least one variable; an empty side
        # contributes a single learned constant variable.
        self._static_pad = n_static_vars == 0
        self._tv_pad = n_tv_vars == 0
        self.n_static_vars = max(n_static_vars, 1)
        self.n_tv_vars = max(n_tv_vars, 1)
        if self._static_pad:
            self.static_pad_vec = nn.Parameter(torch.zeros(hidden))
        if self._tv_pad:
            self.tv_pad_vec = nn.Parameter(torch.zeros(hidden))

        self.static_vsn = _StaticVSN(self.n_static_vars, config)
        self.context_s = GRN(hidden, hidden, config.dropout)
        self.context_h = GRN(hidden, hidden, config.dropout)
        self.context_c = GRN(hidden, hidden, config.dropout)
        self.context_e = GRN(hidden, hidden, config.dropout)

        self.temporal_vsn = VSN(self.n_tv_vars, config)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.post_lstm_glu = GLU(hidden, hidden)
        self.post_lstm_norm = AddNorm(hidden)
        self.enrichment = GRN(hidden, hidden, config.dropout, context_size=hidden)
        self.attention = InterpretableMultiHeadAttention(hidden, config.attention_heads)
        self.post_attn_glu = GLU(hidden, hidden)
        self.post_attn_norm = AddNorm(hidden)
        self.ffn = GRN(hidden, hidden, config.dropout)
        self.post_ffn_glu = GLU(hidden, hidden)
        self.post_ffn_norm = AddNorm(hidden)

    def _encode_static(self, batch: dict[str, Tensor]) -> Tensor:
        static_cat = batch["static_categorical"]  # (B, n_static_cat)
        static_real = batch["static_real"]  # (B, n_static_real)
        bsz = static_cat.shape[0]
        parts: list[Tensor] = []
        for i, emb in enumerate(self.static_cat_embeddings):
            parts.append(emb(static_cat[:, i]))  # (B, hidden)
        for i, proj in enumerate(self.static_real_proj):
            parts.append(proj(static_real[:, i : i + 1]))  # (B, hidden)
        if self._static_pad:
            parts.append(self.static_pad_vec.expand(bsz, -1))  # (B, hidden)
        return torch.stack(parts, dim=1)  # (B, n_static_vars, hidden)

    def _encode_temporal(self, batch: dict[str, Tensor]) -> Tensor:
        tv_real = batch["time_varying_real"]  # (B, L, n_tv_real)
        tv_cat = batch["time_varying_categorical"]  # (B, L, n_tv_cat)
        bsz, seq_len = batch["padding_mask"].shape
        parts: list[Tensor] = []
        for i, emb in enumerate(self.tv_cat_embeddings):
            parts.append(emb(tv_cat[:, :, i]))  # (B, L, hidden)
        for i, proj in enumerate(self.tv_real_proj):
            parts.append(proj(tv_real[:, :, i : i + 1]))  # (B, L, hidden)
        if self._tv_pad:
            parts.append(self.tv_pad_vec.expand(bsz, seq_len, -1))  # (B, L, hidden)
        return torch.stack(parts, dim=2)  # (B, L, n_tv_vars, hidden)

    def _run_lstm(self, vsn_out: Tensor, c_h: Tensor, c_c: Tensor, mask: Tensor) -> Tensor:
        # The data contract left-pads (padding is a leading block, valid
        # a contiguous suffix). pack_padded_sequence takes the LEADING
        # `length` steps, so each row is gathered valid-first before
        # packing and scattered back to its original positions after.
        seq_len = vsn_out.shape[1]
        valid = ~mask  # (B, L); True at valid positions
        lengths = valid.sum(dim=1)  # (B,); valid timesteps per element
        # F3 guarantees padding is a contiguous LEADING block. Interior
        # or leading-valid padding would make the argsort gather
        # silently reorder timesteps, so reject it rather than corrupt
        # the LSTM with a wrong sequence order.
        if not self.onnx_export:
            # Data-dependent eager guard; skipped on the export path
            # (the graph trusts its inputs, and `bool(...)` here would
            # raise GuardOnDataDependentSymNode under torch.export).
            positions = torch.arange(seq_len, device=mask.device)
            expected = positions[None, :] >= (seq_len - lengths)[:, None]  # (B, L)
            if not bool((valid == expected).all()):
                raise PredictionError(
                    "padding_mask is not a contiguous leading block; the F3 "
                    "left-pad contract is required for the packed LSTM"
                )
        if self.onnx_export:
            # `aten.sort.stable` (torch.argsort(stable=True)) has no
            # ONNX lowering. For the F3-guaranteed left-padded
            # contiguous block the stable valid-first permutation is
            # exactly a per-row modular ROLL by (seq_len - lengths):
            # new index j maps to old index (j + roll) % seq_len, which
            # places the trailing valid suffix first (in order) then
            # the leading pad block (in order). Identical to the
            # stable-argsort result on left-padded input; only Range/
            # Add/Sub/Mod ops, all ONNX-safe.
            roll = seq_len - lengths  # (B,); lengths is int64 (bool sum)
            pos = torch.arange(seq_len, device=mask.device)  # (L,)
            order = (pos[None, :] + roll[:, None]) % seq_len  # (B, L)
            inverse = (pos[None, :] - roll[:, None]) % seq_len  # (B, L)
        else:
            order = torch.argsort(  # (B, L); valid first, original order kept
                (~valid).int(), dim=1, stable=True
            )
            inverse = torch.argsort(order, dim=1)  # maps back to original
        gathered = torch.gather(
            vsn_out, 1, order[:, :, None].expand(-1, -1, vsn_out.shape[2])
        )  # (B, L, hidden); valid steps left-justified
        # nn.LSTM init state is (h_0, c_0); A6 names these (c_h, c_c).
        h_0 = c_h.unsqueeze(0)  # (1, B, hidden)
        c_0 = c_c.unsqueeze(0)  # (1, B, hidden)
        if self.onnx_export:
            # ONNX-export-only (Phase 10): pack_padded_sequence does not
            # lower under torch.onnx.export(dynamo=True). Run the LSTM
            # directly over the valid-first `gathered` sequence. This is
            # numerically identical to the packed path at every VALID
            # position: the LSTM is causal, the valid steps are a
            # contiguous prefix of `gathered`, so trailing-pad steps
            # cannot affect the recurrent state of earlier valid steps
            # (pack_padded_sequence merely stops the recurrence early).
            # The trailing-pad outputs land, via the `inverse` scatter,
            # at masked positions both readouts ignore.
            lstm_left, _ = self.lstm(gathered, (h_0, c_0))  # (B, L, hidden)
        else:
            packed = pack_padded_sequence(
                gathered, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed, (h_0, c_0))
            lstm_left, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=seq_len
            )  # (B, L, hidden); valid-first layout
        return torch.gather(
            lstm_left, 1, inverse[:, :, None].expand(-1, -1, lstm_left.shape[2])
        )  # (B, L, hidden)

    def _readout(self, hidden_seq: Tensor, mask: Tensor) -> Tensor:
        valid = ~mask  # (B, L); True at valid positions
        if self.prediction_readout == "mean_pool":
            valid_f = valid.to(hidden_seq.dtype)  # (B, L)
            summed = (hidden_seq * valid_f[:, :, None]).sum(dim=1)  # (B, hidden)
            return summed / valid_f.sum(dim=1, keepdim=True)
        # last_valid: index of the last True in (~mask) per batch element.
        last_idx = (valid.shape[1] - 1) - valid.flip(dims=[1]).int().argmax(dim=1)  # (B,)
        return hidden_seq[torch.arange(hidden_seq.shape[0]), last_idx]  # (B, hidden)

    def forward(self, batch: dict[str, Tensor]) -> TransformerBackboneOutput:
        """Run the A6 pipeline and return a TransformerBackboneOutput.

        Raises:
            PredictionError: A real-valued input contains NaN or inf, a
                categorical index is outside its embedding table (a
                direct caller bypassing ``TabularToSequence``), or a
                window has zero valid timesteps after preprocessing (the
                A6 ``mask.any(dim=1).all()`` readout precondition,
                checked up front since the packed LSTM rejects a
                zero-length sequence).
        """
        # These three blocks are eager input-validation guards. ONNX
        # has no exception mechanism and a serialized inference graph
        # trusts its inputs (validation stays the responsibility of the
        # eager `predict` path and the serving layer), so they are
        # SKIPPED on the export path. They are also data-dependent
        # (`feat.numel()` is symbolic under a dynamic batch dim,
        # `.all()`/`nonzero` are unbacked), so leaving them in would
        # raise GuardOnDataDependentSymNode under torch.export even
        # with strict=False; skipping them is the explicit realization
        # of the Phase 10 "graph trusts its inputs" invariant.
        if not self.onnx_export:
            for name in ("time_varying_real", "static_real"):
                feat = batch[name]
                if feat.numel() and not bool(torch.isfinite(feat).all()):
                    raise PredictionError(
                        f"{name} contains NaN or inf; clean or impute before predict"
                    )
            for name, embeddings in (
                ("static_categorical", self.static_cat_embeddings),
                ("time_varying_categorical", self.tv_cat_embeddings),
            ):
                codes = batch[name]
                # Range only, not column order: ordering correctness
                # between the cardinality list and the encoder columns
                # is the Phase 4 estimator's responsibility (it owns
                # the fitted TabularToSequence). This guard just
                # prevents an opaque embedding-kernel abort for a
                # direct caller.
                for i, module in enumerate(embeddings):
                    emb = cast(nn.Embedding, module)
                    col = codes[..., i]
                    if not bool(((col >= 0) & (col < emb.num_embeddings)).all()):
                        raise PredictionError(
                            f"{name} column {i} has an index outside "
                            f"[0, {emb.num_embeddings}); fit and transform via "
                            "TabularToSequence before predict"
                        )
        mask = batch["padding_mask"]  # (B, L); True = padding
        if not self.onnx_export:
            all_pad = (~mask).any(dim=1).logical_not()  # (B,); True where row all padding
            if bool(all_pad.any()):
                bad_rows = [int(r) for r in torch.nonzero(all_pad, as_tuple=False).flatten()]
                raise PredictionError(
                    f"window had zero valid timesteps after preprocessing (rows {bad_rows})"
                )

        static_vars = self._encode_static(batch)  # (B, n_static_vars, hidden)
        static_selected, static_weights = self.static_vsn(static_vars)
        c_s = self.context_s(static_selected)  # (B, hidden)
        c_h = self.context_h(static_selected)  # (B, hidden)
        c_c = self.context_c(static_selected)  # (B, hidden)
        c_e = self.context_e(static_selected)  # (B, hidden)

        temporal_vars = self._encode_temporal(batch)  # (B, L, n_tv_vars, hidden)
        vsn_out, var_weights = self.temporal_vsn(temporal_vars, c_s, mask)

        lstm_out = self._run_lstm(vsn_out, c_h, c_c, mask)  # (B, L, hidden)
        gated = self.post_lstm_glu(lstm_out)  # (B, L, hidden)
        post_lstm = self.post_lstm_norm(gated, vsn_out)  # (B, L, hidden)

        c_e_seq = c_e[:, None, :].expand(-1, post_lstm.shape[1], -1)  # (B, L, hidden)
        attn_in = self.enrichment(post_lstm, c_e_seq)  # (B, L, hidden)

        attn_out, attn_weights = self.attention.forward_interpretable(attn_in, mask)
        post_attn = self.post_attn_norm(self.post_attn_glu(attn_out), attn_in)

        ffn_out = self.ffn(post_attn)  # (B, L, hidden)
        encoded = self.post_ffn_norm(self.post_ffn_glu(ffn_out), post_attn)

        representation = self._readout(encoded, mask)  # (B, hidden)
        return TransformerBackboneOutput(
            representation=representation,
            padding_mask=mask,
            var_selection_weights=var_weights,
            attention_weights=attn_weights,
            static_var_selection_weights=static_weights,
        )

    def compute_training_metrics(self, output: BackboneOutput) -> dict[str, object]:
        """Reduce the introspection tensors to F11 entropy payloads (A15).

        The ``padding_mask`` is applied to the temporal and attention
        reductions so padded timesteps do not bias the metrics. A padded
        query still has valid keys and so produces a normal attention
        distribution, not a zero row; it is excluded explicitly via the
        mask, not assumed zero. The static branch has no time axis and
        skips the mask.

        The parameter is typed as the base ``BackboneOutput`` so the
        override stays Liskov-compatible with ``BaseBackbone``; this
        backbone only ever produces a ``TransformerBackboneOutput``, so
        a narrower input is a programming error.

        Raises:
            TypeError: ``output`` is not a ``TransformerBackboneOutput``.
        """
        if not isinstance(output, TransformerBackboneOutput):
            raise TypeError(
                "TFTBackbone.compute_training_metrics requires a "
                f"TransformerBackboneOutput, got {type(output).__name__}"
            )
        mask = output.padding_mask  # (B, L); True = padding
        valid = (~mask).float()  # (B, L); 1 at valid timesteps
        valid_count = valid.sum().clamp_min(1.0)  # scalar; >= 1 per the A6 mask invariant

        sw = output.static_var_selection_weights  # (B, n_static_vars)
        static_h = -(sw * sw.clamp_min(1e-12).log()).sum(dim=-1)  # (B,)
        static_entropy = static_h.mean().item()

        tw = output.var_selection_weights  # (B, L, n_vars)
        temporal_h = -(tw * tw.clamp_min(1e-12).log()).sum(dim=-1)  # (B, L)
        temporal_entropy = ((temporal_h * valid).sum() / valid_count).item()

        aw = output.attention_weights  # (B, H, L, L)
        attn_h = -(aw * aw.clamp_min(1e-12).log()).sum(dim=-1)  # (B, H, L)
        valid_h = valid.unsqueeze(1)  # (B, 1, L)
        per_head = (attn_h * valid_h).sum(dim=(0, 2)) / valid_count
        entropy_per_head = per_head.tolist()  # list[float], length H

        return {
            "train.var_selection_entropy": {
                "static_entropy": static_entropy,
                "temporal_entropy": temporal_entropy,
            },
            "train.attention_entropy": {
                "entropy_per_head": entropy_per_head,
            },
        }


class _StaticVSN(nn.Module):
    """Variable selection over the static inputs (no time axis).

    A static-only counterpart to the temporal :class:`VSN`: a GRN over
    the flattened static variables produces one softmax weight per
    variable; a per-variable GRN transforms each; the output is the
    weighted sum ``(B, hidden)`` plus the ``(B, n_static_vars)`` weights.
    """

    def __init__(self, n_vars: int, config: TFTConfig) -> None:
        super().__init__()
        self.n_vars = n_vars
        hidden = config.hidden_size
        # n_vars == 1 (an empty static side) makes the flatten GRN's
        # AddNorm a LayerNorm over a length-1 axis, so the logit is a
        # learned constant and softmax over one variable is exactly 1.0.
        # Degenerate but correct: a single-variable selection must weight
        # 1.0, and the entropy metric is a well-defined 0.
        self.flatten_grn = GRN(n_vars * hidden, n_vars, config.variable_selection_dropout)
        self.var_grns = nn.ModuleList(
            GRN(hidden, hidden, config.variable_selection_dropout) for _ in range(n_vars)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Select static variables from ``x`` of shape ``(B, n_vars, hidden)``."""
        bsz, n_vars, hidden = x.shape
        flat = x.reshape(bsz, n_vars * hidden)  # (B, n_vars * hidden)
        weights = torch.softmax(self.flatten_grn(flat), dim=-1)  # (B, n_vars)
        transformed = torch.stack(
            [self.var_grns[i](x[:, i, :]) for i in range(n_vars)], dim=1
        )  # (B, n_vars, hidden)
        selected = (weights[:, :, None] * transformed).sum(dim=1)  # (B, hidden)
        return selected, weights
