# TFT research brief (v1)

Decision-oriented summary of the Temporal Fusion Transformer (Lim et al.,
2021) and its open-source ecosystem, scoped to inform the seq-sklearn v1
classification + regression adaptation.

## Source citations

- [arXiv:1912.09363 Lim et al., 2021](https://arxiv.org/abs/1912.09363) original TFT paper, multi-horizon forecasting with quantile output.
- [Google Research TF1 reference implementation `tft_model.py`](https://github.com/google-research/google-research/blob/master/tft/libs/tft_model.py) the canonical reference used by the paper authors.
- [pytorch-forecasting `_tft.py` v1.3.0 module source](https://pytorch-forecasting.readthedocs.io/en/v1.3.0/_modules/pytorch_forecasting/models/temporal_fusion_transformer/_tft.html) the canonical PyTorch TFT under active maintenance by sktime.
- [pytorch-forecasting `TemporalFusionTransformer` API docs](https://pytorch-forecasting.readthedocs.io/en/v1.0.0/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html) hyperparameters and `interpret_output()` surface.
- [PlaytikaOSS `tft-torch` GitHub repo](https://github.com/PlaytikaOSS/tft-torch) third-party PyTorch port, useful as a second opinion on naming and tensor shapes.
- [Yash Gupta, "TFT for time series classification" (Medium, 2024)](https://medium.com/@eryash15/temporal-fusion-transformer-for-time-series-classification-a-complete-walkthrough-5c455f488047) walkthrough of pytorch-forecasting set to `CrossEntropy()` for classification.
- [sktime/pytorch-forecasting issue #1792, "TFT for binary classification" (Mar 2025)](https://github.com/sktime/pytorch-forecasting/issues/1792) user question, no maintainer answer at time of writing.
- [PyTorch issue #135615, `scaled_dot_product_attention` ONNX export failure](https://github.com/pytorch/pytorch/issues/135615) explicit `scale` argument breaks export.
- [PyTorch issue #149662, ONNX `Attention` op for SDPA (opset 23)](https://github.com/pytorch/pytorch/issues/149662) future direction; currently SDPA decomposes into primitive ops.
- [Quantum TFT, arXiv:2508.04048 (2025)](https://arxiv.org/abs/2508.04048) the only "TFT v2"-like proposal found in 2024-2025; quantum hybrid, not applicable here.
- [`torch.nn.functional.scaled_dot_product_attention` API docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) backend selection and signature.

## Verified block flow

The four static context vectors are produced by four separate GRNs over
the static encoder output. In pytorch-forecasting they are named exactly
[pytorch-forecasting `_tft.py`](https://pytorch-forecasting.readthedocs.io/en/v1.3.0/_modules/pytorch_forecasting/models/temporal_fusion_transformer/_tft.html):

- `static_context_variable_selection` (c_s) gates the temporal VSN.
- `static_context_initial_hidden_lstm` (c_h) initializes LSTM hidden.
- `static_context_initial_cell_lstm` (c_c) initializes LSTM cell.
- `static_context_enrichment` (c_e) feeds the post-LSTM enrichment GRN.

**LSTM init resolution.** The architecture draft is correct on which two
vectors initialize the LSTM, but the assignment is the opposite of what
the draft's `(c_c, c_h)` label suggests if read positionally. Both
reference implementations are explicit:

- pytorch-forecasting passes `(input_hidden, input_cell)` to its LSTM
  wrapper, where `input_hidden = static_context_initial_hidden_lstm(...)`
  and `input_cell = static_context_initial_cell_lstm(...)`
  [pytorch-forecasting `_tft.py`](https://pytorch-forecasting.readthedocs.io/en/v1.3.0/_modules/pytorch_forecasting/models/temporal_fusion_transformer/_tft.html).
- The Google Research TF1 source passes
  `initial_state=[static_context_state_h, static_context_state_c]` to
  its Keras LSTM, in the same order
  [google-research `tft_model.py`](https://github.com/google-research/google-research/blob/master/tft/libs/tft_model.py).
- PlaytikaOSS `tft-torch` calls
  `self.past_lstm(selected_historical, (c_seq_hidden, c_seq_cell))`
  [PlaytikaOSS `tft-torch`](https://github.com/PlaytikaOSS/tft-torch).

PyTorch's `nn.LSTM` expects `(h_0, c_0)` in that order. So the contract
is `h_0 = c_h`, `c_0 = c_c`. The architecture draft phrasing "c_c
initializes the LSTM cell state; c_h initializes the LSTM hidden state"
is correct by name; any `(c_c, c_h)` tuple labels in the doc should be
read as a set, not as the call-order pair. seq-sklearn should follow
PyTorch's call order: `lstm(x, (c_h, c_c))`.

**Static enrichment placement.** `c_e` is broadcast across timesteps and
combined with the LSTM output through a GRN, BEFORE self-attention, not
after. pytorch-forecasting:
`attn_input = self.static_enrichment(lstm_output, self.expand_static_context(c_e, timesteps))`,
applied after `post_lstm_add_norm`
[pytorch-forecasting `_tft.py`](https://pytorch-forecasting.readthedocs.io/en/v1.3.0/_modules/pytorch_forecasting/models/temporal_fusion_transformer/_tft.html).
The current architecture draft says c_e is "added before the post-LSTM
GRN + add-norm block" which is backwards: the post-LSTM add-norm comes
FIRST, then the enrichment GRN consumes its output, then self-attention.
This is a correctness item the architecture phase needs to fix.

**Block flow with shapes** (B = batch, L = lookback, H = hidden_size,
S = static feature count):

```
static raw -> per-feature embed/scale -> (B, S, H)
            -> static VSN over S features -> static_embedding (B, H)
            -> 4 GRNs -> c_s, c_e, c_h, c_c each (B, H)

temporal raw (B, L, T) -> per-feature embed/scale -> (B, L, T, H)
            -> temporal VSN gated by c_s -> (B, L, H)
LSTM (1 layer, H -> H) with (h_0=c_h, c_0=c_c) -> (B, L, H)
post-LSTM gate + add-norm with VSN residual -> (B, L, H)
static enrichment GRN(., expand(c_e, L)) -> (B, L, H)
interpretable multi-head self-attention (masked) -> (B, L, H)
post-attn gate + add-norm -> (B, L, H)
position-wise FFN GRN -> (B, L, H)
final gate + add-norm with enrichment residual -> (B, L, H)
readout (v1: last_valid or mean_pool) -> (B, H)
head -> logits or regression scalars
```

This matches the Google TF1 reference and the pytorch-forecasting v1.3
source. The architecture draft's A6 diagram has the right blocks but
misorders the static enrichment relative to the post-LSTM add-norm; see
the "Implementation gotchas" section.

## Classification / regression adaptation findings

The original TFT ships only a quantile head over the decoder horizon
[arXiv:1912.09363](https://arxiv.org/abs/1912.09363). No published
"TFT classifier" exists as a named architecture.

The de facto adaptation in the community is to swap the quantile loss
for `CrossEntropy` and set `output_size = num_classes` on
pytorch-forecasting's `TemporalFusionTransformer`
[Yash Gupta walkthrough](https://medium.com/@eryash15/temporal-fusion-transformer-for-time-series-classification-a-complete-walkthrough-5c455f488047),
[issue #1792](https://github.com/sktime/pytorch-forecasting/issues/1792).
This keeps the decoder and predicts one class distribution per future
horizon step, which is not what seq-sklearn wants: panel
classification has one label per (entity, period), not one per
forecasted horizon. No open-source classifier-only or regressor-only
TFT variant surfaced in the search.

**Implications for seq-sklearn v1.**

- The "drop the decoder" choice is correct and has no reference impl;
  we are building the first one and should validate block re-ordering
  carefully.
- `last_valid` vs `mean_pool` has no prior art on TFT specifically;
  both are standard sequence-encoder readouts. `last_valid` is the
  natural default for causal panel data ("as of period t").
- Raw-logits head matches pytorch-forecasting's `mode="raw"` semantics
  [pytorch-forecasting API](https://pytorch-forecasting.readthedocs.io/en/v1.0.0/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html).

## Mask handling

pytorch-forecasting uses BOTH `pack_padded_sequence` AND attention
masks
[pytorch-forecasting `_tft.py`](https://pytorch-forecasting.readthedocs.io/en/v1.3.0/_modules/pytorch_forecasting/models/temporal_fusion_transformer/_tft.html):

- Its LSTM wrapper calls `pack_padded_sequence(..., enforce_sorted=False)`
  via the `lengths=encoder_lengths` argument on
  `self.lstm_encoder(embeddings_varying_encoder, (input_hidden, input_cell), lengths=encoder_lengths, enforce_sorted=False)`.
- Attention masking is built explicitly through `get_attention_mask()`
  which combines encoder and decoder padding.
- The VSN does not see an explicit mask: zero-valued padded inputs flow
  through and are softmax-weighted at zero positions; the LSTM packing
  is what guarantees padded steps don't contribute to the recurrent
  state.

PlaytikaOSS's `tft-torch` does NOT use `pack_padded_sequence` and
relies on attention masks plus a causal triangular mask only
[PlaytikaOSS `tft-torch`](https://github.com/PlaytikaOSS/tft-torch).
This is one reason to prefer pytorch-forecasting's pattern: packing is
the canonical PyTorch way to keep LSTM hidden states clean.

**Decision for seq-sklearn.** Use `pack_padded_sequence` at the LSTM
AND zero-out padded positions in the VSN BEFORE its softmax (the
requirements doc already mandates this), AND apply -inf attention
masking at both keys and queries. The mask-correctness test in the
requirements doc covers all three paths.

## Implementation gotchas

1. **Static enrichment ordering.** As noted above, c_e feeds a GRN that
   consumes the post-LSTM add-norm output as its primary input and c_e
   (broadcast across L) as its context input. The architecture draft
   A6 currently inverts the order; fix before coding.

2. **VSN softmax with mask.** The VSN softmax is over the FEATURE
   dimension, not the time dimension. So padding zeroing happens at
   the input level (zero out the padded-row feature embeddings), and
   the softmax over features at a padded row is still well-defined
   even if it produces meaningless weights (downstream uses of those
   timesteps are masked anyway). Verified pattern in pytorch-forecasting
   [_tft.py](https://pytorch-forecasting.readthedocs.io/en/v1.3.0/_modules/pytorch_forecasting/models/temporal_fusion_transformer/_tft.html).

3. **Lookback = 1 entities.** With one valid timestep, the LSTM packs a
   length-1 sequence and produces one hidden output; attention has one
   key/value and self-attends trivially; the last_valid readout picks
   that single position. No special-case code path is needed if pack
   and -inf masking are correct. The mask-correctness test in
   requirements §N1 will catch any regression here.

4. **GRN numerical stability.** GRNs combine an ELU branch with a GLU
   gate and a residual; the original paper uses `LayerNorm` after the
   gate. Both reference implementations follow that order and put
   dropout BETWEEN the linear and the ELU
   [pytorch-forecasting `_tft.py`](https://pytorch-forecasting.readthedocs.io/en/v1.3.0/_modules/pytorch_forecasting/models/temporal_fusion_transformer/_tft.html),
   [google-research `tft_model.py`](https://github.com/google-research/google-research/blob/master/tft/libs/tft_model.py).
   No known FP16 numerical issue; layer-norm at the end stabilizes the
   activation distribution.

5. **Dropout placement.** Apply dropout inside the GRN's
   non-residual branch and inside the attention; do NOT add a final
   dropout on the readout vector before the head (no reference impl
   does this and it just hurts calibration).

6. **Attention "interpretability."** The paper's
   `InterpretableMultiHeadAttention` shares VALUES across heads and
   averages attention weights across heads after softmax. Both
   references implement it as a custom class. This is incompatible
   with vanilla `nn.MultiheadAttention` on a fused QKV tensor; shared-V
   requires K and Q per-head while V is single-head and broadcast.
   Plan: hand-implement, with V replicated across the head dim so SDPA
   can fuse in one call. Memory cost is small at H=128.

7. **`predict` vs train mask.** seq-sklearn has no decoder, so the mask
   is a simple `(B, L)` boolean broadcast to `(B, 1, L, L)` for SDPA.
   Don't carry the forecasting causal triangular mask forward; for a
   classification readout from the last valid timestep, full
   bidirectional attention over the past window is the right default
   and matches encoder-only transformers.

## Recent (2024-2025) refinements

No "TFT v2" exists as a maintained reference. The 2024-2025 literature
treats TFT as a fixed reference and varies only hyperparameter search
or input encoding ([smart-grid TFT + Aquila optimizer](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1542320/full),
[crypto TFT trading strategy](https://www.mdpi.com/2079-8954/13/6/474),
[blood-pressure TFT](https://pmc.ncbi.nlm.nih.gov/articles/PMC11402414/)).
The only architectural variant is the quantum hybrid
[QTFT](https://arxiv.org/abs/2508.04048), not applicable here.
Overfit pressure on small datasets is acknowledged
([Frontiers load-forecasting review](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1797906/full));
mitigations are conventional (dropout, early stopping, weight decay).
No published evidence of attention-head collapse specific to TFT.

## Decisions implied for seq-sklearn v1

1. **LSTM init order.** Call `lstm(x, (c_h, c_c))` to match PyTorch's
   `(h_0, c_0)` signature. Fix the architecture A6 doc which currently
   writes `(c_c, c_h)` as a tuple.

2. **Static enrichment block placement.** c_e feeds a GRN whose primary
   input is the post-LSTM add-norm output. Static enrichment GRN
   output is the INPUT to attention, not its output. A6 needs a
   correction here.

3. **Attention implementation.** Hand-implement
   `InterpretableMultiHeadAttention` with shared-V, using
   `F.scaled_dot_product_attention(..., attn_mask=mask)` with V
   broadcast across the head dimension. Wrap in
   `torch.nn.attention.sdpa_kernel([SDPBackend.MATH])` for the ONNX
   export path
   [SDPA API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

4. **Mask handling.** `pack_padded_sequence(enforce_sorted=False)` at
   the LSTM, zero-out padded inputs into VSN, additive -inf mask at
   attention keys AND queries.

5. **Readout default.** `last_valid` for both classifier and regressor;
   `mean_pool` as the second option. No CLS token in v1 (no prior art
   on TFT for it and it adds a learnable parameter with no clear win).

6. **Head shape.** Raw logits for classification, raw scalars for
   regression, with a `(out_dim, n_quantiles)` reshape for quantile
   regression. Matches the requirements doc.

7. **ONNX path.** Export with opset 17 via the math SDPA backend.
   Track [PyTorch issue #149662](https://github.com/pytorch/pytorch/issues/149662)
   for the eventual opset-23 native Attention op; revisit when stable.

8. **Numerical / dropout defaults.** GRN order matches both references
   (linear -> ELU -> linear -> GLU gate -> add residual -> LayerNorm).
   Dropout = 0.1 inside GRN's pre-gate branch and inside attention; no
   dropout on the readout vector. Hidden size 128, attention_heads 4
   per the config defaults.

9. **No CLS, no causal mask.** v1 attention is bidirectional over the
   past window. The forecasting causal mask is dropped along with the
   decoder.

10. **Compute all four context vectors.** Even though the requirements
    doc deferred the "compute and discard the decoder-bound ones"
    decision, both references compute all four cheaply and v1 uses all
    four (c_s, c_e, c_h, c_c). Keep the four-GRN static block. There
    are no unused context vectors in v1.
