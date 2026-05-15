# TST (Zerveas et al., 2021), research notes for seq-sklearn v2

## Source citations

- Paper: Zerveas, Jayaraman, Patel, Bhamidipaty, Eickhoff. "A Transformer-based
  Framework for Multivariate Time Series Representation Learning." KDD 2021.
  arXiv preprint Oct 2020. <https://arxiv.org/abs/2010.02803>,
  <https://dl.acm.org/doi/10.1145/3447548.3467401>.
- Original code: <https://github.com/gzerveas/mvts_transformer>. Model file:
  <https://github.com/gzerveas/mvts_transformer/blob/master/src/models/ts_transformer.py>.
- tsai reimplementation: <https://timeseriesai.github.io/tsai/models.tst.html>,
  source <https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TST.py>.
- thuml Time-Series-Library: no TST entry; library is forecasting-centric, model
  list confirmed at <https://github.com/thuml/Time-Series-Library/tree/main/models>.
- ConvTran (tAPE / eRPE refinement of TST positional encoding), Foumani et al.,
  ECML-PKDD 2023, <https://arxiv.org/abs/2305.16642>, code
  <https://github.com/Navidfoumani/ConvTran>.
- Positional-encoding survey 2025: <https://arxiv.org/html/2502.12370v1>.

## Architecture (block flow with dimensions)

Per `ts_transformer.py` and the paper (sec. 3):

```
X: (batch, max_len, feat_dim)               raw multivariate series
  -> permute -> (max_len, batch, feat_dim)  PyTorch nn.Transformer convention
  -> Linear(feat_dim, d_model) * sqrt(d_model)
  -> + LearnablePositionalEncoding(max_len, d_model)
  -> Dropout
  -> TransformerEncoder x num_layers
       each layer:
         MultiheadAttention(d_model, n_heads,
                            src_key_padding_mask = ~padding_mask)
         + residual
         BatchNorm1d over (batch * seq) per feature  (POST-NORM)
         FFN: Linear(d_model, dim_feedforward) -> GELU -> Dropout -> Linear
         + residual
         BatchNorm1d                                 (POST-NORM)
  -> GELU
  -> permute back -> (batch, max_len, d_model)
  -> Dropout
  -> task head
```

Defaults from the paper, table 2: `d_model = 128`, `n_heads = 8` (16 for the
largest variant), `num_layers = 3`, `dim_feedforward = 256` (or 512), dropout
0.1. Source: <https://arxiv.org/abs/2010.02803>.

Layer norm placement is **post-norm with BatchNorm1d substituted for
LayerNorm**, contra the original "Attention Is All You Need". The paper (sec.
3.1) argues BatchNorm better handles the outlier scale of raw time-series
values; verified in `TransformerBatchNormEncoderLayer.forward()` in
<https://github.com/gzerveas/mvts_transformer/blob/master/src/models/ts_transformer.py>.
The plain LayerNorm variant (`TransformerEncoderLayer` from torch) is also
selectable via `norm='LayerNorm'` and is likewise post-norm.

## Classification head (verified from reference impl)

The paper does **not** use mean-pool, CLS token, or last-valid pooling. It uses
**flatten-concat**:

```python
# TSTransformerEncoderClassiregressor.forward, ts_transformer.py
output = output * padding_masks.unsqueeze(-1)   # zero padded positions
output = output.reshape(output.shape[0], -1)    # (batch, max_len * d_model)
output = self.output_layer(output)              # nn.Linear(d_model*max_len, num_classes)
```

Source: <https://github.com/gzerveas/mvts_transformer/blob/master/src/models/ts_transformer.py>.

Loss is `CrossEntropyLoss` over `num_classes` logits; no softmax in the head.
The same flatten-concat head is used by the tsai reimplementation
(<https://timeseriesai.github.io/tsai/models.tst.html> source), confirming this
is the canonical TST head, not an idiosyncrasy.

Consequence: TST's classifier requires a **fixed `max_len`** at construction
time. Variable-length sequences must be padded to `max_len`; the padding mask
zeros their contribution before flattening.

## Regression head (verified from reference impl)

Same class, `TSTransformerEncoderClassiregressor`, same flatten-concat,
`nn.Linear(d_model * max_len, num_classes)` where `num_classes` is repurposed
as the regression output dimension (per-sample scalar => `num_classes=1`,
multi-output => `num_classes=k`). The paper's regression experiments
(BeijingPM25Quality, AppliancesEnergy, etc.) use MSE loss; no quantile or
distributional head.

The README confirms: "the simplest classifier/regressor can be either regressor
or classifier because the output does not include softmax"
(<https://github.com/gzerveas/mvts_transformer/blob/master/README.md>).

## Mask handling

Two masks, kept distinct:

1. **Padding mask** `padding_masks: (batch, max_len)` bool, 1 = real, 0 = pad.
   Passed to `nn.MultiheadAttention` as `src_key_padding_mask=~padding_masks`
   (PyTorch convention: True = ignore). Also used as a multiplicative gate
   `output * padding_masks.unsqueeze(-1)` before flattening, so padded
   timesteps contribute zero to the classifier logits.

2. **Noise mask** `target_masks: (batch, max_len, feat_dim)`, used **only**
   during unsupervised pre-training. Random geometric-distribution runs of
   masked positions per feature (sec. 3.2 of the paper). The model is asked to
   reconstruct masked entries.

Pre-training loss (from `src/running.py`):
`target_masks = target_masks * padding_masks.unsqueeze(-1)`, then MSE is
summed only over positions where the combined mask is 1. Padded positions are
never scored; observed (un-noised) positions are not scored either, so the
model has to actually impute rather than copy. Source:
<https://github.com/gzerveas/mvts_transformer/blob/master/src/running.py>.

## Pre-training vs. downstream-only usage in seq-sklearn

seq-sklearn v2 is downstream-only (classification + regression on labelled
panels). The pre-training pipeline does **not** carry over: it requires the
self-supervised `TSTransformerEncoder` (imputation head, not classifier head),
a separate optimiser, and the noise-mask sampler. The two are clean siblings
in the upstream repo, selected by `model_factory()` on the task argument, so
adopting only the supervised path is straightforward.

Implications:

- Build only `TSTransformerEncoderClassiregressor`. The imputation sibling and
  the `MaskedMSELoss` machinery are out of scope for v2.
- The headline empirical claim of the paper, that TST beats supervised SOTA
  *after* unsupervised pre-training on the same training set, does **not**
  transfer when we skip pre-training. The supervised-only TST is competitive
  but not dominant (paper table 1, "TST sup. only" column). This is fine for
  v2 positioning where TFT is the flagship and TST is the simple-transformer
  baseline.
- Future v3 could add the pre-training path as an optional `fit_pretrain`
  hook; the architecture file would be reused intact.

## Implementation gotchas

1. **`max_len` is baked into the classifier head**. The output linear is
   `Linear(d_model * max_len, num_classes)`. Changing sequence length at
   inference requires re-padding or re-instantiation. seq-sklearn's config
   should expose `max_len` and pad all batches to it. Verified at
   <https://github.com/gzerveas/mvts_transformer/blob/master/src/models/ts_transformer.py>.

2. **BatchNorm over (batch, seq) needs care with tiny batches**. The custom
   `TransformerBatchNormEncoderLayer` reshapes to apply `nn.BatchNorm1d` over
   pooled `batch * seq_len` samples per feature. Single-sample batches break
   it. seq-sklearn should either default to LayerNorm (`norm='LayerNorm'`) or
   enforce `batch_size >= 2` in training.

3. **PyTorch tensor convention is `(seq, batch, d_model)`**, not
   `(batch, seq, d_model)`. The reference code permutes on entry and exit;
   easy to drop a permute and silently mis-pool.

4. **Padding-mask sign convention**. PyTorch's MHA wants `True = ignore`;
   Zerveas stores `True = keep` and inverts at the call site
   (`src_key_padding_mask=~padding_masks`). seq-sklearn's `BatchSpec` already
   uses `True = real`; one inversion at the encoder boundary keeps us aligned.

5. **No CLS token, no mean-pool**. Do not "improve" the head by replacing
   flatten-concat with mean-pool unless we accept that the model is no longer
   TST. tsai keeps flatten-concat
   (<https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TST.py>).

6. **Input projection has a `sqrt(d_model)` scaling** (`* math.sqrt(d_model)`)
   matching the original "Attention Is All You Need" recipe. Easy to omit and
   then puzzle over slow convergence.

## Recent refinements

- **ConvTran (Foumani et al., ECML-PKDD 2023)**: drops TST's learnable absolute
  PE for **tAPE** (time-aware absolute positional encoding scaled by sequence
  length and `d_model`) plus **eRPE** (efficient relative positional encoding
  applied after softmax), and swaps the input linear for a convolutional
  embedding. Reports significant accuracy gains over TST on 32 UEA datasets.
  <https://arxiv.org/abs/2305.16642>,
  <https://github.com/Navidfoumani/ConvTran>.
- **PE survey (Feb 2025)**: catalogues TST as the canonical learnable-PE
  baseline; notes that for classification (as opposed to forecasting), relative
  PE consistently beats absolute. <https://arxiv.org/html/2502.12370v1>.
- No major issues reported with the supervised-only TST itself; the original
  repo is unmaintained (last substantive commit 2022) but stable. Most 2024 to
  2025 transformer-for-time-series work has gone toward forecasting
  (iTransformer, PatchTST refinements, TimeMixer) rather than classification,
  leaving TST + ConvTran as the de facto classification baselines.

## Decisions implied for seq-sklearn v2

1. **Implement TST as the simple-transformer baseline.** Single file under
   `src/seq_sklearn/models/tst.py`. Mirror the Zerveas head exactly:
   flatten-concat + Linear, with `padding_mask` zeroing pads. No CLS, no
   mean-pool, no "modernisation".

2. **Require `max_len` in the config.** It is a constructor argument, not an
   inferred quantity. Document the implication: variable-length panels must
   be padded; the head is not length-agnostic.

3. **Default `norm='LayerNorm'` (pre-norm wrapper available later).**
   BatchNorm is the paper default and slightly stronger, but LayerNorm
   sidesteps the small-batch failure mode and matches seq-sklearn's other
   models. Expose `norm` as a config field; ship LayerNorm default, allow
   `'BatchNorm'` for parity experiments.

4. **Learnable absolute positional encoding.** Fixed sinusoidal is selectable
   via config (`pos_encoding='fixed'`) for ablations.

5. **Defaults**: `d_model=128, n_heads=8, num_layers=3, dim_feedforward=256,
   dropout=0.1, activation='gelu'`. Lifted from the paper's table 2.

6. **Single supervised path only in v2.** No masked-pretraining head, no
   `MaskedMSELoss`, no noise-mask sampler. Flag as a v3 extension; the
   architecture file will be reused unchanged.

7. **Classifier and regressor share the same backbone class** with
   `num_outputs` (renaming Zerveas's `num_classes`) controlling the output
   width. CE loss vs MSE/Gaussian-NLL is selected by the surrounding sklearn
   estimator (`SeqClassifier` vs `SeqRegressor`), not the model.

8. **Readout choice diverges from TFT.** TFT's classifier uses last-valid or
   mean-pool; TST uses flatten-concat. Do not unify these by forcing TST onto
   the TFT readout API; expose a TST-specific readout that knows about
   `max_len`. This is the price of literal parity with the paper.

9. **Skip ConvTran for v2.** Note it as a v2.x extension once the core TST is
   landed and benchmarked; tAPE + eRPE are a drop-in PE swap and a relative-PE
   attention modification, both contained changes.

10. **TST vs PatchTST vs TimesNet for "cleanest fit"**: confirmed. PatchTST is
    forecasting-first; its classification adaptations (PAttn in
    Time-Series-Library) are recent and less canonical. TimesNet is a
    multi-task convnet over reshaped 2D periodicity blocks, conceptually
    distant from the sklearn-style "encode + head" contract. TST's
    `TSTransformerEncoderClassiregressor` literally has classification and
    regression as first-class targets sharing one class, which maps onto
    seq-sklearn's `SeqClassifier` / `SeqRegressor` split with zero
    architectural strain. Keep TST as the v2 transformer baseline.

Sources:
- [Zerveas et al. 2021, arXiv 2010.02803](https://arxiv.org/abs/2010.02803)
- [mvts_transformer ts_transformer.py](https://github.com/gzerveas/mvts_transformer/blob/master/src/models/ts_transformer.py)
- [mvts_transformer running.py](https://github.com/gzerveas/mvts_transformer/blob/master/src/running.py)
- [tsai TST docs](https://timeseriesai.github.io/tsai/models.tst.html)
- [tsai TST.py](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TST.py)
- [Time-Series-Library models](https://github.com/thuml/Time-Series-Library/tree/main/models)
- [ConvTran, Foumani 2023](https://arxiv.org/abs/2305.16642)
- [PE survey 2025](https://arxiv.org/html/2502.12370v1)
