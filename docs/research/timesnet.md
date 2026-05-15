# TimesNet research notes

Scope: evaluate TimesNet (Wu et al., ICLR 2023) as a candidate v2 backbone in
seq-sklearn for classification and regression on tabular panel time-series.
Forecasting and anomaly detection are out of scope for this library and are
not evaluated here.

## Source citations

- Paper, arXiv: https://arxiv.org/abs/2210.02186
- ICLR 2023 camera-ready PDF: https://ise.thss.tsinghua.edu.cn/~mlong/doc/TimesNet-iclr23.pdf
- OpenReview: https://openreview.net/pdf?id=ju_Uqw384Oq
- Original code repo (archived, points to TSL): https://github.com/thuml/TimesNet
- Maintained reference impl, Time-Series-Library (TSL): https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py
- TSL classification experiment: https://github.com/thuml/Time-Series-Library/blob/main/exp/exp_classification.py
- TSL UEA loader and collate: https://github.com/thuml/Time-Series-Library/blob/main/data_provider/uea.py
- TSLANet (ICML 2024), used as a 2024 comparison baseline: https://arxiv.org/pdf/2404.08472
- Nixtla neuralforecast TimesNet (forecasting-only port): https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/timesnet.py

## Architecture

TimesNet is a CNN, not a transformer. The name is misleading. There is no
self-attention anywhere in the model. The "Times" refers to the period-based
2D reshape, and the network is a stack of 2D Inception-style convolutional
blocks. Source: https://arxiv.org/abs/2210.02186 and the TSL implementation
https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py.

Block flow per TimesBlock:

1. Input `x` shape `[B, T, C]` (after token embedding `d_model = C`).
2. `FFT_for_Period(x, k)`: compute `rfft` along time, average amplitude over
   batch and channels, zero the DC bin, take top-k frequency bins, derive
   `period_i = T // freq_i`. The k amplitudes become softmax weights for
   step 5. Code: https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py.
3. For each of the k periods, right-pad `x` along time to a multiple of
   `period_i`, then reshape `[B, T_pad, C] -> [B, C, T_pad // period_i, period_i]`.
   Rows are inter-period variation, columns are intra-period variation.
4. Apply an Inception-style 2D conv block (parameter-shared across the k
   reshapes) and reshape back to `[B, T_pad, C]`, crop to `T`.
5. Stack the k outputs and combine with the softmax-normalised period
   amplitudes from step 2. Residual-add to the input.

The full model is `embedding -> N x (TimesBlock + LayerNorm) -> task head`.
Classification head is in the next section. Source:
https://arxiv.org/abs/2210.02186 section 3 and TSL `models/TimesNet.py`.

## Classification mechanics from reference implementation

The TSL `Model.classification(x_enc, x_mark_enc)` method, lifted verbatim from
https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py:

```
enc_out = self.enc_embedding(x_enc, None)        # [B, T, d_model]
for i in range(self.layer):
    enc_out = self.layer_norm(self.model[i](enc_out))
output = self.act(enc_out)                        # GELU
output = self.dropout(output)
output = output * x_mark_enc.unsqueeze(-1)        # zero pad rows
output = output.reshape(output.shape[0], -1)      # [B, T * d_model]
output = self.projection(output)                  # Linear(T*d_model, num_class)
```

Three things are load-bearing:

1. The head is a flatten plus a single `Linear(seq_len * d_model, num_class)`.
   No pooling, no CLS token. The model has a fixed input length `T` baked
   into the projection weight shape. Source: TSL `models/TimesNet.py`.
2. The padding mask `x_mark_enc` is a binary float tensor of shape `[B, T]`
   with `1` for valid timesteps and `0` for right-pad. It is multiplied into
   the embeddings before flatten so padded positions contribute zero to the
   linear head. Confirmed in TSL `data_provider/uea.py` `padding_mask()` and
   the docstring "1 means keep element at this position (time step)".
3. The pre-padding to a global `max_seq_len = max(train_data.max_seq_len,
   test_data.max_seq_len)` happens in `exp_classification.py`. All UEA
   sequences are right-padded with zeros to this max. Source:
   https://github.com/thuml/Time-Series-Library/blob/main/exp/exp_classification.py.

UEA classification reported 73.6% average accuracy across 10 datasets,
beating Rocket and DLinear in the original paper (Table 17,
https://arxiv.org/abs/2210.02186). TSLANet later reported 72.73% for
TimesNet vs 77.56% for TSLANet on a 26-dataset slice
(https://arxiv.org/pdf/2404.08472), so TimesNet is no longer SOTA on UEA in
2024, but it remains competitive.

## Mask handling for variable-length

The flatten-and-Linear head means TimesNet does not naturally accept
variable-length input. The reference impl handles it by:

- Globally fixing `T = max_seq_len` across the whole dataset, right-padding
  every shorter sequence with zeros.
- Passing a `[B, T]` binary validity mask `x_mark_enc` and multiplying it
  into the post-block embeddings before flatten, so padded positions
  contribute zero to the head.
- Importantly, the mask is NOT used inside TimesBlock. The FFT,
  period-reshape, and 2D conv all see padded values as real zeros. There is
  no mask-aware FFT and no mask-aware convolution.

Sources: TSL `models/TimesNet.py` and TSL `data_provider/uea.py`.

## Regression adaptation feasibility

The paper does not evaluate sequence-level regression on a panel input
(continuous target per sequence). TSL has no regression task head. Adding
one is mechanically trivial: keep everything up to the classification head,
swap the final `Linear(T * d_model, num_class)` for `Linear(T * d_model, 1)`
or `Linear(T * d_model, n_targets)`, train with MSE or Huber.

Open question: with a flatten head, regression on long sequences burns a
`T * d_model * n_targets` weight matrix. For `T = 500, d_model = 64,
n_targets = 1` that is 32k params, fine. For long sequences this gets
expensive fast and a `mean(masked) -> Linear(d_model, n_targets)` pooled
head is the obvious cheaper variant. The paper does not justify the flatten
choice over pooling; the natural seq-sklearn extension would offer both.

## Implementation gotchas

1. **FFT on padded input.** `FFT_for_Period` runs on the post-embedding
   tensor including padded zero rows, with no mask awareness. Right-padding
   a periodic signal with zeros injects spurious low-frequency energy and
   distorts the amplitude spectrum the model uses to pick its top-k periods.
   On heavily padded short sequences in a batch with a much longer
   `max_seq_len`, the chosen periods can be wrong. The reference impl does
   not address this. Source:
   https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py.

2. **Period selection picks DC neighbours.** `FFT_for_Period` zeros only
   the DC bin (`frequency_list[0] = 0`) before topk. Very-low-frequency
   components from non-stationary trends or zero-pad bias can still
   dominate, yielding period equal to `T // 1 = T` (or `T // 2`), which
   degenerates the 2D reshape to `[B, C, 1, T]` (a trivial 1-row map) and
   the 2D conv reduces to a 1D conv. Source: same file.

3. **Period chosen per batch, not per sample.** `frequency_list = abs(xf).mean(0).mean(-1)`
   averages amplitudes over the batch dim. The k periods are batch-global,
   not per-sample. A batch with heterogeneous periodicities gets a single
   compromise period set. Source: same file.

4. **Fixed `seq_len` baked into the head.** Because `self.projection` is
   `Linear(d_model * seq_len, num_class)`, the model cannot be applied at
   inference time to a sequence longer or shorter than the training
   `seq_len` without either truncating, padding, or re-initialising the
   head. This is a real friction point for an sklearn-style `.fit/.predict`
   contract on heterogeneous-length panels. Source: same file.

5. **Top-k is non-differentiable.** `torch.topk` over the FFT amplitude has
   no gradient flowing into the period choice. Periods are selected from a
   discrete forward pass each step, which is fine in practice but means
   the model cannot learn to prefer periods; it only learns conv weights
   conditioned on whatever the FFT proposes. Source: same file.

6. **Conv kernels are not period-aware.** The 2D Inception block is the
   same regardless of which period produced the reshape. The "period
   adaptivity" lives entirely in the reshape and the softmax mixing
   weights, not the convolution.

## Recent refinements

- TSLANet (ICML 2024, https://arxiv.org/pdf/2404.08472) reports TimesNet at
  72.73% UEA vs TSLANet at 77.56% with 84% fewer parameters. TSLANet uses
  an adaptive spectral block plus SSL pretraining.
- Domain variants in 2024-2025 swap the Inception block for ConvNeXt or
  fuse time and frequency features, e.g. HVAC fault diagnosis
  (https://www.tandfonline.com/doi/full/10.1080/19401493.2025.2459714).
  The FFT-period-reshape generalises; the conv block is improvable.
- The Nixtla port
  (https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/timesnet.py)
  is forecast-only and does not implement the classification head.
- No "TimesNet v2" exists as of May 2026. Superseded on UEA by TSLANet and
  ModernTCN but still a frequent baseline.

## Decisions implied for seq-sklearn v2

1. Include TimesNet as a v2 backbone, behind the same `BaseSequenceEstimator`
   contract as PatchTST and TST. Its CNN nature gives a useful inductive
   bias different from the patch transformers.

2. **Fixed sequence length is mandatory at fit time.** Pin `seq_len` from
   the training set max; validate that predict-time sequences are pre-padded
   or pre-truncated to this exact length. Document this constraint
   explicitly in the estimator docstring. The flatten head leaves no
   alternative without architectural change.

3. **Offer a pooled head variant.** Add a config switch `head: "flatten" |
   "mean_pool"` where `mean_pool` masks invalid timesteps and averages,
   then `Linear(d_model, n_outputs)`. This decouples params from `seq_len`,
   matches how variable-length sequences are typically handled in TST and
   PatchTST, and is the obvious regression default.

4. **Mask-aware FFT.** Pass the validity mask into period detection.
   Compute FFT per-sample on the valid prefix only and average amplitudes
   over valid positions. Fixes gotcha 1 and gotcha 3 at the cost of a
   per-sample FFT. Cheap relative to the conv stack.

5. **Guard against degenerate periods.** Clamp detected periods to
   `[2, T // 2]` and skip any reshape yielding fewer than 2 rows. Falls
   back to a 1D conv path. Reference impl has no such guard.

6. **Regression head.** Same pooled or flatten head with `n_outputs`
   instead of `num_class` and MSE or Huber loss. No architectural change.

7. **Compared to PatchTST and TST.** PatchTST and TST use attention with a
   native attention mask, handling variable-length cleanly without the
   global-max-pad trick. TimesNet does not, by construction. Advertise
   this: prefer PatchTST or TST for heterogeneous lengths; consider
   TimesNet for roughly homogeneous lengths with strong periodicity.

8. **Defer SSL pretraining.** TSLANet's gains lean on SSL. Out of scope
   for v2 per requirements; revisit in v3.
