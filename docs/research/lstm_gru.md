# LSTM and GRU for time-series classification and regression

Research notes for v3 recurrent estimators. Targets bounded-length
panel sequences (L=12 to 60), classification and regression heads.

## Source citations

- Hochreiter and Schmidhuber, LSTM (1997). https://www.bioinf.jku.at/publications/older/2604.pdf
- Cho et al., RNN Encoder Decoder, origin of GRU (2014). https://arxiv.org/abs/1406.1078
- Karim et al., LSTM-FCN (2018). https://arxiv.org/abs/1709.05206
- Karim et al., MLSTM-FCN (2019). https://arxiv.org/abs/1801.04503
- Karim et al., Insights into LSTM-FCN (2019). https://arxiv.org/abs/1902.10756
- Fawaz et al., Deep learning for TSC: a review (2019). https://arxiv.org/abs/1809.04356
- Ruiz et al., The great multivariate TSC bake off (2021). https://link.springer.com/article/10.1007/s10618-020-00727-3
- Gal and Ghahramani, variational dropout for RNNs (2016). https://arxiv.org/abs/1512.05287
- Merity et al., AWD-LSTM (2017). https://arxiv.org/abs/1708.02182
- Mohajerin and Waslander, trainable initial hidden states (2020). https://arxiv.org/abs/2007.06848
- tsai RNN source. https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN.py
- aeon deep learning (TensorFlow-based). https://www.aeon-toolkit.org/en/latest/examples/classification/deep_learning.html
- pyts (no RNN classifiers). https://github.com/johannfaouzi/pyts
- PyTorch LSTM docs. https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html

## Mask handling: pack_padded_sequence vs. masking patterns

`pack_padded_sequence` plus `pad_packed_sequence` remains the canonical
PyTorch path for variable-length input to `nn.LSTM` / `nn.GRU`. The
packed form skips padding at every recurrent step, so the forward
direction never sees pad positions and `h_n` corresponds to each
sequence's true final timestep.
https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html

For bidirectional layers, PyTorch packs the same lengths for the reverse
direction. The backward pass reads each sequence in reverse starting
from the last valid timestep, so it also skips padding correctly. There
is no separate user-supplied backward mask. Gotcha: `h_n` for the
reverse direction is the hidden state at t=0, not t=L-1, so the two
directions must be assembled carefully on readout.
https://discuss.pytorch.org/t/how-bilstm-works-with-padding-pack-padded-sequence/132898

Open issue pytorch/pytorch#517 documents that inter-layer dropout in
stacked recurrent layers does not get a per-step mask, which can leak
signal through padded positions in multi-layer bidirectional stacks.
Mitigation: pack again between layers or mask outputs before pooling.
https://github.com/pytorch/pytorch/issues/517

2024+ implementations still use pack/unpack because that is the only
path that gets the fused cuDNN kernel. Attention-style boolean masks
sit on top of unpacked outputs for pooling and attention readouts,
they do not replace packing. tsai's `_RNN_Base` skips packing entirely
and assumes fixed-length input
(https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN.py),
which works for UCR/UEA where each dataset has a single length but
fails on true variable-length panels.

## Recurrent dropout: variational vs. Bernoulli, modern recommendation

`nn.LSTM(dropout=p)` applies inter-layer Bernoulli dropout between
stacked layers only, fresh mask per timestep. No recurrent dropout on
the h_{t-1} to h_t path. Silently does nothing if `num_layers=1`.
https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html

Gal and Ghahramani's variational dropout fixes one dropout mask per
sequence and reuses it across all timesteps, on inputs and recurrent
connections. Theoretically motivated for RNNs, outperforms per-step
Bernoulli on language modeling and TSC. https://arxiv.org/abs/1512.05287

In 2025 variational remains the recommended form for recurrent
regularization, but the dominant TSC architectures (LSTM-FCN,
MLSTM-FCN, InceptionTime hybrids) keep the RNN shallow and rely on
the convolutional branch plus weight decay rather than recurrent
dropout. https://arxiv.org/abs/1801.04503

PyTorch has no native variational dropout. Custom implementations
sample a mask per sequence and apply it inside a manual unroll with
`LSTMCell` / `GRUCell`, which loses cuDNN. Reference impls:
mourga/variational-lstm (https://github.com/mourga/variational-lstm)
and AWD-LSTM weight drop, which dropouts the hidden-to-hidden weight
matrix directly and preserves cuDNN. https://arxiv.org/abs/1708.02182

For L=12 to 60 with shallow stacks the practical difference between
variational and per-step Bernoulli is small. AWD-LSTM weight drop is
the best compromise: regularizes recurrent connections without
breaking cuDNN.

## Initial hidden state strategies

Three strategies appear in the literature.

Zero init is the PyTorch default and the standard in TSC. LSTM-FCN,
MLSTM-FCN, and the tsai `_RNN_Base` all init to zero.
https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN.py

Learned-parameter init treats `h_0` and `c_0` as trainable parameters
shared across the dataset. Mohajerin and Waslander show modest gains
for financial TSC with trainable init.
https://arxiv.org/abs/2007.06848

Per-entity init conditions `h_0` on static covariates through a small
MLP. This is the pattern used by the original TFT and by panel models
that carry entity identity. It matters most when sequences are short
relative to the warmup horizon, which is exactly the L=12 to 60 regime.
https://arxiv.org/abs/1912.09363

Standard in the TSC benchmark literature is zero init. The seq-sklearn
recurrent skeleton should expose this as a config knob with zero as the
default and learned and per-entity as opt-in.

## Readout strategies for classification

Four patterns dominate.

Last-valid-timestep readout: take `h_T` at each sequence's true final
position. With packing this falls out of `h_n` directly. This is the
most common choice and what tsai uses
(https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN.py).

Masked mean-pool: average the hidden states across valid timesteps. Used
in several deep TSC baselines and robust when discriminative information
is spread across the sequence rather than concentrated at the end.
https://arxiv.org/abs/1809.04356

Attention readout: learn a softmax over timesteps and take the weighted
sum of hidden states. ALSTM-FCN uses this pattern and improves over
plain LSTM-FCN on a majority of UCR datasets, with the additional
benefit of interpretable per-timestep weights.
https://arxiv.org/abs/1709.05206

Concat-of-first-and-last for bidirectional: take the forward `h_T` and
the backward `h_0` (which is the reverse direction's view of the last
real timestep) and concatenate. This is the canonical bidirectional
readout. https://discuss.pytorch.org/t/how-bilstm-works-with-padding-pack-padded-sequence/132898

For seq-sklearn the `_readout` abstract method should support at least
last, mean, and attention. Attention adds one learnable vector and a
softmax, cheap to implement and frequently the best on heterogeneous
TSC datasets.

## Truncated BPTT relevance for L=12 to 60

TBPTT chunks long sequences into segments of k timesteps and backprops
only within each chunk. Practical k sits in 20 to 200; full BPTT stays
tractable up to several hundred timesteps on modern GPUs.
https://machinelearningmastery.com/gentle-introduction-backpropagation-time/

For L=12 to 60 full BPTT is the correct choice. Activation memory is
O(L * batch * hidden); at L=60, batch=512, hidden=256 that fits in
under 100 MB and is trivial on any modern GPU. None of tsai, the
LSTM-FCN family, or InceptionTime use TBPTT for TSC.
https://arxiv.org/abs/1801.04503

`bptt_window` should be an extension point in the v1 abstract base
class but default to `None` meaning full BPTT. v3 surfaces it without
implementing chunked TBPTT, so long-sequence users get a clean API
when it lands later.

## GRU vs. LSTM empirical defaults

Across recent comparative studies the result is consistent: GRU and
LSTM are statistically indistinguishable on most TSC and forecasting
benchmarks, with GRU running 20 to 30 percent faster per epoch due to
having three gates instead of four and no separate cell state.

A 2024 study on CNN-feature-extractor RNN variants reports GRU
outperforming LSTM in classification accuracy on its benchmark.
https://www.sciencedirect.com/science/article/pii/S1877050924025717

A 2025 Monte Carlo comparison of nine RNN variants across three
real-world datasets finds no statistically significant differences
between LSTM, GRU, and hybrid configurations.
https://pmc.ncbi.nlm.nih.gov/articles/PMC12329085/

Practical guidance for seq-sklearn: default to GRU for the user-facing
quickstart (faster, fewer params, comparable accuracy on short panel
sequences), keep LSTM as the canonical option for parity with the
classical TSC literature. Both must be available; neither is the
strictly correct default.

## Implementation gotchas

NaN gradients and exploding gradients are the dominant failure modes
when training recurrent networks on tabular panel data. Recommended
mitigations:

- Apply `torch.nn.utils.clip_grad_norm_` with max norm in [1.0, 5.0]
  every step. https://apxml.com/courses/advanced-pytorch/chapter-3-optimization-training-strategies/gradient-clipping-accumulation
- Initialize the forget-gate bias to 1.0 so the cell retains state at
  startup. tsai follows the Keras convention here:
  https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN.py
- Use orthogonal init for hidden-to-hidden weights to keep singular
  values near 1 and bound gradient growth through time. Same source.
- For bidirectional layers, be careful that loss is computed only on
  valid (non-padded) outputs. The reverse direction can leak through
  padded positions if you forget to mask before pooling.
  https://github.com/pytorch/pytorch/issues/517
- Mixed-precision (autocast bfloat16) on LSTM/GRU is safe in PyTorch
  2.x because the cuDNN kernel handles dtype conversion internally;
  fp16 occasionally produces NaN on long sequences and should be
  guarded with a gradient scaler.
- Always sort batches by length descending before `pack_padded_sequence`
  unless using `enforce_sorted=False`, which adds a sort internally.
  https://discuss.pytorch.org/t/how-to-pack-padded-sequence-and-pad-packed-sequence/124646

## Decisions implied for seq-sklearn

v1 skeleton (`RecurrentSequenceEstimator` abstract base class):

- `_init_hidden(batch, static_features) -> tuple[Tensor, ...]`. Returns
  the per-cell-type tuple (LSTM: `(h_0, c_0)`, GRU: `(h_0,)`). The
  default implementation returns zeros. The hook lets v3 subclasses
  implement learned and per-entity init without API change.
- `_readout(outputs, lengths, hidden) -> Tensor`. Receives unpacked
  outputs `(B, L, H)`, true lengths `(B,)`, and the final hidden tuple.
  Returns `(B, readout_dim)`. Implementations: last, mean, attention.
- `_bptt_window: int | None`. None means full BPTT. v1 ships with None
  as the only supported value; v3 implements chunked TBPTT when set.
- Config knobs: `bidirectional: bool`, `recurrent_dropout: float`,
  `recurrent_dropout_kind: Literal["variational", "weight_drop", "none"]`,
  `hidden_init_strategy: Literal["zero", "learned", "per_entity"]`,
  `readout: Literal["last", "mean", "attention"]`, `bptt_window: int | None`.

v3 concrete models (`LSTMClassifier`, `LSTMRegressor`, `GRUClassifier`,
`GRURegressor`):

- Backed by `nn.LSTM` / `nn.GRU` with cuDNN. Use
  `pack_padded_sequence(enforce_sorted=False)` for variable-length
  panels; skip packing when all lengths in a batch are equal.
- Default `recurrent_dropout=0.0`, `recurrent_dropout_kind="weight_drop"`
  (AWD-LSTM style, preserves cuDNN). Variational opt-in via manual
  unroll with `LSTMCell`.
- Default `hidden_init_strategy="zero"`, `readout="last"`,
  `bidirectional=False`, `bptt_window=None`. Defaults match the TSC
  benchmark literature and tsai.
- Forget-gate bias to 1.0, orthogonal hidden-to-hidden init, Xavier
  normal input-to-hidden init. Match tsai conventions.
- Gradient clipping at 1.0 in the default training callback. Surface
  as a config knob.
- Quickstart docs default to `GRUClassifier` for speed; benchmark
  parity uses `LSTMClassifier`.
