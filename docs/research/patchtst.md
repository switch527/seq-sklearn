# PatchTST research notes (seq-sklearn v2)

## Source citations

- Paper: Nie, Nguyen, Sinthong, Kalagnanam, "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers", ICLR 2023. arXiv 2211.14730. https://arxiv.org/abs/2211.14730
- Official repo (Yuqi Nie): https://github.com/yuqinie98/PatchTST
- Official backbone source: https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/layers/PatchTST_backbone.py
- ETTh1 supervised script (defaults): https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/scripts/PatchTST/etth1.sh
- Time-Series-Library (THUML) model: https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py
- Time-Series-Library classification runner: https://github.com/thuml/Time-Series-Library/blob/main/exp/exp_classification.py
- Hugging Face PatchTST docs (config, classification, regression heads): https://huggingface.co/docs/transformers/en/model_doc/patchtst
- CT-PatchTST (channel attention variant, 2025): https://arxiv.org/html/2501.08620v3
- DualPathTST (cross-variable pathway, 2024): https://www.sciopen.com/article/10.26599/TST.2024.9010195
- PITS, Lee et al., ICLR 2024 (patch-independent embedding): https://arxiv.org/pdf/2312.16427
- TSLANet, 2024 (UEA classification benchmark numbers): https://arxiv.org/pdf/2404.08472

## Architecture (block flow with dimensions, patching mechanics)

Input panel: `(batch=B, seq_len=L, channels=C)`. PatchTST treats each of the `C`
channels as an independent univariate series and reshapes to `(B*C, L)` before
patching ([HF docs](https://huggingface.co/docs/transformers/en/model_doc/patchtst),
[arXiv](https://arxiv.org/abs/2211.14730)).

Patching with length `P` and stride `S` yields `N = floor((L - P) / S) + 1`
patches per channel; the official `etth1.sh` uses `P=16, S=8`
([etth1.sh](https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/scripts/PatchTST/etth1.sh)).
Each patch is linearly projected to `d_model`. Defaults differ across
implementations: Hugging Face ships `d_model=128, num_hidden_layers=3,
num_attention_heads=4, ffn_dim=512, patch_length=1, patch_stride=1`
([HF docs](https://huggingface.co/docs/transformers/en/model_doc/patchtst));
official supervised ETTh1 uses `d_model=16` with 3 layers and 16 heads
([etth1.sh](https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/scripts/PatchTST/etth1.sh));
tsai exposes `d_model=512` as its library default
([tsai](https://timeseriesai.github.io/tsai/models.patchtst.html)).

Block flow:

1. `RevIN` instance normalization on `(B, L, C)`.
2. Permute to `(B, C, L)`, unfold to `(B, C, N, P)`, project to `(B, C, N, d_model)`.
3. Add positional embedding (sinusoidal by default in HF, learned in the
   official repo) and optional CLS token
   ([HF docs](https://huggingface.co/docs/transformers/en/model_doc/patchtst)).
4. Flatten channels into batch: `(B*C, N, d_model)` and feed `num_hidden_layers`
   of standard Transformer encoder blocks. Attention runs over the `N` patch
   tokens, not over `C` channels.
5. Reshape back to `(B, C, N, d_model)` for the task head.

## Classification / regression adaptation

The Time-Series-Library `PatchTST.py` adds a classification branch that takes
the encoder output `(B, C, d_model, N)`, applies a `Flatten(start_dim=-2)`,
then concatenates across channels and projects:
`Linear(d_model * N * enc_in -> num_class)`
([PatchTST.py](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py)).
No CLS token and no temporal pooling: it is a full flatten-then-project. The
classification runner trains with `nn.CrossEntropyLoss()` and standardizes
sequence length to `max(train_data.max_seq_len, test_data.max_seq_len)`
before passing a `padding_mask` to the model
([exp_classification.py](https://github.com/thuml/Time-Series-Library/blob/main/exp/exp_classification.py)).

Hugging Face exposes `PatchTSTForClassification` and `PatchTSTForRegression`
as first-class heads with `num_targets` for class count or regression width
and `pooling_type` in `{"mean", "max", None}`; `use_cls_token=True` switches
the readout to a CLS slot
([HF docs](https://huggingface.co/docs/transformers/en/model_doc/patchtst)).
This is the cleaner canonical adaptation: HF's classification example sets
`use_cls_token=True` with `patch_length=12, stride=12, context_length=512`.
Regression uses MSE on `(batch, num_targets)`
([HF docs](https://huggingface.co/docs/transformers/en/model_doc/patchtst)).

So there are two canonical heads in the wild: flatten-then-project (THUML)
and CLS-or-pool-then-project (HF, IBM Granite). The paper itself proposes
neither; both are downstream community work
([arXiv abstract](https://arxiv.org/abs/2211.14730)).

For supervised regression on a panel input (not forecasting), HF's
`PatchTSTForRegression` is exactly that: encoder output gets pooled or CLS-read,
projected to `num_targets` scalars, optimized with MSE and an optional
`output_range` clamp
([HF docs](https://huggingface.co/docs/transformers/en/model_doc/patchtst)).

## Mask handling

The original paper does not address variable-length inputs; it assumes a
fixed `context_length`
([arXiv](https://arxiv.org/abs/2211.14730)). Two pragmatic patterns dominate:

- THUML pads all sequences to the dataset's global max and passes a
  `padding_mask` of shape `(B, L)` into the model
  ([exp_classification.py](https://github.com/thuml/Time-Series-Library/blob/main/exp/exp_classification.py)).
  The mask sits at the token-time level, before patching.
- Hugging Face accepts `past_observed_mask: BoolTensor[B, L, C]` indicating
  which raw timesteps are observed vs. missing; missing values are zero-filled
  by the input scaler before patching
  ([HF docs](https://huggingface.co/docs/transformers/en/model_doc/patchtst)).

Neither approach masks at the patch level natively. A patch that straddles a
padded region is still embedded; the loss simply ignores its eventual readout
position. For seq-sklearn this implies a derived patch mask of shape
`(B, C, N)` set to false whenever any timestep inside the patch is masked,
applied as an additive `-inf` bias to attention logits and to the readout
aggregation if pooling is used.

## Channel-independence and feature mixing

Channel-independence is a structural choice in the original paper, not a
toggle: the same encoder weights process every channel in parallel and
attention never crosses channels
([arXiv](https://arxiv.org/abs/2211.14730)).
Implementations vary on how rigid this is:

- Official repo: hard-coded channel-independence; cross-channel mixing only
  happens in the forecasting head, which concatenates per-channel patch
  embeddings before the final linear
  ([PatchTST_backbone.py](https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/layers/PatchTST_backbone.py)).
- THUML classification: mixing is the flatten + linear in the head, which
  collapses `(C, d_model, N)` into one vector of length `C * d_model * N`
  before projecting to logits
  ([PatchTST.py](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py)).
- Hugging Face: `share_embedding=True` and `share_projection=True` keep the
  original channel-independent behavior; `channel_attention=True` inserts an
  explicit channel-attention block inside the Transformer
  ([HF docs](https://huggingface.co/docs/transformers/en/model_doc/patchtst)).

For seq-sklearn, "channel" needs to be redefined: time-varying-real,
time-varying-categorical (post-embedding), and broadcast static covariates
all become channels of the encoder input. Cross-channel mixing must happen
somewhere downstream of the channel-independent encoder. The two viable
sites are (a) flatten + linear like THUML, parameter-heavy and tied to a
fixed `(C, N)`, or (b) a channel-axis attention or MLP block before the head,
matching HF's `channel_attention` or iTransformer's variate-as-token framing
([iTransformer comparison summary in CT-PatchTST](https://arxiv.org/html/2501.08620v3)).

## Implementation gotchas

- Patch count `N` depends on `L`, `P`, `S` and is fixed at model build time.
  Variable-length panels need padding or interpolation; the THUML choice of
  global max length is wasteful but simple
  ([exp_classification.py](https://github.com/thuml/Time-Series-Library/blob/main/exp/exp_classification.py)).
- HF's `patch_length=1, patch_stride=1` defaults degenerate to a per-timestep
  Transformer; never use these for a real run
  ([HF docs](https://huggingface.co/docs/transformers/en/model_doc/patchtst)).
- `d_model` ranges from 16 (official supervised scripts) to 512 (tsai default);
  it is dataset-dependent and a primary tuning knob
  ([etth1.sh](https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/scripts/PatchTST/etth1.sh),
  [tsai](https://timeseriesai.github.io/tsai/models.patchtst.html)).
- THUML's flatten head produces `head_nf * enc_in` input features to the final
  Linear; for high `C` this is a parameter blowup
  ([PatchTST.py](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py)).
- `RevIN` is applied per-channel at instance level; if features are already
  scaled by an upstream sklearn preprocessor, disable it.
- CLS-token classification heads need positional embedding to cover `N + 1`
  patches
  ([HF docs](https://huggingface.co/docs/transformers/en/model_doc/patchtst)).
- Time-Series-Library reports PatchTST at 69.4% mean accuracy on the UEA
  classification benchmark, ranking second behind TSLANet's 72.7%, so it is
  a credible classification backbone
  ([TSLANet paper](https://arxiv.org/pdf/2404.08472)).

## Recent refinements

- CT-PatchTST (2025) adds explicit channel attention and time attention as
  parallel branches to capture inter-channel relationships the original drops
  ([CT-PatchTST](https://arxiv.org/html/2501.08620v3)).
- DualPathTST (2024) keeps the channel-independent path and adds a
  convolutional cross-variable path with gated fusion
  ([DualPathTST](https://www.sciopen.com/article/10.26599/TST.2024.9010195)).
- PITS (Lee et al., ICLR 2024) drops cross-patch attention entirely and uses
  a patch-wise MLP plus contrastive pretraining; the authors report gains on
  classification as well as forecasting
  ([PITS](https://arxiv.org/pdf/2312.16427)).
- iTransformer is the contrary direction: tokenize variates instead of patches,
  attention computes inter-channel correlations. Hybrid frameworks combining
  the two paradigms are active 2025 research
  ([CT-PatchTST](https://arxiv.org/html/2501.08620v3)).

## Decisions implied for seq-sklearn v2

1. Adopt the HF parameterization: `patch_length`, `patch_stride`, `d_model`,
   `num_hidden_layers`, `num_attention_heads`, `ffn_dim`, `use_cls_token`,
   `pooling_type`, `channel_attention`. These cover every published variant
   without committing to one head shape
   ([HF docs](https://huggingface.co/docs/transformers/en/model_doc/patchtst)).
2. Default head: CLS token plus mean pool fallback, projected to `num_targets`.
   The flatten head is feasible but scales poorly with `C * N` and is awkward
   when `L` is variable. Make it opt-in.
3. Default values: `patch_length=16, patch_stride=8`, matching the official
   supervised configuration
   ([etth1.sh](https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/scripts/PatchTST/etth1.sh)).
   `d_model=128` matches HF and is a reasonable middle ground
   ([HF docs](https://huggingface.co/docs/transformers/en/model_doc/patchtst)).
4. Channel handling: build the input channel stack as
   `[real, embedded_categorical, broadcast_static]` and feed the channel-
   independent encoder. Provide an optional channel-mixing block (channel
   attention) between encoder and head, default off, mirroring HF
   `channel_attention`
   ([HF docs](https://huggingface.co/docs/transformers/en/model_doc/patchtst),
   [CT-PatchTST](https://arxiv.org/html/2501.08620v3)).
5. Mask handling: accept a timestep-level mask, derive a patch-level mask
   (`any` reduction over each patch), apply to attention bias and pooling.
   Do not require global-max padding the way THUML does
   ([exp_classification.py](https://github.com/thuml/Time-Series-Library/blob/main/exp/exp_classification.py)).
6. Regression: reuse the same encoder and CLS/pool readout, project to a
   scalar (or vector for multi-target), MSE loss with optional Huber.
   This matches HF's `PatchTSTForRegression`
   ([HF docs](https://huggingface.co/docs/transformers/en/model_doc/patchtst)).
7. PatchTST is the easier classification port than TST (Zerveas et al.) only
   because of the existing thuml and HF reference heads; TimesNet has a
   classification branch in Time-Series-Library too, but its 2D temporal
   blocks are heavier to port than PatchTST's linear-projection patching
   ([Time-Series-Library](https://github.com/thuml/Time-Series-Library)).
   Recommend PatchTST first, TimesNet second, TST not at all unless a
   masked-pretraining story is wanted.
