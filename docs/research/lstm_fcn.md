# LSTM-FCN research notes (seq-sklearn v3)

## Source citations

- Original paper: Karim, Majumdar, Darabi, Chen, "LSTM Fully Convolutional Networks for Time Series Classification", IEEE Access 2018. arXiv 1709.05206 v1 submitted 8 Sep 2017. https://arxiv.org/abs/1709.05206
- Follow-up: Karim, Majumdar, Darabi, Harford, "Multivariate LSTM-FCNs for Time Series Classification", Neural Networks 116 (2019) 237-245. arXiv 1801.04503 v1 14 Jan 2018, v2 1 Jul 2019. https://arxiv.org/abs/1801.04503
- Original Keras code (univariate, ALSTM-FCN, attention LSTM): https://github.com/houshd/LSTM-FCN
- Original Keras code (multivariate, MLSTM-FCN with SE): https://github.com/titu1994/MLSTM-FCN
- tsai PyTorch port (LSTM_FCN, MLSTM_FCN, GRU_FCN, RNN_FCN): https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py
- tsai SqueezeExciteBlock source: https://github.com/timeseriesAI/tsai/blob/main/tsai/models/layers.py
- aeon classifier listing (FCN, InceptionTime, ResNet, LITE; no LSTM-FCN): https://www.aeon-toolkit.org/en/latest/examples/classification/deep_learning.html
- Foumani et al., "Deep Learning for Time Series Classification and Extrinsic Regression: A Current Survey", arXiv 2302.02515, ACM Computing Surveys 2024. https://arxiv.org/html/2302.02515v2
- MONSTER scalable TSC benchmark, arXiv 2502.15122 (2025). https://arxiv.org/html/2502.15122

## Architecture (FCN branch + LSTM branch with dimension shuffle)

Two parallel branches that consume the same `(B, C, T)` tensor and concatenate before a linear head. Verified from the original Keras file
[`all_datasets_training.py`](https://github.com/houshd/LSTM-FCN/blob/master/all_datasets_training.py).

**FCN branch** (channels-first, three Conv1D blocks):

| block | filters | kernel | activation | norm |
|-------|---------|--------|------------|------|
| 1 | 128 | 8 | ReLU | BatchNorm |
| 2 | 256 | 5 | ReLU | BatchNorm |
| 3 | 128 | 3 | ReLU | BatchNorm |

Each block is `Conv1D(filters, kernel, padding="same", he_uniform) -> BatchNorm -> ReLU`. After block 3, a `GlobalAveragePooling1D` collapses time, yielding a 128-d vector. The Keras source is unambiguous on filter counts `128 -> 256 -> 128` and kernels `8, 5, 3` ([houshd/LSTM-FCN](https://github.com/houshd/LSTM-FCN/blob/master/all_datasets_training.py)). tsai keeps the same `[128, 256, 128]` channels but defaults kernels to `[7, 5, 3]` ([tsai RNN_FCN.py](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py)); the `kss=[7,5,3]` deviation is a tsai choice, not the paper. We follow the paper: `[8, 5, 3]`.

**LSTM branch** with dimension shuffle. The Keras input shape is `(1, MAX_SEQUENCE_LENGTH)`, i.e. one channel and many timesteps. The LSTM consumes the input WITHOUT permutation, so Keras processes the input as a single time step with `MAX_SEQUENCE_LENGTH` features. The FCN branch sees the same tensor `Permute((2, 1))`-d to `(MAX_SEQUENCE_LENGTH, 1)`. This is the "dimension shuffle": the LSTM and FCN see transposed views of the same series. The LSTM runs over the channel axis treated as time (univariate case: a single step), the FCN convolves along the actual time axis ([houshd/LSTM-FCN](https://github.com/houshd/LSTM-FCN/blob/master/all_datasets_training.py)). The paper motivates this as a regularizer that reduces the LSTM workload to one step in the univariate setting, cutting parameters by orders of magnitude (Karim 2017, §III, [arXiv 1709.05206](https://arxiv.org/abs/1709.05206)).

LSTM size defaults to `NUM_CELLS=8`; the paper sweeps `{8, 64, 128}`. After the LSTM, `Dropout(0.8)`. tsai exposes `hidden_size=100, rnn_dropout=0.8` ([tsai RNN_FCN.py](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py)).

**Combine**: `concatenate([lstm_out, fcn_gap])` then `Dense(num_classes, activation="softmax")`. Total feature dim into the head is `lstm_hidden + 128`.

## Classification head verified from reference impl

Keras original: `Dense(NB_CLASS, activation="softmax")` on the concatenated vector, trained with categorical cross-entropy ([houshd/LSTM-FCN](https://github.com/houshd/LSTM-FCN/blob/master/all_datasets_training.py)). tsai's `_RNN_FCN_Base.forward` does `self.concat([last_out, x])` then a single `nn.Linear(hidden + conv_layers[-1], c_out)` ([tsai RNN_FCN.py](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py)). No hidden FC layer between concat and output. `fc_dropout=0.` by default.

For seq-sklearn the head reduces to `Linear(hidden + 128, n_classes)` with `nn.CrossEntropyLoss` (logits, no softmax in the module).

## Regression adaptation feasibility

No published LSTM-FCN regression variant exists. The 2024 ACM survey ([arXiv 2302.02515](https://arxiv.org/html/2302.02515v2)) discusses MLSTM-FCN only under classification and adapts ResNet, FCN, and Inception (not LSTM-FCN) to TSER. The Monash TSER archive omits LSTM-FCN. No tsai example, no aeon classifier, no Keras fork ships a regression head.

Mechanically the adaptation is trivial: swap `Linear -> n_classes` for `Linear -> n_targets` and switch the loss to MSE/Huber. Both branches are output-agnostic. The dimension shuffle was justified empirically on UCR classification accuracy; its regression value is unvalidated. seq-sklearn v3 ships regression as an unvalidated extension and documents the lack of evidence.

## Mask handling under dimension shuffle

This is the hard part and the reason the model is awkward for variable-length panels.

In the original paper the input is fixed-length `(1, T)` and there is no mask. In the univariate setting the LSTM consumes a single timestep of `T` features, so a time-axis mask is undefined for that branch: padding tokens in `T` become input features to the LSTM, not steps to skip. The FCN branch convolves along `T` and could in principle mask, but the Keras code does not.

In the multivariate setting ([arXiv 1801.04503](https://arxiv.org/abs/1801.04503)) the dimension shuffle sends `(T, C)` to the LSTM, so the LSTM runs over true time with `C` features per step and `pack_padded_sequence` works. The FCN sees `(C, T)` and needs its conv outputs masked before GAP, otherwise pad positions contaminate the mean.

Reference implementations punt: tsai's `_RNN_FCN_Base` has no `lengths` argument, no `pack_padded_sequence`, no masked GAP, and assumes fixed-length input ([tsai RNN_FCN.py](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py)). The original Keras code pads to `MAX_SEQUENCE_LENGTH` at load time without telling the model ([houshd/LSTM-FCN](https://github.com/houshd/LSTM-FCN/blob/master/all_datasets_training.py)). aeon does not ship LSTM-FCN at all ([aeon classifier list](https://www.aeon-toolkit.org/en/latest/examples/classification/deep_learning.html)).

For seq-sklearn v3: take `lengths` on `forward`, build a boolean mask, apply masked GAP on the FCN branch (`(x * mask).sum / lengths`), and use `pack_padded_sequence` only on the multivariate LSTM path. In the univariate path the LSTM has no time axis to mask, so we feed padded `(1, T)` directly and document the asymmetry.

## Squeeze-and-Excite variant

The follow-up paper ([arXiv 1801.04503](https://arxiv.org/abs/1801.04503)) inserts SE blocks between conv blocks 1-2 and 2-3 to recalibrate channels. The tsai port codes it as ([tsai layers.py](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/layers.py)):

```
SqueezeExciteBlock(ni, reduction=16):
  GAP1d -> Linear(ni, ni//reduction) -> ReLU
        -> Linear(ni//reduction, ni) -> Sigmoid
  output = x * y.unsqueeze(2).expand_as(x)
```

Default reduction ratio 16. The block multiplies each conv channel by a learned scalar in `[0, 1]` derived from the global pooled feature. The MLSTM-FCN paper reports the SE block as the primary accuracy lift over MLSTM-FCN without SE on the multivariate UEA archive. tsai turns SE on for the `M*` variants (`se=16`) and off (`se=0`) for the plain ones ([tsai RNN_FCN.py](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py)).

seq-sklearn v3 exposes `se_reduction: int | None`. `None` means no SE; integer means reduction ratio. Default `None` for univariate (paper baseline) and `16` for multivariate (matches MLSTM-FCN).

## Modern relevance and gotchas

LSTM-FCN is no longer SOTA on either archive. The 2024 ACM survey ([arXiv 2302.02515](https://arxiv.org/html/2302.02515v2)) calls out ConvTran as multivariate SOTA in 2023 and InceptionTime as the headline deep classifier, with HIVE-COTE 2.0 still leading non-deep ensembles. The 2025 MONSTER benchmark ([arXiv 2502.15122](https://arxiv.org/html/2502.15122)) ranks ROCKET-family methods (excluded from seq-sklearn) and ConvTran above LSTM-FCN at scale. LSTM-FCN survives in library catalogs (tsai, sktime forks) as a small, fast baseline rather than a frontier model.

Why we still ship it: small parameter count (sub-100k for the univariate `NUM_CELLS=8` config), trains in minutes on CPU on small UCR datasets, and the GAP-based feature vector gives a clean interpretability surface for activation maps. It is the right "deep but cheap" baseline next to plain LSTM/GRU.

PyTorch implementation gotchas (2024-2025):

1. BatchNorm goes BEFORE ReLU, matching the paper and tsai. Inverting these (a common copy-paste error) destroys the residual-free training stability the paper relied on.
2. Dropout on the LSTM is applied to the LAST hidden state, not to inter-step states. The Keras `Dropout(0.8)` sits on the LSTM output tensor. In PyTorch `nn.LSTM(..., dropout=p)` is between stacked layers, not on the output; we need an explicit `nn.Dropout` after `lstm_out[:, -1]`.
3. The dimension shuffle is easy to forget. With `(B, C, T)` input, the FCN branch is happy; the LSTM branch wants `(B, T, C)` for the multivariate case or `(B, 1, T)` (read as one step of `T` features) for the univariate case. Wire it wrong and the LSTM silently treats channels as time or vice versa with no shape error.
4. `padding="same"` on Conv1D with even kernel size 8 is asymmetric in PyTorch. Either keep kernel 8 and pad manually (`F.pad(x, (3, 4))`) or switch to kernel 7 like tsai does. The paper's kernel-8 block was Keras `padding="same"` which left-pads 3, right-pads 4 internally; replicate that.
5. `GlobalAveragePooling1D` is `x.mean(dim=2)` over `(B, C, T)`. Under a mask, it becomes `(x * mask).sum(dim=2) / lengths`.

## Decisions implied for seq-sklearn v3

- Module: `seq_sklearn/models/lstm_fcn.py` with `LSTMFCNClassifier` and `LSTMFCNRegressor`, both wrapping a shared `LSTMFCNBackbone(pydantic Config)`.
- Defaults locked to the paper: conv channels `(128, 256, 128)`, kernels `(8, 5, 3)`, LSTM `hidden_size=8` (paper univariate baseline), `rnn_dropout=0.8`, BatchNorm before ReLU.
- `se_reduction: int | None = None` for univariate, `16` exposed for the multivariate variant. SE block reused from a shared `layers.py` (matches PatchTST/TimesNet research notes' pattern of shared building blocks).
- Mask handling: FCN branch uses masked GAP. Multivariate LSTM branch uses `pack_padded_sequence`. Univariate LSTM branch consumes padded `(B, 1, T)` directly, with a docstring caveat that pad positions become input features by paper design.
- Head: `nn.Linear(hidden + 128, n_outputs)`. `CrossEntropyLoss` for classification, `MSELoss`/`HuberLoss` for regression (selected via the regressor wrapper).
- Regression is marked experimental in the README and the docstring. No published baseline to compare against; v3 ships smoke tests on Monash TSER but does not claim accuracy parity with ResNet/Inception regressors.
- No SOTA claim. LSTM-FCN is documented as a small, interpretable baseline alongside InceptionTime (v3) and ConvTran-class transformers (v2 PatchTST).
