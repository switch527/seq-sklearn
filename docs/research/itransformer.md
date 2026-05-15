# iTransformer (Liu et al., 2024). Research brief

Status in seq-sklearn: **experimental, deferred**. Not on the v1/v2/v3 roadmap.
This brief documents the deferral rationale and the watch-list trigger.

## Source citations

1. Liu, Hu, Zhang, Wu, Wang, Ma, Long. *iTransformer: Inverted Transformers Are
   Effective for Time Series Forecasting*. arXiv:2310.06625. ICLR 2024 Spotlight.
   <https://arxiv.org/abs/2310.06625>
2. Reference implementation: <https://github.com/thuml/iTransformer> (Tsinghua
   THUML, ~2.1k stars as of early 2026, forecasting scripts only).
3. Wang, Wang, Song, Wang. *AutoLDT: a lightweight spatio-temporal decoupling
   transformer framework with AutoML method for time series classification*.
   Scientific Reports 14, 2024. DOI:10.1038/s41598-024-81000-1.
4. Li, Yang. *Enhanced Inverted Transformer: Advancing Variate Token Encoding
   and Blending for Time Series Forecasting*. Applied Intelligence, 2025.
   <https://link.springer.com/article/10.1007/s10489-025-06886-4>

## Architecture

iTransformer keeps every standard Transformer component (self-attention, LayerNorm,
FFN, residuals) and inverts only the tokenization axis. Each variate's full lookback
series is embedded into one *variate token*; self-attention then operates over the
V variates rather than the T timesteps, modeling cross-variate correlations, and the
position-wise FFN is applied per variate token to learn temporal nonlinear
representations across the lookback. A linear projection maps each refined variate
token to its forecast horizon. The authors frame this as a fix for "meaningless
attention maps" produced when standard time-axis attention fuses heterogeneous
variates at each timestamp. The whole design is forecasting-shaped: tokens are
variates, the output head produces a per-variate horizon vector.

## Why classification adaptation is non-trivial

Three architectural facts make the inversion awkward for sequence classification on
panel data.

1. *Sequence-length signal lives in the wrong axis.* In UCR/UEA-style TSC and in
   panel-data classification, the discriminative pattern is typically a temporal
   shape inside one or a few variates (rising slope, regime shift, motif).
   iTransformer's attention never compares timesteps; it compares whole-series
   variate embeddings. Temporal pattern learning is left entirely to the FFN, which
   has no inductive bias for ordering, locality, or multi-scale structure.
2. *No natural pooling target.* Variate tokens are not exchangeable with each other
   the way timestep tokens or patches are. Mean-pooling across V variates throws
   away which variate carried the signal, and a `[CLS]` variate would mix
   semantically distinct measurement channels. The reference implementation ships
   no classification head, no pooling code, and no `[CLS]` token (verified against
   the thuml repo).
3. *Variable-length sequences are problematic.* The variate-token embedding is a
   linear map from a fixed lookback L to model dimension d. Panel data with
   variable per-entity sequence lengths needs padding or windowing, both of which
   contaminate the variate token with mask noise that attention cannot route
   around (it sees variates, not timesteps).

Follow-on work has already started patching the temporal side. Li and Yang's
*Enhanced Inverted Transformer* (2025) argues iTransformer's variate tokens fail to
encode fine-grained temporal structure and adds a temporal-blending stage; this
implicitly concedes the discriminative-signal-on-the-wrong-axis problem, even
within forecasting.

## 2024-2025 evidence for or against classification feasibility

One serious data point exists. Wang et al.'s AutoLDT paper (Scientific Reports
2024) benchmarks iTransformer as a classification baseline on a subset of UCR/UEA
datasets alongside Transformer, Informer, Pyraformer, Crossformer, TimesNet, and
MiniRocket. They report iTransformer winning on 4 datasets with 80.58% average
accuracy, finishing second to their proposed AutoLDT. The paper does not detail
the classification-head adaptation (pooling strategy, output projection) and the
dataset subset is small, so this is suggestive rather than conclusive. No
follow-on study has reproduced iTransformer-for-classification at full UCR/UEA
scale, no preprint compares it head-to-head with TST or PatchTST on TSC, and the
thuml repo still ships forecasting-only code.

For regression on supervised panel data, no paper, repo, or blog post surfaced.
The variant ecosystem (FE-iTransformer for traffic flow, MM-iTransformer for
multimodal economic series) stays inside forecasting.

## Watch-list trigger for revisiting

Revisit iTransformer for seq-sklearn when any one of the following lands.

1. A peer-reviewed or arXiv paper publishes iTransformer-for-classification
   benchmarks on the full UCR archive or the UEA multivariate archive, with
   accuracy competitive with TST or PatchTST (within ~2 points of the
   transformer-family leader on the published averages).
2. The thuml reference repo merges a `classification/` task directory with a
   documented head and pooling choice, mirroring the existing
   `forecasting/` and `imputation/` task scaffolds.
3. A regression-on-panel-data evaluation appears that shows the inversion
   beating a per-variate-temporal baseline on panel regression with R^2 or RMSE
   gains outside noise.

Until then iTransformer stays in the "experimental, future exploration" tier:
worth tracking because the AutoLDT result is non-trivial, not worth implementing
because the architecture's prior is forecasting-cross-variate, not
classification-temporal-pattern.
