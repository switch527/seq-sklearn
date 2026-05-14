# seq-sklearn

A sklearn-compatible Python library for **sequence learning on tabular
time-series data**, covering modern deep models across three families:
recurrent (LSTM, GRU, LSTM-FCN), transformer (TFT, PatchTST,
iTransformer, TimesNet, TST), and foundation (Chronos, MOMENT, TimesFM).

For **standard supervised tasks** (classification, regression), not
forecasting. The gap this library fills sits at the intersection of
the sklearn ecosystem and modern deep sequence models for supervised
tabular panel work.

## Status

Pre-implementation. The full v1 requirements doc lives at
[`docs/requirements.md`](docs/requirements.md). Architecture doc, code,
and tests follow.

## Roadmap

| Version | Scope |
|---|---|
| **v1** | TFT classifier + regressor; full `TabularToSequence` pipeline; Optuna; all library-wide infrastructure |
| v1.1 | Multi-output regression, multi-label classification |
| v2 | PatchTST, iTransformer, TimesNet, TST (transformer family completion) |
| v3 | LSTM, GRU, LSTM-FCN (recurrent family) |
| v4 | Chronos, MOMENT, TimesFM (foundation family with adapter heads) |

## Quick mental model

- Input is a tabular panel: one row per entity (customer, account,
  device) per period (day, week, month, quarter, year). Same shape your
  existing classifiers already consume.
- The library restructures it into masked sequences and feeds them
  through the model backbone.
- Every model contributes a classifier and a regressor variant
  (`<Model>Classifier`, `<Model>Regressor`). v1 ships `TFTClassifier`
  and `TFTRegressor`; subsequent versions add nine more across the
  three families.
- Every estimator implements the sklearn contract, so they compose
  into `Pipeline`, `GridSearchCV`, `cross_val_score`, and Optuna search.

## Repository layout

```
docs/                requirements, architecture, examples
src/seq_sklearn/     library code (planned)
tests/               unit, integration, e2e, deploy (planned)
.meta/               agent configuration (separate repo, mounted via symlinks)
```

Agent configuration (CLAUDE.md, GEMINI.md, the `.claude/` and `.gemini/`
directories) is kept in the separate
[`seq-sklearn-meta`](https://github.com/switch527/seq-sklearn-meta) repo,
mounted at `.meta/` here via `bash .meta/bootstrap.sh`. See
[`.meta/README.md`](.meta/README.md) for the rationale.
