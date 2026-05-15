# seq-sklearn

A sklearn-compatible Python library for **sequence learning on tabular
time-series data**, covering modern deep models across two families:
transformer (TFT, PatchTST, TimesNet, TST) and recurrent (LSTM, GRU,
LSTM-FCN).

For **standard supervised tasks** (classification, regression), not
forecasting. The gap this library fills sits at the intersection of
the sklearn ecosystem and modern deep sequence models for supervised
tabular panel work. v1's TFT classifier and regressor are a genuine
architectural adaptation of the original (forecasting-only) paper,
which is the project's core contribution at first ship.

## Status

Pre-implementation. The full v1 requirements doc lives at
[`docs/requirements.md`](docs/requirements.md). Architecture doc, code,
and tests follow.

## Roadmap

| Version | Scope |
|---|---|
| **v1** | TFT classifier + regressor; full `TabularToSequence` pipeline; Optuna; all library-wide infrastructure |
| v1.1 | Multi-output regression, multi-label classification |
| v2 | PatchTST, TimesNet, TST (transformer family completion) |
| v3 | LSTM, GRU, LSTM-FCN (recurrent family) |

iTransformer is tracked as experimental future exploration (no
classification or regression evaluation exists in the literature).
Foundation models (Chronos, MOMENT, TimesFM) are out of scope: only
MOMENT has classification support, and a one-model family does not
justify the abstraction work.

## Quick mental model

- Input is a tabular panel: one row per entity (customer, account,
  device) per period (day, week, month, quarter, year). Same shape your
  existing classifiers already consume.
- The library restructures it into masked sequences and feeds them
  through the model backbone.
- Every model contributes a classifier and a regressor variant
  (`<Model>Classifier`, `<Model>Regressor`). v1 ships `TFTClassifier`
  and `TFTRegressor`; subsequent versions add six more across the
  two families.
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
