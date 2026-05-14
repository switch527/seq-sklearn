# tft-sklearn

A sklearn-compatible Temporal Fusion Transformer for classification and
regression on tabular time-series data.

Built for ensemble use: a sequence-modeling view that is structurally
decorrelated from the tabular-flattened view that XGBoost and tabular
transformers produce, so adding it to a mixed ensemble cancels noise
and finds signal the other models miss.

## Status

Pre-implementation. Requirements doc lives at
[`docs/requirements.md`](docs/requirements.md). Architecture doc, code,
and tests follow.

## Quick mental model

- Input is a tabular panel: one row per entity (customer, account,
  device) per period (day, week, month, quarter, year). Same shape your
  existing classifier already consumes.
- The library restructures it into masked sequences and feeds them
  through the TFT backbone.
- Two estimators sit on top of the shared backbone: `TFTClassifier`
  (binary or multi-class) and `TFTRegressor` (point or quantile).
- Both implement the sklearn estimator contract, so they compose into
  `Pipeline`, `GridSearchCV`, `cross_val_score`, and Optuna search.

## Repository layout

```
docs/                requirements, architecture, examples
src/tft_sklearn/     library code (planned)
tests/               unit, integration, e2e, deploy (planned)
.meta/               agent configuration (separate repo, mounted via symlinks)
```

Agent configuration (CLAUDE.md, GEMINI.md, the `.claude/` and `.gemini/`
directories) is kept in the separate
[`tft-sklearn-meta`](https://github.com/switch527/tft-sklearn-meta) repo,
mounted at `.meta/` here via `bash .meta/bootstrap.sh`. See
[`.meta/README.md`](.meta/README.md) for the rationale.
