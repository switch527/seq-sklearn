<div align="center">

# seq-sklearn

**A scikit-learn compatible Temporal Fusion Transformer for classification
and regression on multivariate time series, with interpretable variable
selection and attention built in.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](docs/requirements.md)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://docs.astral.sh/ruff/)

</div>

The Temporal Fusion Transformer (Lim et al., 2021) is a strong,
interpretable sequence model, but the published architecture and every
mature implementation target multi-horizon **forecasting**. Using it for
ordinary supervised **classification or regression** today means
hand-rolling a head swap and dozens of lines of dataloader and trainer
wiring, or bending a forecasting library off-label. seq-sklearn closes
that gap: a TFT classifier and regressor on tabular panel data, behind
the standard `fit` / `predict` estimator contract, with the model's
variable-selection and attention surfaces preserved as first-class
outputs.

## The gap this fills

The sklearn ecosystem covers tabular classification and regression
broadly. The forecasting world has solid deep-learning coverage
(pytorch-forecasting, darts, neuralforecast, sktime). Nothing sits at
the intersection: **modern deep sequence models used for standard
supervised tasks, wrapped in the sklearn estimator contract, on tabular
panel input**. seq-sklearn is built for that intersection, with a shared
preprocessing pipeline, calibration story, and Optuna integration across
every model family it will ship.

The driving use case is customer-churn prediction on a payments panel.
The same panel shape works for any entity-by-period problem: customers
by month, patients by visit, devices by day, sensors by hour.

## Status

Pre-implementation, actively built. The phase-1 foundation (configs,
serialization, data pipeline, model blocks) is landing now. The full v1
requirements doc is [`docs/requirements.md`](docs/requirements.md); the
README and documentation strategy is
[`docs/readme_and_docs_plan.md`](docs/readme_and_docs_plan.md).

Star or watch the repo to follow the v1 release.

## Planned API (not yet released)

This is the target v1 surface. It is shown so the design is legible
before code-complete; the import will not work until v1 ships.

```python
from seq_sklearn import TFTClassifier
from sklearn.metrics import roc_auc_score

clf = TFTClassifier(lookback=12, hidden_size=128)
clf.fit(X_train, y_train)                    # X: tidy entity-by-period DataFrame
proba = clf.predict_proba(X_test)
print(f"AUC: {roc_auc_score(y_test, proba[:, 1]):.3f}")
```

Every estimator implements the sklearn contract, so it composes into
`Pipeline`, `GridSearchCV`, `cross_val_score`, and Optuna search
unchanged:

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([("clf", TFTClassifier(lookback=12))]).fit(X_train, y_train)
```

Interpretability is a returned output, not an afterthought:

```python
out = clf.predict_with_attention(X_test)     # frozen dataclass
out.variable_selection_weights               # which features mattered, per step
out.temporal_attention                       # which timesteps mattered
```

## What ships in v1

- `TFTClassifier` and `TFTRegressor`, a genuine architectural adaptation
  of the forecasting-only TFT (head and loss swapped, not a wrapper).
- Full sklearn estimator contract: `fit` / `predict` / `predict_proba` /
  `score` / `get_params` / `set_params`, `Pipeline` and `GridSearchCV`
  compatible.
- One `TabularToSequence` preprocessing path shared by every future
  model.
- pytorch-lightning training backend, single CPU or single GPU,
  automatic.
- pydantic-typed configuration, no hidden hyperparameter defaults in
  model code.
- Calibrated class probabilities; conformal quantile regression.
- Interpretable variable-selection and temporal-attention outputs.
- safetensors + JSON serialization; ONNX export via the optional
  `seq-sklearn[onnx]` extra.
- Optuna as a first-class tuning integration, including in-training
  pruning.

## Roadmap

| Version | Scope |
|---|---|
| **v1** | TFT classifier + regressor; full `TabularToSequence` pipeline; Optuna; all library-wide infrastructure |
| v1.1 | Multi-output regression, multi-label classification |
| v2 | PatchTST, TimesNet, TST (transformer family completion) |
| v3 | LSTM, GRU, LSTM-FCN (recurrent family) |

iTransformer is tracked as experimental. Foundation models (Chronos,
MOMENT, TimesFM) are out of scope: only MOMENT supports classification,
and a one-model family does not justify the abstraction work.

## Citation

A citable release (DOI and a JOSS paper) lands with v1. Until then, cite
the original architecture:

```bibtex
@article{lim2021tft,
  title   = {Temporal Fusion Transformers for interpretable
             multi-horizon time series forecasting},
  author  = {Lim, Bryan and Ar{\i}k, Sercan {\"O} and Loeff, Nicolas
             and Pfister, Tomas},
  journal = {International Journal of Forecasting},
  volume  = {37},
  number  = {4},
  pages   = {1748--1764},
  year    = {2021}
}
```

## Contributing

Contribution guidelines are in [`CONTRIBUTING.md`](CONTRIBUTING.md);
expected behavior is in [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md);
security reports go through [`SECURITY.md`](SECURITY.md).

Agent configuration (CLAUDE.md, GEMINI.md, the `.claude/` and `.gemini/`
directories) lives in the separate
[`seq-sklearn-meta`](https://github.com/switch527/seq-sklearn-meta) repo,
mounted at `.meta/` via `bash .meta/bootstrap.sh`. See
[`.meta/README.md`](.meta/README.md) for the rationale.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
