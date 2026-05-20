<div align="center">

# seq-sklearn

**Modern deep sequence models, as easy to use as scikit-learn. One
`fit` / `predict` API for classification and regression on multivariate
time series, across the transformer and recurrent model families.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](docs/requirements.md)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://docs.astral.sh/ruff/)

</div>

Sequence learning for ordinary classification and regression is harder
than it should be. The sklearn ecosystem covers tabular tasks broadly
but stops at shallow models. The deep-learning libraries that handle
multivariate time series (pytorch-forecasting, darts, neuralforecast,
sktime) are built for **forecasting**, a structurally different task,
and using them off-label for supervised targets means impedance
mismatch and hundreds of lines of dataloader and trainer wiring.

seq-sklearn fills that gap. It brings modern deep sequence models behind
the familiar `fit` / `predict` / `predict_proba` estimator contract, so
they drop into `Pipeline`, `GridSearchCV`, `cross_val_score`, and Optuna
unchanged. One `TabularToSequence` preprocessing path, one calibration
story, and one tuning integration are shared across **every** model the
library ships, spanning the transformer family (TFT, then PatchTST,
TimesNet, TST) and the recurrent family (LSTM, GRU, LSTM-FCN), with room
to extend further. You pick the model; the API and the workflow stay the
same.

## The gap this fills

```
            shallow tabular models          deep sequence models
          +-----------------------+    +---------------------------+
supervised|  scikit-learn, XGBoost|    |       (the gap)           |
          |  LightGBM, CatBoost   |    |       seq-sklearn         |
          +-----------------------+    +---------------------------+
forecasting|         n/a          |    | pytorch-forecasting,      |
          |                       |    | darts, neuralforecast     |
          +-----------------------+    +---------------------------+
```

Nothing sits at the intersection of **modern deep sequence models** and
**standard supervised tasks under the sklearn contract** on tabular
panel input. That is the cell seq-sklearn occupies.

The driving use case is customer-churn prediction on a payments panel.
The same panel shape works for any entity-by-period problem: customers
by month, patients by visit, devices by day, sensors by hour. The model
never learns what an entity is; it sees one sequence per entity.

## Status

Pre-implementation, actively built. The phase-1 foundation (configs,
serialization, data pipeline, model blocks) is landing now. The
library-wide infrastructure (the sklearn API, `TabularToSequence`,
training, calibration, Optuna) is built once and reused by every model.

**TFT is the first model, not the only one.** v1 ships `TFTClassifier`
and `TFTRegressor` as a genuine architectural adaptation of the
forecasting-only Temporal Fusion Transformer (head and loss swapped, not
a thin wrapper). PatchTST, TimesNet, TST, LSTM, GRU, and LSTM-FCN follow
in later versions behind the identical API. See the roadmap below.

The full v1 requirements doc is
[`docs/requirements.md`](docs/requirements.md); the README and
documentation strategy is
[`docs/readme_and_docs_plan.md`](docs/readme_and_docs_plan.md).

Star or watch the repo to follow the v1 release.

## Quickstart

A complete, runnable binary classifier on a synthetic panel. This
is the legible shape of `examples/quickstart.py` and runs end to
end on CPU; the executable file is what `tests/e2e/test_quickstart.py`
imports and gates in CI, so this snippet cannot rot.

```python
from sklearn.metrics import accuracy_score

from seq_sklearn import SchedulerParams, TabularConfigParams, TFTClassifier
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator

# 1. A small synthetic panel (real data slots in here unchanged).
gen = SyntheticPanelGenerator(
    target_kind="binary",
    num_entities=160,
    periods_per_entity=24,
    signal_strength=0.9,
    lookback=12,
    seed=42,
)
panel, y = gen.generate(seed=42)

# 2. Declare the panel schema once. The model never guesses columns.
tabular_config = TabularConfigParams(
    id_col=gen.id_col,
    time_col=gen.time_col,
    static_categorical_cols=tuple(gen.static_categorical_cols),
    static_real_cols=tuple(gen.static_real_cols),
    time_varying_real_cols=tuple(gen.time_varying_real_cols),
    time_varying_categorical_cols=tuple(gen.time_varying_categorical_cols),
    lookback=gen.lookback,
    min_periods=1,
    min_periods_predict=1,
    max_categorical_cardinality=10_000,
)

# 3. The classifier is a regular sklearn estimator.
clf = TFTClassifier(
    task_type="binary",
    tabular_config=tabular_config,
    scheduler=SchedulerParams(name="cosine_with_warmup", warmup_steps=50),
    hidden_size=64,
    attention_heads=4,
    max_epochs=60,
    batch_size=64,
    val_fraction=0.2,
    cal_fraction=0.0,
    precision="32-true",
    verbose=False,
    seed=42,
)
clf.fit(panel, y)
print(f"accuracy: {accuracy_score(y, clf.predict(panel)):.3f}")
```

Every estimator implements the sklearn contract, so it composes into
`Pipeline`, `GridSearchCV`, `cross_val_score`, and the Optuna
integration unchanged.

Where a model has introspectable internals, they come back as a typed
output, not a separate analysis step. TFT exposes variable selection
and attention:

```python
out = clf.predict_with_attention(panel)      # AttentionOutput dataclass
out.var_selection_weights                    # which features mattered, per step
out.attention_weights                        # which timesteps mattered
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
- TFT's interpretable variable-selection and temporal-attention outputs
  (later models expose their own introspection through the same shape).
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
