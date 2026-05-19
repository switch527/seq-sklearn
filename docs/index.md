# seq-sklearn

**Modern deep sequence models, as easy to use as scikit-learn.**
One `fit` / `predict` API for classification and regression on
multivariate time-series panels, across the transformer and
recurrent model families, with built-in interpretability surfaces
and a first-class Optuna integration.

## Why seq-sklearn

The sklearn ecosystem stops at shallow tabular models. The deep
time-series libraries (`pytorch-forecasting`, `darts`,
`neuralforecast`, `sktime`) are built for **forecasting**, a
different task; using them for ordinary supervised classification
and regression means hundreds of lines of dataloader and trainer
wiring per project. seq-sklearn fills that gap behind the standard
estimator contract.

**v1 ships a Temporal Fusion Transformer** (Lim et al., 2021)
adapted to supervised tasks: classifier + regressor (point and
quantile), the full `fit`/`predict`/`predict_proba` contract,
sklearn pipeline-compatibility, calibrated probabilities, conformal
quantile regression, ONNX export, and the variable-selection and
temporal-attention surfaces as a typed dataclass.

```{toctree}
:hidden:
:caption: Get started

tutorial/index
about/installation
```

```{toctree}
:hidden:
:caption: How-to guides

how-to/index
```

```{toctree}
:hidden:
:caption: Reference

reference/index
examples_gallery/index
```

```{toctree}
:hidden:
:caption: Explanation

explanation/index
```

```{toctree}
:hidden:
:caption: About

about/index
```

```{toctree}
:hidden:
:caption: Design (internal)

design/index
```

## Quickstart

The example below is the literal content of
`examples/quickstart.py`, which the N1 quickstart-in-CI test
imports. It is the single executable source; the docs site embeds
it via `literalinclude` so the snippet and the test never drift.

```{literalinclude} ../examples/quickstart.py
:language: python
:linenos:
```

The full guided walkthrough is in the
[tutorial](tutorial/first_classifier). If you have a specific
question ("how do I export to ONNX", "how do I handle imbalanced
classes", "how do I avoid temporal leakage"), the
[how-to guides](how-to/index) are the place to start. For the
exact signatures and types, the [API reference](reference/api)
and the [config tables](reference/config) are auto-generated from
the source. For the design rationale (why a TFT, how attention
works, the determinism model), the
[explanation pages](explanation/index) are the "why".

## What ships in v1

- **`TFTClassifier`** and **`TFTRegressor`** (point + quantile),
  with the full sklearn contract.
- One shared **`TabularToSequence`** preprocessing path.
- **`EntityTimeSeriesSplit`**: per-entity time-expanding-window CV
  that doesn't leak future periods across folds.
- **Calibrated probabilities** (temperature/isotonic) and
  **conformal quantile regression**.
- **`predict_with_attention`** returning variable-selection and
  temporal-attention weights as a typed dataclass.
- **ONNX export** via the optional `[onnx]` extra (round-trip
  parity tested in CI).
- **First-class Optuna integration** with in-training pruning.
- **Strict determinism mode** for reproducibility (contract on the
  [determinism reference](reference/determinism)).

## Stability

v1.0.0 is the first stable release. The public API surface is
defined by `architecture A3` and gated by
`tests/unit/test_public_api_surface.py`. Breaking changes follow
the [versioning policy](about/versioning) (SemVer, ≥1 minor
deprecation window). The roadmap (PatchTST/TimesNet/TST in v2,
LSTM/GRU/LSTM-FCN in v3) is in the
[requirements doc](design/index).
