# seq-sklearn

Modern deep sequence models, as easy to use as scikit-learn: one
`fit` / `predict` API for classification and regression on multivariate
time series, across the transformer and recurrent model families.

The sklearn ecosystem stops at shallow tabular models; the deep
time-series libraries are built for forecasting, a different task. Using
modern deep sequence models for ordinary supervised classification and
regression means hand-rolling adapters and trainer wiring. seq-sklearn
fills that gap behind the standard estimator contract, with one shared
preprocessing, calibration, and tuning path across every model it ships.
v1's first model is a Temporal Fusion Transformer (Lim et al., 2021)
adapted from forecasting to supervised tasks; more transformer and
recurrent models follow behind the identical API.

```{warning}
Pre-implementation. The phase-1 foundation is landing now. The API
shown in the docs is the target v1 surface and is not yet released.
See [the requirements doc](https://github.com/switch527/seq-sklearn/blob/main/docs/requirements.md)
for the full v1 specification.
```

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

user_guide/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
changelog
contributing
```
