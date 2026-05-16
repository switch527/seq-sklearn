# seq-sklearn

A scikit-learn compatible Temporal Fusion Transformer for classification
and regression on multivariate time series, with interpretable variable
selection and attention built in.

The published TFT (Lim et al., 2021) and every mature implementation
target multi-horizon forecasting. seq-sklearn adapts it to ordinary
supervised classification and regression on tabular panel data, behind
the standard `fit` / `predict` estimator contract, with the model's
variable-selection and attention surfaces kept as first-class outputs.

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
