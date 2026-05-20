# API reference

The estimator and helper symbols re-exported under `seq_sklearn`
(architecture A3). Anything not in this list, or under an
underscore-prefixed module, is INTERNAL and not covered by the
SemVer stability guarantee.

## Estimators

```{eval-rst}
.. autoclass:: seq_sklearn.TFTClassifier
   :members: fit, predict, predict_proba, predict_with_attention,
             score, get_params, set_params, save, load, export_onnx
   :inherited-members:
   :show-inheritance:

.. autoclass:: seq_sklearn.TFTRegressor
   :members: fit, predict, predict_quantiles, predict_with_attention,
             score, get_params, set_params, save, load, export_onnx
   :inherited-members:
   :show-inheritance:
```

## Preprocessing

```{eval-rst}
.. autoclass:: seq_sklearn.TabularToSequence
   :members: fit, transform, inverse_transform, fit_transform
   :inherited-members:
   :show-inheritance:
```

## Configuration adapters

The sklearn-compatible nested-config surface (architecture A4 step 3).
These are the only documented way to construct a configured
estimator (`tabular_config=...`, `scheduler=...`, etc.); field
additions follow the F7 ALPHA → BETA → STABLE promotion path.

```{eval-rst}
.. autoclass:: seq_sklearn.TabularConfigParams
   :members:
   :show-inheritance:

.. autoclass:: seq_sklearn.OptimizerParams
   :members:
   :show-inheritance:

.. autoclass:: seq_sklearn.SchedulerParams
   :members:
   :show-inheritance:

.. autoclass:: seq_sklearn.LossParams
   :members:
   :show-inheritance:

.. autoclass:: seq_sklearn.SamplerParams
   :members:
   :show-inheritance:

.. autoclass:: seq_sklearn.TFTAdvancedParams
   :members:
   :show-inheritance:
```

## Cross-validation

```{eval-rst}
.. autoclass:: seq_sklearn.EntityTimeSeriesSplit
   :members:
   :show-inheritance:
```

## Hardware

```{eval-rst}
.. autoclass:: seq_sklearn.HardwareTier
   :members:

.. autofunction:: seq_sklearn.detect
```

## Errors

```{eval-rst}
.. autoexception:: seq_sklearn.SeqSklearnError
.. autoexception:: seq_sklearn.ConfigError
.. autoexception:: seq_sklearn.DataContractError
.. autoexception:: seq_sklearn.TrainingError
.. autoexception:: seq_sklearn.PredictionError
.. autoexception:: seq_sklearn.NotFittedError
```

## Attention outputs

```{eval-rst}
.. autoclass:: seq_sklearn.AttentionOutput
   :members:

.. autoclass:: seq_sklearn.RegressionAttentionOutput
   :members:
```

## Tuning

```{eval-rst}
.. autofunction:: seq_sklearn.suggest_params

.. autofunction:: seq_sklearn.optuna_trial_guard
```
