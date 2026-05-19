# Glossary

The canonical sklearn-contract terms seq-sklearn uses. These
definitions are the contract; API docs and guides cross-link here
rather than restating them.

## General concepts

```{glossary}
estimator
    An object with a `fit(X, y)` method that returns `self`. After
    `fit`, the object carries fitted state (attributes ending in
    `_`) and supports `predict(X)` (and `predict_proba(X)` for
    classifiers, `predict_quantiles(X)` for quantile regressors).

predictor
    An estimator that also has `predict` (so anything that does both
    `fit` and `predict`). seq-sklearn's TFT classes are predictors.

transformer
    An estimator that also has `transform` (so `fit_transform` is
    available). `TabularToSequence` is the only public transformer
    seq-sklearn ships in v1.

panel
    A tidy DataFrame with one row per `(entity_id, period)` pair.
    The unit seq-sklearn consumes.

entity
    A logical subject in the panel: customer, patient, device,
    sensor. Identified by `id_col`.

period
    An ordered observation index within an entity. Identified by
    `time_col`. Consecutive rows of the same entity are consecutive
    periods, regardless of wall-clock spacing.

lookback
    The number of most-recent periods the model sees per
    prediction. Set on `TabularToSequenceConfig.lookback`; must
    match the value passed to `EntityTimeSeriesSplit.lookback`.

window
    A length-`lookback` slice ending at some period of an entity.
    The model emits one prediction per window.
```

## Target types

```{glossary}
binary
    Two-class classification. `y` is `{0, 1}` or two distinct labels.

multiclass
    Three or more mutually-exclusive classes. `y` is integer-valued
    or label-valued.

regression_point
    Single-value regression. `y` is a float per row.

regression_quantile
    Quantile regression at configured levels (e.g. `[0.1, 0.5, 0.9]`).
    `predict_quantiles` returns one value per quantile per row.
```

## Methods on every estimator

```{glossary}
fit(X, y)
    Train on `X` (a panel DataFrame) and `y` (per-row label/value).
    Returns `self`. After this call, attributes ending in `_` are
    populated and `predict` works.

predict(X)
    Hard predictions: class label for classifiers, point estimate
    for regressors. Caller-row-order is preserved (F1 contract).

predict_proba(X)
    Class probabilities (classifiers only). Calibrated if the
    estimator was fit with `cal_fraction > 0`.

predict_with_attention(X)
    Returns `AttentionOutput` / `RegressionAttentionOutput` with
    predictions plus the interpretable surfaces (variable selection,
    temporal attention).

score(X, y)
    sklearn-conventional default score: accuracy for classifiers,
    R² for point regressors, mean pinball loss for quantile
    regressors.

get_params(deep=True) / set_params(**params)
    The sklearn contract methods that make estimators composable
    with `Pipeline`, `GridSearchCV`, and `cross_val_score`.

save(path) / load(path)
    Safetensors + JSON persistence. NOT pickle.

export_onnx(path, X)
    Export the backbone+head as ONNX (raw logits). Requires the
    optional `[onnx]` extra.
```

## Fitted-state attributes

```{glossary}
classes_
    Classifiers only. The seen target labels in their sklearn order.

n_features_in_ / feature_names_in_
    The sklearn `feature_names_in_` contract. For a panel estimator,
    these are the *columns* of the input DataFrame (id, time,
    static, time-varying), not the engineered features.

calibrator_
    Set when `cal_fraction > 0`. The fitted temperature/isotonic
    calibrator. `None` otherwise.

transformer_
    The fitted `TabularToSequence` instance built from the panel's
    schema. Used internally on `predict`.
```

## Sklearn-contract parameters

```{glossary}
random_state / seed
    Determinism seed. seq-sklearn uses `seed` on estimator init
    (consistent with the wider deep-learning ecosystem) but honors
    `random_state` from sklearn-style protocols where applicable.

n_jobs
    sklearn convention; not used inside the model (Lightning manages
    its own workers). May be honored by surrounding code (Optuna
    parallel trials, `cross_val_score`).
```

## Stability tiers

```{glossary}
STABLE
    Public API surface covered by SemVer. Breaking change requires
    a MAJOR bump. See `docs/requirements.md` for the per-symbol
    tier table.

BETA
    Public but still iterating. Fields may be added in MINOR
    releases (but never removed in MINOR). Prefer attribute access
    over tuple unpacking.

ALPHA
    Experimental defaults that may change without a MINOR bump (the
    `suggest_params` search space is the only ALPHA in v1). Pass an
    explicit search space for stable behavior.

INTERNAL
    `seq_sklearn._*` modules and anything not re-exported by
    `seq_sklearn`. Not covered by SemVer.
```
