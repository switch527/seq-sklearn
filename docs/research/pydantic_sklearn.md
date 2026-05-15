# Pydantic v2 + scikit-learn Estimator Integration

Research for seq-sklearn: canonical pattern for combining a frozen
`pydantic.BaseModel` config with an sklearn-compatible estimator.

## Source citations (URLs)

1. Pydantic releases (v2.13.4 current): https://github.com/pydantic/pydantic/releases
2. Pydantic v2.13 announcement: https://pydantic.dev/articles/pydantic-v2-12-release
3. Pydantic Models concept (frozen, model_copy, pickle): https://docs.pydantic.dev/latest/concepts/models/
4. Pydantic Configuration (`extra`, `frozen`): https://docs.pydantic.dev/latest/concepts/config/
5. Pydantic Serialization (`model_dump`, `round_trip`, mode='json'): https://docs.pydantic.dev/latest/concepts/serialization/
6. Pydantic JSON Schema: https://docs.pydantic.dev/latest/concepts/json_schema/
7. Pydantic Validators (`model_validator`, wrap mode): https://docs.pydantic.dev/latest/concepts/validators/
8. Pydantic Error Handling: https://docs.pydantic.dev/latest/errors/errors/
9. scikit-learn Developing Estimators (1.8): https://scikit-learn.org/stable/developers/develop.html
10. sklearn `BaseEstimator` API: https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
11. sklearn `Tags` API: https://scikit-learn.org/stable/modules/generated/sklearn.utils.Tags.html
12. sklearn 1.6 release highlights (Tags API public): https://scikit-learn.org/stable/whats_new/v1.6.html
13. sklearn `base.py` source (get_params/set_params traversal): https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py
14. Issue: pydantic 2.11.3 + joblib regression: https://github.com/pydantic/pydantic/issues/11746
15. Issue: pickling does not preserve internal state: https://github.com/pydantic/pydantic/issues/11603
16. Discussion: emit enum vs const for single-value Literal: https://github.com/pydantic/pydantic/issues/12148
17. sktime `PytorchForecastingTFT` wrapper (sklearn-style API over PF): https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.pytorchforecasting.PytorchForecastingTFT.html
18. skrub (sklearn-compatible, no pydantic): https://skrub-data.org/stable/index.html

## Version pin recommendation

Current pydantic is **2.13.4** (2026-05-06); 2.13 added Python 3.14
support and the updated `pydantic.v1` namespace [1, 2]. Minor v2 releases
guarantee no breaking changes [4].

Pydantic **2.11.3** has a confirmed regression: joblib parallel jobs
silently produce un-validated model instances [14]. 2.12 fixed that and
a `model_construct` + `MISSING` pickle bug [14].

**Pin `pydantic >= 2.12, < 3`** and **`scikit-learn >= 1.6`** (public
`Tags` API and `__sklearn_tags__` landed in 1.6 [11, 12]).

## Frozen-config + mutable-estimator pattern

`ConfigDict(frozen=True)` makes a pydantic instance immutable: assignment
raises `ValidationError("Instance is frozen")` [3]. This conflicts with
sklearn's `set_params`, which mutates in place and returns `self` [9,
13]. sklearn also forbids validation in `__init__` because `GridSearchCV`
calls `set_params` repeatedly and expects identical semantics [9].

Resolution: the estimator stores hyperparameters as plain attributes
(mutable) and constructs the frozen pydantic config inside `fit`. This
matches sklearn's own "validation deferred to fit" recipe [9].

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted

class TFTClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden_size: int = 64,
        attention_heads: int = 4,
        dropout: float = 0.1,
        lookback: int = 32,
    ):
        # rule from sklearn: assign, do not validate, do not transform
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.lookback = lookback

    def _build_config(self) -> "TFTConfig":
        # pydantic runs cross-field validators here, in fit, not __init__
        return TFTConfig(
            hidden_size=self.hidden_size,
            attention_heads=self.attention_heads,
            dropout=self.dropout,
            tabular_config=TabularToSequenceConfig(lookback=self.lookback),
        )

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.config_ = self._build_config()  # frozen, validated
        # ... training ...
        return self
```

Trailing-underscore `config_` follows sklearn fitted-state convention
[9]. `sklearn.base.clone` round-trips via `get_params`, so it ignores
`config_` and rebuilds from raw scalars. sktime's `PytorchForecastingTFT`
uses the same shape [17]; skrub does too, without pydantic [18].

## Cross-field validator + custom error wrapping

`@model_validator(mode="after")` runs on the constructed instance and
may raise to abort [7]. Non-`ValidationError` exceptions get wrapped
into one [8]. To surface a `ConfigError`, catch at the estimator
boundary, not inside the validator:

```python
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

class ConfigError(ValueError):
    """Public seq-sklearn config error. Carries the pydantic detail."""
    def __init__(self, msg: str, errors: list[dict] | None = None):
        super().__init__(msg)
        self.errors = errors or []

class TFTConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    hidden_size: int
    attention_heads: int

    @model_validator(mode="after")
    def _heads_divide_hidden(self) -> "TFTConfig":
        if self.hidden_size % self.attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"attention_heads ({self.attention_heads})"
            )
        return self

def build_config(**kw) -> TFTConfig:
    try:
        return TFTConfig(**kw)
    except ValidationError as e:
        raise ConfigError(str(e), errors=e.errors()) from e
```

Raising `ValueError` from validators is idiomatic [7]; the wrap happens
at the call site, preserving the structured `errors()` payload. Skip
`mode="wrap"`: it overcomplicates pure exception translation [7].

## `model_dump` round-trip gotchas

`model_dump(mode="python")` preserves Python types: tuples stay tuples,
`Path` stays `Path` [5]. `mode="json"` (and `model_dump_json`) coerces
to JSON-native: tuples to lists, `Path` to string [5]. Reloading via
`model_validate(_json)` round-trips because pydantic coerces lists into
tuple fields and strings into `Path` fields on input.

`round_trip=True` is only required for non-idempotent types like
`Json[T]` [5]; ordinary tuple/Path/Literal fields do not need it.
`Literal[...]` survives round-trip unchanged [5].

Gotcha: pickled models do not preserve `__pydantic_fields_set__` order
[15], so byte-identical pickles across runs are not guaranteed. Not a
correctness issue for joblib; breaks naive pickle-hash equality.

## Nested config + double-underscore traversal

sklearn flattens nested params as `<name>__<param>` and recurses into
attributes that own a `get_params` method [9, 13]. Pydantic models do
not own `get_params`, so the default `BaseEstimator.get_params(deep=True)`
will not descend into `tabular_config`. seq-sklearn must implement the
traversal. Two options:

1. **Flatten at the estimator boundary.** Hoist every nested field
   (`tabular_lookback`). Simple, but the init balloons.
2. **Override `get_params` / `set_params`** to walk the nested config,
   mirroring sklearn's own recursion in `base.py` [13].

```python
class TFTClassifier(ClassifierMixin, BaseEstimator):
    _NESTED = ("tabular_config",)  # names of nested-config attributes

    def get_params(self, deep: bool = True) -> dict:
        out = super().get_params(deep=deep)
        if not deep:
            return out
        for name in self._NESTED:
            nested = getattr(self, name, None)
            if nested is None:
                continue
            for k, v in nested.items():
                out[f"{name}__{k}"] = v
        return out

    def set_params(self, **params) -> "TFTClassifier":
        nested_updates: dict[str, dict] = {}
        flat: dict = {}
        for key, value in params.items():
            head, sep, tail = key.partition("__")
            if sep and head in self._NESTED:
                nested_updates.setdefault(head, {})[tail] = value
            else:
                flat[key] = value
        if flat:
            super().set_params(**flat)
        for name, updates in nested_updates.items():
            current = dict(getattr(self, name))
            current.update(updates)
            setattr(self, name, current)
        return self
```

`tabular_config` is stored as a plain `dict[str, Any]`, not a pydantic
instance, so `set_params` can mutate it. The pydantic
`TabularToSequenceConfig` is built inside `fit`. Same shape as sktime's
PF wrapper [17].

## `__sklearn_tags__` coexistence with pydantic init

`__sklearn_tags__` is an instance method on `BaseEstimator` (1.6+)
returning a `Tags` dataclass [11, 12]. It runs against `self`, not the
pydantic config. No collision: the estimator inherits from
`BaseEstimator` and **owns** a pydantic config; pydantic is not in the MRO.

```python
class TFTClassifier(ClassifierMixin, BaseEstimator):
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        tags.target_tags.required = True
        tags.classifier_tags.multi_class = True
        return tags
```

Tags can depend on hyperparameters because they run on the instance
after `__init__` [11]. Reading `self.hidden_size` is fine; do not call
`self._build_config()` here because `check_estimator` invokes
`__sklearn_tags__` on unconfigured instances. Mixin order:
`Mixin, BaseEstimator` for correct MRO [9].

## Comparable library precedents

- **sktime `PytorchForecastingTFT`** [17]: hyperparameters on wrapper,
  config dict built inside `fit`. No pydantic.
- **skrub** [18]: sklearn-compatible encoders. No pydantic; classic
  shape (attributes on init, validation in `fit`).
- **feature-engine, category_encoders, imbalanced-learn**: surveyed;
  none use pydantic for estimator configs. Vanilla sklearn pattern.
- **pytorch-forecasting** uses Lightning's `BaseModel` plus a
  `TimeSeriesDataSet` spec, not pydantic typed configs [17].

No surveyed library combines a frozen pydantic config with an sklearn
estimator. seq-sklearn is staking new ground; the pattern above is
synthesized from sklearn's documented contract and pydantic's behavior.

## Pickle / joblib compatibility

Pydantic v2 `BaseModel` instances pickle natively via `__reduce__` [3];
`frozen=True` does not change the pickle path. `joblib.dump` / `load`
work for the fitted estimator. Gotchas:

- **Pin `pydantic >= 2.12`.** 2.11.3 silently drops field values in
  joblib worker processes [14].
- `__pydantic_fields_set__` order is not preserved across pickle
  round-trips [15]. Hash by `model_dump_json` (sorted keys), not bytes.
- Do not treat the pydantic config as sole source of truth. Keep raw
  scalars on `self` so `clone` / `get_params` work without `config_`.

## JSON Schema for docs rendering

`model_json_schema()` returns an OpenAPI-shaped dict [6]:

- `Literal["mean", "sum"]` renders as `{"enum": [...]}`.
- Single-value `Literal["x"]` renders as `{"const": "x"}` [16]; some
  UI generators dislike this. Override via `json_schema_extra` if it
  matters.
- `frozen=True` does not affect schema output.
- `extra="forbid"` adds `"additionalProperties": false` [4, 6].

mkdocstrings can embed the schema dict directly. Field docstrings
become `"description"` entries [6].

## Decisions implied for seq-sklearn

1. Pin `pydantic >= 2.12, < 3` and `scikit-learn >= 1.6`.
2. Configs (`TFTConfig`, `TabularToSequenceConfig`, `BaseModelConfig`)
   stay frozen with `extra="forbid"`. Never mutated post-construction.
3. Estimators own raw-scalar attributes mirroring config fields.
   Pydantic instances are built inside `fit` and stored on `self.config_`.
4. Override `get_params` / `set_params` to handle `tabular_config__*`
   traversal. Store the nested attribute as `dict[str, Any]`.
5. Wrap `ValidationError` into `seq_sklearn.errors.ConfigError` at the
   `_build_config()` call site. Validators raise `ValueError`.
6. `__sklearn_tags__` reads raw attributes only, never `self.config_`.
   Mixin order: `ClassifierMixin, BaseEstimator`.
7. Save/load via `joblib`. Configs round-trip through `model_dump_json`
   for human-readable export.
8. `extra="forbid"` is safe for v1.1 evolution: new fields with defaults
   stay backward compatible because old dumps lack the key and pydantic
   supplies the default on `model_validate`. Old caller code does not
   pass the new key to `set_params`, so the estimator is unaffected.
