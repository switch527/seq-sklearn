# scikit-learn 1.6+ estimator contract (research, 2026-05)

Pre-implementation research for the seq-sklearn estimator API. All claims
cited. Target: a TFT classifier (and future regressor/forecaster) that passes
a pinned subset of `check_estimator` while accepting pandas DataFrames as `X`.

## Source citations

1. Release index, latest stable: https://scikit-learn.org/stable/whats_new.html
2. PyPI release history: https://pypi.org/project/scikit-learn/
3. `Tags` dataclass reference: https://scikit-learn.org/stable/modules/generated/sklearn.utils.Tags.html
4. `ClassifierTags` reference: https://scikit-learn.org/stable/modules/generated/sklearn.utils.ClassifierTags.html
5. Developing estimators guide: https://scikit-learn.org/stable/developers/develop.html
6. 1.6 changelog (public tags, `_xfail_checks` removal): https://scikit-learn.org/stable/whats_new/v1.6.html
7. `check_estimator` API: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html
8. `parametrize_with_checks` API: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html
9. `NotFittedError` reference: https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.NotFittedError.html
10. `GridSearchCV` reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
11. SLEP007 (feature names): https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep007/proposal.html
12. sklearn-compat: https://sklearn-compat.readthedocs.io/ and https://github.com/sklearn-compat/sklearn-compat
13. Issue #30479 (ClassifierMixin tag-chain pitfall): https://github.com/scikit-learn/scikit-learn/issues/30479

## Version pin recommendation

Current stable is **1.8.0** (released 2025-12-10) [1]. 1.9 is in development [1].
The public `Tags` API landed in **1.6** [6], and 1.7 introduced no third-party
breaking change per the sklearn-compat maintainers [12]. 1.8 brought native
Array API support and free-threaded CPython, neither of which affects our
estimator surface [1].

**Pin: `scikit-learn>=1.6,<2.0`.** Rationale: 1.6 is the floor we need for
`__sklearn_tags__`; the major boundary is the right deprecation horizon
since sklearn follows SPEC0 (drop in lockstep with NEP29) and signals
breakage one minor in advance [6, 12]. We will not vendor `sklearn-compat`,
since our floor (1.6) is already above the version skew it papers over.

## `__sklearn_tags__` exact signature and field values

Per [3, 5, 6], `__sklearn_tags__` is an **instance method** (not a
classmethod, despite the prompt phrasing). It takes `self`, returns a
`sklearn.utils.Tags` instance, and must chain through `super()`:

```python
def __sklearn_tags__(self):
    tags = super().__sklearn_tags__()
    tags.input_tags.dataframe = True
    tags.input_tags.allow_nan = False
    tags.target_tags.required = True
    tags.classifier_tags.multi_class = True
    tags.classifier_tags.multi_label = False
    return tags
```

The `Tags` dataclass [3]:

```python
@dataclass
class Tags:
    estimator_type: str | None        # "classifier" | "regressor" | "transformer" | "clusterer" | "outlier_detector" | "density_estimator"
    target_tags: TargetTags
    transformer_tags: TransformerTags | None = None
    classifier_tags: ClassifierTags | None = None
    regressor_tags: RegressorTags | None = None
    array_api_support: bool = False
    no_validation: bool = False
    non_deterministic: bool = False
    requires_fit: bool = True
    _skip_test: bool = False
    input_tags: InputTags = <factory>
```

Sub-tag dataclasses [3, 4]:

```python
@dataclass
class InputTags:
    sparse: bool
    allow_nan: bool
    dataframe: bool        # this is the "accepts DataFrames" flag
    two_d_array: bool
    string: bool
    dict: bool
    categorical: bool
    positive_only: bool

@dataclass
class TargetTags:
    required: bool
    one_d_labels: bool
    two_d_labels: bool
    positive_only: bool
    multi_output: bool
    single_output: bool

@dataclass
class ClassifierTags:
    poor_score: bool
    multi_class: bool
    multi_label: bool

@dataclass
class RegressorTags:
    poor_score: bool
```

The legacy `X_types=["dataframe"]` / `X_types=["2darray"]` strings from
`_more_tags` are **gone in 1.6**; use `tags.input_tags.dataframe = True`
and `tags.input_tags.two_d_array = True` [6].

`_xfail_checks` as a tag is **removed in 1.6** [6]. Skipping individual
checks now goes through `expected_failed_checks` on `check_estimator` /
`parametrize_with_checks`, not the tags object.

Inherit from `BaseEstimator` and the relevant mixin (`ClassifierMixin`,
`RegressorMixin`, etc.); setting the private `_estimator_type` attribute
is deprecated in 1.6 [6]. The mixin's `__sklearn_tags__` sets
`estimator_type` for us; we override only what we need. Issue #30479 is
the canonical pitfall: forgetting `super().__sklearn_tags__()` breaks
the chain [13].

## Subsetting `check_estimator` (2026 API)

Two entrypoints, same parameters [7, 8].

```python
sklearn.utils.estimator_checks.check_estimator(
    estimator=None,
    *,
    legacy: bool = True,
    expected_failed_checks: dict[str, str] | None = None,
    on_skip: Literal["warn"] | None = "warn",
    on_fail: Literal["raise", "warn"] | None = "raise",
    callback: Callable | None = None,
)

sklearn.utils.estimator_checks.parametrize_with_checks(
    estimators,
    *,
    legacy: bool = True,
    expected_failed_checks: Callable | None = None,
    xfail_strict: bool | None = None,
)
```

For `parametrize_with_checks`, `expected_failed_checks` is a **callable**
returning a dict; for `check_estimator` it is the dict directly [7, 8].
`generate_only=True` is deprecated; use `estimator_checks_generator`
instead [6].

**Our `tests/conftest.py` pattern (per F1.1):**

```python
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks
from seq_sklearn.models import TFTClassifier

def _xfail(estimator):
    return {
        "check_estimators_dtypes":
            "TFT requires float32 sequence input, not float64/int casts",
        "check_dtype_object":
            "object-dtype X unsupported, DataFrame schema enforced upstream",
        "check_methods_sample_order_invariance":
            "attention is order-sensitive by design",
        "check_fit_idempotent":
            "stochastic optimizer; idempotency requires seeded re-fit",
    }

@parametrize_with_checks(
    [TFTClassifier(tabular_config={"lookback": 6, "horizon": 1})],
    expected_failed_checks=_xfail,
    xfail_strict=True,
)
def test_sklearn_compatible(estimator, check):
    check(estimator)
```

`xfail_strict=True` makes an unexpectedly passing check a hard failure,
which is what we want to catch silent compatibility wins [8].

## `set_params` / `get_params` nested-config worked example

`get_params(deep=True)` walks nested estimators via `__init__` parameter
names and exposes leaves as `<name>__<param>` [5, 10]. The convention
chains: `preprocessor__num__imputer__strategy` is valid [10].

Pydantic `BaseModel` is **not** an sklearn estimator. For the double-
underscore protocol to work on `tabular_config__lookback`, the config
object must either (a) implement `get_params`/`set_params` itself, or
(b) be unpacked into estimator-level parameters at `__init__` time.

Our chosen pattern is **(a)**: wrap each pydantic config in a thin
`BaseEstimator` adapter so it participates in the recursion. Sketch:

```python
class TabularConfigParams(BaseEstimator):
    def __init__(self, lookback: int = 12, horizon: int = 1, ...):
        self.lookback = lookback
        self.horizon = horizon
        ...

    def to_pydantic(self) -> TabularConfig:
        return TabularConfig(lookback=self.lookback, horizon=self.horizon, ...)

class TFTClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, tabular_config: TabularConfigParams | None = None, ...):
        self.tabular_config = tabular_config or TabularConfigParams()
```

`get_params(deep=True)` then yields `tabular_config__lookback`, and
`GridSearchCV(param_grid={"tabular_config__lookback": [6, 12, 24]})`
works without further plumbing [5, 10].

A simpler alternative is to flatten: lift every config field to a
top-level `__init__` parameter on the estimator. We reject this because
it duplicates the pydantic schema and loses validation grouping.

## Required fit-state attributes (full list as of 2026)

Set inside `fit`, every public learned attribute ends with `_` [5]:

- `n_features_in_: int`. Required on every tabular estimator. Set
  automatically if you call `utils.validation.validate_data(self, X, ...)` [5, 6].
- `feature_names_in_: np.ndarray[str]`. Required when `X` is a DataFrame
  (or any input with string feature names). 1d, object dtype, strings only.
  Also set by `validate_data` [5, 11].
- `classes_: np.ndarray`. Required on classifiers, sorted ascending. Spec
  pattern is `self.classes_, y = np.unique(y, return_inverse=True)` [5].
- `n_outputs_: int`. Required on multi-output regressors and multi-label
  classifiers; sklearn checks rely on it whenever `target_tags.multi_output`
  or `classifier_tags.multi_label` is True [5].
- `labels_: np.ndarray`. Clusterers only [5]. Irrelevant to us at v1.

No new mandatory fit-state attribute landed in 1.6 through 1.8 for
classifiers or regressors beyond the above. Array API support added in
1.8 affects internal dtype handling only, not the post-fit surface [1].

`check_is_fitted(self)` is the validation gate at the top of `predict`,
`predict_proba`, etc., and raises `NotFittedError` if no `*_` attribute
is set [5].

## DataFrame input contract

sklearn accepts `pd.DataFrame` (and polars frames) as `X` from 1.6+ via
`validate_data` [5, 6]. The flow:

1. Estimator sets `input_tags.dataframe = True` in `__sklearn_tags__`.
2. `fit(self, X, y)` calls `validate_data(self, X, y, ensure_2d=False, ...)`
   which populates `n_features_in_` and `feature_names_in_` and returns
   the validated array(s) [5, 6].
3. `predict(self, X)` calls `validate_data(self, X, reset=False, ...)`,
   which verifies the incoming column names match `feature_names_in_` and
   raises if not [5, 11].

`check_estimator` does not auto-skip array-only checks when
`input_tags.dataframe=True`. It still runs both DataFrame and ndarray
inputs against estimators that declare DataFrame support; the tag adds
DataFrame coverage, it does not remove ndarray coverage [5]. If our TFT
genuinely cannot accept a raw 2D ndarray (because it needs a time-index
column or an entity-id column), we must mark the relevant ndarray-only
checks as expected failures via `expected_failed_checks`. See the
decisions section for the concrete list.

Constraint to flag for the architecture doc: SLEP007 requires
`feature_names_in_` to be string-typed [11]. DataFrames with integer
column names raise on round-trip; we must enforce string column names
in our data adapter or in `validate_data` kwargs.

## `NotFittedError` multiple-inheritance pattern

`sklearn.exceptions.NotFittedError(ValueError, AttributeError)` [9]. The
inheritance is unchanged since 0.18 [9]. The dual base is intentional:
catching `AttributeError` (e.g., `hasattr(est, "coef_")`) and catching
`ValueError` both work.

Library-internal subclassing for a richer exception (e.g., to attach a
`stage` or `expected_attr` field) is straightforward MRO:

```python
class SeqSklearnError(Exception):
    """Root for library-internal errors."""

class TFTNotFittedError(NotFittedError, SeqSklearnError):
    def __init__(self, estimator_name: str, missing: str):
        self.estimator_name = estimator_name
        self.missing = missing
        super().__init__(
            f"{estimator_name} is not fitted yet (missing {missing!r}). "
            f"Call fit() before predict()."
        )
```

MRO: `TFTNotFittedError → NotFittedError → ValueError → AttributeError →
SeqSklearnError → Exception`. Downstream code catching either
`NotFittedError` or `SeqSklearnError` still works. No 1.6 through 1.8
change touches this surface [6, 9].

If `check_is_fitted` is what raises, it raises `NotFittedError` directly;
to substitute our subclass, wrap the call site:

```python
def predict(self, X):
    try:
        check_is_fitted(self)
    except NotFittedError as e:
        raise TFTNotFittedError(type(self).__name__, "classes_") from e
    ...
```

## Pipeline + GridSearchCV composition

`GridSearchCV` consumes any estimator whose `set_params` accepts the
keys in `param_grid` [10]. With the adapter pattern from §5, the
following works without changes:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from seq_sklearn.models import TFTClassifier

pipe = Pipeline([("clf", TFTClassifier())])

grid = GridSearchCV(
    pipe,
    param_grid={
        "clf__tabular_config__lookback": [6, 12, 24],
        "clf__training_config__lr": [1e-3, 3e-4],
    },
    cv=3,
    n_jobs=1,
)
grid.fit(df, y)
```

`clf__tabular_config__lookback` resolves by recursive `set_params`:
`pipe.set_params(clf__tabular_config__lookback=12)` →
`pipe.named_steps["clf"].set_params(tabular_config__lookback=12)` →
`tabular_config.set_params(lookback=12)`. Each `set_params` rebuilds the
pydantic config when `to_pydantic()` is called inside `fit`, so
validation runs every fit cycle and bad combinations fail fast.

The double-underscore protocol is unchanged in 1.6 through 1.8 [5, 10].

## Decisions implied for seq-sklearn

**Pin.** `scikit-learn>=1.6,<2.0`. Single major; no `sklearn-compat`.

**Tags.** Implement `__sklearn_tags__` on every estimator. For
`TFTClassifier`:

```python
def __sklearn_tags__(self):
    tags = super().__sklearn_tags__()
    tags.input_tags.dataframe = True
    tags.input_tags.two_d_array = False    # we reject raw ndarrays
    tags.input_tags.sparse = False
    tags.input_tags.allow_nan = False      # NaN policy enforced upstream
    tags.input_tags.categorical = True     # via DataFrame schema
    tags.target_tags.required = True
    tags.classifier_tags.multi_class = True
    tags.classifier_tags.multi_label = False
    tags.non_deterministic = True          # stochastic SGD
    tags.array_api_support = False         # PyTorch tensors internally
    return tags
```

**Checks to include** (legacy=True, all checks generated). The base set
is the full `parametrize_with_checks` generator.

**Checks to mark `expected_failed_checks`** (ship with rationale, all
cite the requirement they violate):

| Check | Rationale |
|---|---|
| `check_estimators_dtypes` | TFT requires float32 sequence tensors; the check casts to float64/int. |
| `check_dtype_object` | object-dtype `X` rejected by our DataFrame schema (F5.2). |
| `check_fit2d_1sample` | 1-sample fit is meaningless for a sequence model with `lookback>=1`. |
| `check_fit2d_1feature` | single-feature fit collides with our entity/time columns. |
| `check_methods_sample_order_invariance` | attention is order-sensitive by design (Lim 2021, eq. 4). |
| `check_fit_idempotent` | stochastic optimizer; idempotency requires seeded re-fit, which the check does not arrange. |
| `check_n_features_in_after_fitting` | only relevant for ndarray reshape paths we do not take. |
| `check_estimators_pickle` (only if torch state dict round-trip fails) | revisit after impl; do not pre-defer. |

This list is the **v1 starting point** for `tests/conftest.py::_xfail`.
Each entry must keep a one-line rationale so the next swarm run sees the
deferral, per the consensus protocol.

**Test target.** `xfail_strict=True`. Any unexpected pass is a real
compatibility win and we want to remove the deferral the same PR.

**Out of scope for v1** (deferred): Array API tag, `set_output` API for
DataFrame output from `predict_proba`, sparse input, multi-label
classifier head.

Word count: 1755.
