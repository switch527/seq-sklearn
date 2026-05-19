# Tune with Optuna

seq-sklearn ships a first-class Optuna integration that handles the
hyperparameter search space, in-training pruning, and the
sklearn-compatible glue. You don't write the search space; you call
one helper.

## The minimal recipe

```python
import optuna
from sklearn.model_selection import cross_val_score

from seq_sklearn import TFTClassifier, suggest_params, optuna_trial_guard
from seq_sklearn import EntityTimeSeriesSplit


def objective(trial):
    with optuna_trial_guard(trial):
        params = suggest_params(trial, model="tft", task_type="binary")
        clf = TFTClassifier(
            task_type="binary",
            tabular_config=tabular_config,    # your schema
            optuna_trial=trial,               # enables in-training pruning
            **params,
        )
        cv = EntityTimeSeriesSplit(n_splits=3, lookback=params["lookback"])
        scores = cross_val_score(clf, panel, y, cv=cv, scoring="roc_auc")
        return scores.mean()


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
)
study.optimize(objective, n_trials=50)
print(study.best_params)
```

## What `suggest_params` does

`suggest_params(trial, model="tft", task_type="binary")` returns a
`dict` of `{hyperparameter: value}` where each value was drawn from
the library-curated search space for that model/task pair. The space
is conservative (ranges that empirically converge on a single GPU
within a sensible time budget), not maximally exploratory.

The space is ALPHA-tier in v1: search-space *defaults* may change
without a MINOR bump. If you need a stable search space across
versions (e.g. for a reproducible paper run), pass an explicit
`search_space` argument to `suggest_params`, which fixes the bounds.

## In-training pruning

`optuna_trial=trial` on the estimator wires the library's native
`_LightningModule` pruning hook to Optuna's pruner. Trials that
underperform the median at an early epoch are killed before completing
all `max_epochs`, which saves a large fraction of the search budget.

The library does NOT use Optuna's
`PyTorchLightningPruningCallback`; the native hook is shipped to
preserve Lightning 2.6 lifecycle events. You don't have to do
anything to opt in — passing `optuna_trial` is enough.

## The `optuna_trial_guard` context

`with optuna_trial_guard(trial):` wraps the body so that a trial
failing for a recoverable reason (NaN loss, GPU OOM at an aggressive
`batch_size`) is reported as a `TrialState.PRUNED` instead of
crashing the study. The guard re-raises unrecoverable errors.

## Common pitfalls

- **Not setting `cv` to `EntityTimeSeriesSplit`.** sklearn's default
  KFold leaks future periods on multi-entity panels; see
  [time series splitting](time_series_splitting).
- **Using a global seed inside `objective`.** Optuna's sampler has its
  own seed; the estimator's `seed` should come from `params` or stay
  unset to draw a different value per trial. Setting the same seed
  every trial defeats the search.
- **Forgetting to pass `lookback` consistently** between the splitter
  and the suggested `tabular_config.lookback`.

```{testcode}
from seq_sklearn import optuna_trial_guard, suggest_params

assert callable(suggest_params)
assert callable(optuna_trial_guard)
```
