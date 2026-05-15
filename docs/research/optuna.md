# Optuna 4.x research notes (seq-sklearn v1)

## Source citations

- Optuna on PyPI (4.8.0, 2026-03-16, Python 3.9 to 3.14): https://pypi.org/project/optuna/
- Optuna 4.0 release notes (deprecation removals): https://github.com/optuna/optuna/releases/tag/v4.0.0
- Optuna 4.0 migration guide (multi-objective, CLI, integrations): https://github.com/optuna/optuna/discussions/5573
- `optuna.trial.Trial` API reference (suggest_*, report, should_prune): https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
- `optuna.trial.FixedTrial` reference: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.FixedTrial.html
- `optuna.pruners.MedianPruner` reference: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html
- `optuna.pruners.HyperbandPruner` reference: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html
- Pruners index: https://optuna.readthedocs.io/en/stable/reference/pruners.html
- `optuna.TrialPruned` exception: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.exceptions.TrialPruned.html
- Samplers index (defaults, GP, TPE, Random, NSGA-II, CMA-ES): https://optuna.readthedocs.io/en/stable/reference/samplers/index.html
- Easy parallelization tutorial (`n_jobs`, RDB storage): https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
- `optuna-integration` package (4.8.0, 2026-03-16): https://github.com/optuna/optuna-integration
- `PyTorchLightningPruningCallback` reference (optuna-integration 4.6): https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.PyTorchLightningPruningCallback.html
- `optuna-dashboard` on PyPI (0.20.0, 2025-11-10): https://pypi.org/project/optuna-dashboard/

## Version pin recommendation (optuna + optuna-integration)

Pin `optuna>=4.0,<5` and `optuna-integration>=4.0,<5`. Optuna 4.0 was the
breaking release: it removed `optuna.multi_objective`, the `SkoptSampler`,
`CatalystPruningCallback`, `FastAIV1PruningCallback`, the deprecated
`CmaEsSampler` from the integration namespace, `optuna.samplers.MOTPESampler`,
`samplers.intersection`, the `study optimize` CLI subcommand, and the private
`_ask`/`_tell` helpers ([4.0 notes](https://github.com/optuna/optuna/releases/tag/v4.0.0),
[migration guide](https://github.com/optuna/optuna/discussions/5573)). The
current published versions are `optuna==4.8.0` (2026-03-16) and
`optuna-integration==4.8.0` (2026-03-16), both requiring Python 3.9+
([PyPI](https://pypi.org/project/optuna/),
[repo](https://github.com/optuna/optuna-integration)).

Practical floor for seq-sklearn: `optuna>=4.4` (the GPSampler reduced-cost
work landed in 4.5/4.6 but is not load-bearing). Holding to `>=4.0` is enough
for the dropped-API list; `>=4.4` buys faster GP and avoids 4.0 to 4.3 rough
edges.

## Trial API surface

Active suggest methods on `optuna.trial.Trial`
([Trial reference](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html)):

- `suggest_categorical(name, choices)`
- `suggest_int(name, low, high, *, step=1, log=False)`
- `suggest_float(name, low, high, *, step=None, log=False)`

`suggest_uniform`, `suggest_loguniform`, and `suggest_discrete_uniform` are
still callable in 4.x but flagged for removal in 6.0. The docs say to use
`suggest_float(...)`, `suggest_float(..., log=True)`, and
`suggest_float(..., step=...)` instead
([Trial reference](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html)).
seq-sklearn's `suggest_params` should use the unified `suggest_float` form
exclusively.

## Pruning lifecycle (report -> should_prune -> TrialPruned)

The idiom is unchanged from 3.x. Per the Trial reference, `trial.report(value,
step)` records the intermediate objective for the given step: "The reported
values are used by the pruners to determine whether this trial should be
pruned" and "only the first reported value per step is retained"
([Trial reference](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html)).
The metric becomes "the metric for step N" at `report` time, not at
`should_prune` time. `should_prune()` then asks the pruner, with the recorded
history as input, whether to stop; it returns a boolean. The objective is
responsible for raising: the docs on `TrialPruned` say the exception "should
be raised after calling `trial.should_prune()`"
([TrialPruned](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.exceptions.TrialPruned.html)).

For seq-sklearn this means: call `trial.report(val_metric, epoch)` first,
then `if trial.should_prune(): raise optuna.TrialPruned()`. Reporting
without a should_prune check still updates pruner state for the next step.

## `PyTorchLightningPruningCallback` import path and gotchas

The callback lives in the separate `optuna-integration` package on PyPI
([repo](https://github.com/optuna/optuna-integration),
[callback ref](https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.PyTorchLightningPruningCallback.html)).
The canonical 2026 import is:

```python
from optuna_integration import PyTorchLightningPruningCallback
```

Constructor: `PyTorchLightningPruningCallback(trial, monitor)` where `monitor`
is the Lightning metric key, e.g. `"val_loss"`. The docs reference
`lightning.pytorch.Trainer` and `lightning.pytorch.LightningModule`, so the
callback is built against the modern `lightning` namespace, not the legacy
`pytorch_lightning` one
([callback ref](https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.PyTorchLightningPruningCallback.html)).

Gotchas:

1. Distributed Lightning training requires calling
   `PyTorchLightningPruningCallback.check_pruned()` manually after `fit`, and
   the study must use RDB storage rather than in-memory storage
   ([callback ref](https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.PyTorchLightningPruningCallback.html)).
   seq-sklearn v1 is single-process, so this is not yet a concern.
2. The callback raises `optuna.TrialPruned` from inside Lightning's callback
   pipeline; the seq-sklearn Trainer must let that exception bubble up rather
   than wrap it as a `TrainingError`.

## MedianPruner / HyperbandPruner semantics

`MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1, *,
n_min_trials=1)`
([MedianPruner ref](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html)):

- `n_startup_trials`: "Pruning is disabled until the given number of trials
  finish in the same study."
- `n_warmup_steps`: "Pruning is disabled until the trial exceeds the given
  number of step."
- `interval_steps`: "Interval in number of steps between the pruning checks,
  offset by the warmup steps."
- `n_min_trials`: "Minimum number of reported trial results at a step to
  judge whether to prune."

`n_startup_trials=0, n_warmup_steps=0, n_min_trials=1` is the loosest
configuration. After trial 1 completes and reports a value at step 0, trial 2
can be pruned at step 0 if its reported value is worse than the median (a
median of one). The N1 requirement (second trial prunes at epoch 0) is
reachable with these settings.

`HyperbandPruner(min_resource=1, max_resource='auto', reduction_factor=3,
bootstrap_count=0)`
([HyperbandPruner ref](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html)).
With `max_resource='auto'`, the maximum resource is inferred from the largest
step reported by the first completed trial. `bootstrap_count` is
"incompatible with max_resource='auto'", so if seq-sklearn wants bootstrap
behavior it must pass an explicit `max_resource=num_epochs`. Steps are
counted via `trial.report(value, step)`.

## `FixedTrial` for testing

`optuna.trial.FixedTrial(params: dict[str, Any], number: int = 0)` ships a
trial that returns deterministic values from a dict
([FixedTrial ref](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.FixedTrial.html)).
The docs say it "has the same methods as Trial", which includes `report` and
`should_prune` (both no-ops in practice; `should_prune` returns `False`).
This is the right tool for seq-sklearn's `suggest_params` unit tests: pass a
dict covering every key the function reads, assert the returned
`BaseModelConfig` matches.

## Trial failure modes (Pruned vs. Failed vs. unhandled)

`Study.optimize(..., catch=(), n_jobs=1, ...)`
([Study reference](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html)):

- `optuna.TrialPruned` raised inside the objective marks the trial `PRUNED`
  and the study continues. This is the special-case exception.
- Any other exception by default is re-raised and stops `optimize`. To make
  the study survive arbitrary failures, pass `catch=(Exception,)` (or a
  narrower tuple); caught exceptions mark the trial `FAIL`. Default `catch`
  is the empty tuple, so default behavior is "fail loud"
  ([Study reference](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html)).
- Trial states are `COMPLETE`, `FAIL`, `PRUNED`, plus running/waiting states.

Seq-sklearn's requirement maps `ConfigError` and `TrainingError` to
`optuna.TrialPruned`. That is a deliberate choice: a malformed config or a
training divergence is treated as "not worth ranking" rather than a study
killer. The conversion must happen inside the objective the user writes; the
Trainer raises `TrainingError` and the objective wrapper catches it. The
study itself should keep `catch=()` so anything genuinely unexpected fails
loud.

## Parallel trials and thread safety

`Study.optimize` supports `n_jobs > 1` via joblib's threading backend
([parallel tutorial](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)).
Multiple objective invocations run concurrently in threads against the same
study. Optuna guards study state with an internal lock, but the user's
objective (and any callback it constructs) must be thread-safe with respect
to shared resources: model files, GPU allocators, logging handles. seq-sklearn
v1 disclaims distributed training but the pruning hook will still see
concurrent `trial.report` calls from sibling threads when a user opts into
`n_jobs > 1`. Design implication: the hook must hold no module-level mutable
state. Per-trial state lives on the `optuna.Trial` object, which Optuna
already serializes; the hook itself stays stateless.

For true multi-process or multi-node parallelism, Optuna requires RDB or
journal storage rather than in-memory storage
([parallel tutorial](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)).
seq-sklearn does not pick a storage backend; the user does.

## Decisions implied for seq-sklearn

1. Pin `optuna>=4.4,<5` and `optuna-integration>=4.4,<5` in
   `pyproject.toml` extras (e.g. `[tune]`). Both are optional dependencies;
   `seq_sklearn.training` falls back to a no-op when the user passes
   `optuna_pruning_trial=None`.
2. `suggest_params(trial, model_class, base=None)` uses `suggest_categorical`,
   `suggest_int`, and `suggest_float` exclusively. No `suggest_uniform`.
3. The Trainer's pruning hook calls `trial.report(val_metric, epoch)` at the
   end of each validation epoch, then `if trial.should_prune(): raise
   optuna.TrialPruned()`. Report first, prune second.
4. Import the Lightning callback as `from optuna_integration import
   PyTorchLightningPruningCallback`. Document that the `optuna-integration`
   PyPI package must be installed alongside `optuna`.
5. The N1 acceptance test seeds `MedianPruner(n_startup_trials=0,
   n_warmup_steps=0, n_min_trials=1)` and asserts trial 2 stops at epoch 0.
6. The objective wrapper documented in the README catches `ConfigError` and
   `TrainingError` and re-raises as `optuna.TrialPruned`; the underlying
   `study.optimize` runs with default `catch=()` so genuinely unexpected
   exceptions surface.
7. `optuna-dashboard>=0.20` is a documentation-only suggestion. The library
   does not import it ([dashboard PyPI](https://pypi.org/project/optuna-dashboard/)).
8. The pruning hook holds no module-level mutable state, which keeps
   `n_jobs > 1` users unblocked even though seq-sklearn does not test
   multi-thread runs in v1.
