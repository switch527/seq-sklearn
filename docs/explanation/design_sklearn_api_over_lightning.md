# Design: the sklearn estimator API over a Lightning trainer

seq-sklearn is two layers stacked on each other: a Lightning-based
training/inference engine inside, a scikit-learn-compatible estimator
contract outside. This page explains why.

## The user-facing layer is scikit-learn

Every public estimator (`TFTClassifier`, `TFTRegressor`) implements
the sklearn API: `__init__` stores parameters verbatim with no
logic, `fit(X, y)` returns `self` and sets `*_` attributes,
`predict` / `predict_proba` exist, `get_params` / `set_params` are
the parameter contract, and the class passes
`sklearn.utils.estimator_checks.check_estimator`'s curated v1.6
suite (with documented expected-fails for the panel-shape mismatch
points). The CI gate (Phase 9 N1) runs that suite over both
estimators on every PR.

Why: the ML community we target (anyone using scikit-learn) has
strong expectations about what an estimator looks like. Meeting
those expectations means our estimators compose into `Pipeline`,
`GridSearchCV`, `cross_val_score`, `OptunaSearchCV`, and any
third-party tool that reads `sklearn.base.is_classifier`. We get
that composability without writing per-tool adapters.

## The internal layer is pytorch-lightning

Inside `fit`, the estimator constructs a `LightningModule` (the
TFT backbone + head + loss + optimizer + scheduler) and a
`Trainer` (Lightning's loop manager). Lightning owns the training
loop: device placement, mixed precision, gradient accumulation,
callbacks, checkpointing.

Why: writing a correct training loop is hard. Hardware detection
and precision management are hard. Logging, early stopping, and
gradient clipping are hard. Lightning has solved each of these and
maintains them; reimplementing them inside seq-sklearn would
trade real engineering risk for zero user-visible benefit.

## The seam between the two

`BaseSequenceEstimator` is the bridge. It owns:

- the pydantic config classes (the user-facing parameter surface);
- the `LightningModule` factory (`_make_module`);
- the `Trainer` factory (`_make_trainer`);
- the fit-state attributes (`_module`, `_trainer`, `transformer_`,
  `calibrator_`, `n_features_in_`, `feature_names_in_`, etc.);
- the prediction-time orchestration (`_predict_raw` → calibration
  → caller-row-order restore).

Subclass estimators (`TFTClassifier`, `TFTRegressor`) provide a
small, focused override surface: the backbone factory, the head
factory, the loss factory, the prediction post-processing for their
task type. Everything else inherits.

## Why not just expose Lightning directly?

You could write a user-facing API that wraps a Lightning Trainer
and is the only entry point ("here's a Trainer, here's a
LightningModule, fit"). pytorch-forecasting does roughly that. The
costs:

- **No sklearn composability.** Users have to write their own glue
  to plug it into a Pipeline or GridSearchCV; they all write the
  same glue, slightly wrong each time.
- **No standard fit-state contract.** `fit` returning `self`,
  `*_` attributes — these are conventions sklearn users rely on for
  introspection and serialization.
- **Wider surface area to maintain.** Lightning's Trainer has many
  knobs that don't make sense to expose for a fixed library use
  case (distributed strategies, custom logger backends).

The sklearn-shaped facade narrows that surface to the
hyperparameters that actually matter for our model family, while
preserving full Lightning capability internally.

## Stability boundary

Public:

- The estimator classes and their methods (STABLE per architecture
  A3).
- The pydantic config schemas (STABLE).
- `seq_sklearn.detect`, `HardwareTier`, `EntityTimeSeriesSplit`
  (STABLE).

Not public:

- The internal `_LightningModule` subclass, the internal `Trainer`
  args, the backbone module structure. INTERNAL-tier; can change
  in MINOR releases. If you find yourself reaching into
  `clf._module.backbone.lstm` for a feature, file an issue —
  that's a request for a public surface.
