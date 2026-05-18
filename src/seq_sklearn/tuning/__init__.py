"""Optuna integration (A16): search-space sampling and trial pruning.

Public surface:

* :func:`seq_sklearn.tuning.suggest_params.suggest_params` samples a
  config from the per-model default search space, closed under the F5
  validity matrix by construction.
* :func:`seq_sklearn.tuning.pruning.optuna_trial_guard` wraps a user
  ``objective(trial)`` body so library-raised :class:`ConfigError` /
  :class:`TrainingError` become :class:`optuna.TrialPruned`.

The frozen-pydantic-dump to mutable-adapter-kwargs bridge
(``config_to_estimator_kwargs``, in ``_estimator_bridge``) and the
per-family ALPHA-key list (``_alpha_keys``) are dispatch plumbing for
the documented ``objective`` pattern, imported from their modules
directly and deliberately kept out of the headline ``__all__``.
"""

from seq_sklearn.tuning.pruning import optuna_trial_guard
from seq_sklearn.tuning.suggest_params import suggest_params

__all__ = ["optuna_trial_guard", "suggest_params"]
