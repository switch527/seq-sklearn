"""optuna_trial_guard A16 contract (Phase 8, no training).

FixedTrial is the right tool here: the guard never calls report /
should_prune, so a Study is unnecessary and the trial values are
irrelevant. The line the guard draws: recoverable library failures
(ConfigError, TrainingError) prune the trial; a contract violation, a
KeyboardInterrupt, or any unexpected error fails the study fast.
"""

import optuna
import pytest

from seq_sklearn.errors import ConfigError, DataContractError, TrainingError
from seq_sklearn.tuning.pruning import optuna_trial_guard


def _trial() -> optuna.trial.FixedTrial:
    return optuna.trial.FixedTrial({})


def test_config_error_is_pruned_with_original_message() -> None:
    with pytest.raises(optuna.TrialPruned, match="bad cell"), optuna_trial_guard(_trial()):
        raise ConfigError("bad cell")


def test_training_error_is_pruned_with_original_message() -> None:
    with (
        pytest.raises(optuna.TrialPruned, match="non-finite loss"),
        optuna_trial_guard(_trial()),
    ):
        raise TrainingError("non-finite loss")


def test_data_contract_error_propagates() -> None:
    with pytest.raises(DataContractError), optuna_trial_guard(_trial()):
        raise DataContractError("schema mismatch")


def test_unexpected_exception_propagates() -> None:
    with pytest.raises(ValueError, match="genuine bug"), optuna_trial_guard(_trial()):
        raise ValueError("genuine bug")


def test_keyboard_interrupt_propagates() -> None:
    with pytest.raises(KeyboardInterrupt), optuna_trial_guard(_trial()):
        raise KeyboardInterrupt


def test_clean_body_runs_and_does_not_raise() -> None:
    ran = []
    with optuna_trial_guard(_trial()):
        ran.append(True)
    assert ran == [True]


def test_pruned_exception_chains_original_cause() -> None:
    with pytest.raises(optuna.TrialPruned) as excinfo, optuna_trial_guard(_trial()):
        raise TrainingError("root cause")
    cause = excinfo.value.__cause__
    assert isinstance(cause, TrainingError)
    assert str(cause) == "root cause"
