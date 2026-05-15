"""Tests for the exception hierarchy at architecture A10."""

import pickle

import pytest
import sklearn.exceptions as sk_exc

from seq_sklearn.errors import (
    ConfigError,
    DataContractError,
    NotFittedError,
    PredictionError,
    SeqSklearnError,
    TrainingError,
)


def test_seq_sklearn_error_root_is_exception() -> None:
    assert issubclass(SeqSklearnError, Exception)


@pytest.mark.parametrize(
    "exc",
    [ConfigError, DataContractError, TrainingError, PredictionError],
)
def test_subclasses_inherit_from_root(exc: type[Exception]) -> None:
    assert issubclass(exc, SeqSklearnError)


def test_not_fitted_error_dual_parent_mro() -> None:
    """Both library-side and sklearn-side except clauses must catch."""
    assert issubclass(NotFittedError, SeqSklearnError)
    assert issubclass(NotFittedError, sk_exc.NotFittedError)
    # MRO order is load-bearing: library root first, sklearn root second.
    mro = NotFittedError.__mro__
    assert mro.index(SeqSklearnError) < mro.index(sk_exc.NotFittedError)


def test_not_fitted_caught_by_library_root() -> None:
    with pytest.raises(SeqSklearnError):
        raise NotFittedError("not fit")


def test_not_fitted_caught_by_sklearn_root() -> None:
    with pytest.raises(sk_exc.NotFittedError):
        raise NotFittedError("not fit")


def test_not_fitted_pickle_roundtrip() -> None:
    err = NotFittedError("not fit")
    revived = pickle.loads(pickle.dumps(err))
    assert isinstance(revived, NotFittedError)
    assert isinstance(revived, SeqSklearnError)
    assert isinstance(revived, sk_exc.NotFittedError)
    assert str(revived) == "not fit"


@pytest.mark.parametrize(
    "exc",
    [ConfigError, DataContractError, TrainingError, PredictionError],
)
def test_subclass_carries_message(exc: type[SeqSklearnError]) -> None:
    err = exc("specific reason")
    assert "specific reason" in str(err)
