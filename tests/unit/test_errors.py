"""Tests for the exception hierarchy at architecture A10."""

import importlib
import pickle

import pytest
import sklearn.exceptions as sk_exc

from seq_sklearn.errors import (
    ConfigError,
    DataContractError,
    NotFittedError,
    OnnxExportError,
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


def test_onnx_export_error_is_root_subclass() -> None:
    assert issubclass(OnnxExportError, SeqSklearnError)
    assert "boom" in str(OnnxExportError("boom"))


def test_export_onnx_raises_without_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 10 Step 0b: the [onnx]-missing guard, environment-independent.

    Forces ``import onnx`` to fail even when the extra IS installed
    (so this runs in the default test-unit CI) and pins the
    OnnxExportError message + the chained ImportError __cause__.
    """
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    from seq_sklearn.config._adapters import SchedulerParams, TabularConfigParams
    from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
    from seq_sklearn.models.transformer.tft.classifier import TFTClassifier

    gen = SyntheticPanelGenerator(
        target_kind="binary",
        num_entities=4,
        periods_per_entity=6,
        lookback=3,
        seed=0,
    )
    panel, y = gen.generate(seed=0)
    est = TFTClassifier(
        task_type="binary",
        tabular_config=TabularConfigParams(
            id_col=gen.id_col,
            time_col=gen.time_col,
            static_categorical_cols=tuple(gen.static_categorical_cols),
            static_real_cols=tuple(gen.static_real_cols),
            time_varying_real_cols=tuple(gen.time_varying_real_cols),
            time_varying_categorical_cols=tuple(gen.time_varying_categorical_cols),
            lookback=gen.lookback,
            min_periods=1,
            min_periods_predict=1,
            max_categorical_cardinality=10_000,
        ),
        scheduler=SchedulerParams(name="constant", warmup_steps=0),
        hidden_size=8,
        attention_heads=2,
        max_epochs=1,
        batch_size=16,
        val_fraction=0.25,
        cal_fraction=0.0,
        precision="32-true",
        seed=0,
        verbose=False,
    ).fit(panel, y)

    real_import = importlib.import_module

    def _no_onnx(name: str, *a: object, **k: object) -> object:
        if name in ("onnx", "onnxruntime"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib, "import_module", _no_onnx)

    with pytest.raises(OnnxExportError, match=r"pip install seq-sklearn\[onnx\]") as ei:
        est.export_onnx("/tmp/should_not_be_written.onnx", panel)
    assert isinstance(ei.value.__cause__, ImportError)
