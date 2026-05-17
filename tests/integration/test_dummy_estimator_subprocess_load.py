"""Fresh-subprocess load (Phase 6a, N1).

Fit + save in this process, then reload in a brand-new Python process
and re-predict; predictions must be byte-equal. This is the test that
genuinely needs the fitted ``TabularToSequence`` (encoder vocab, scaler
stats) persisted in ``state.json`` rather than re-derived. The second
test mutates the persisted ``seq_sklearn_version`` and reloads through
the estimator's ``load`` classmethod (not just the Phase-3 helper),
asserting exactly one version-mismatch ``UserWarning``.
"""

import json
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from seq_sklearn.config._adapters import SchedulerParams, TabularConfigParams
from tests._test_models._dummy_estimator import _DummySequenceClassifier

_REPO = Path(__file__).resolve().parents[2]


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _panel() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(3)
    rows: list[dict[str, object]] = []
    y: list[int] = []
    for e in range(20):
        label = e % 2
        for t in range(6):
            rows.append(
                {
                    "id": e,
                    "time": pd.Timestamp("2021-01-01") + pd.offsets.Day(t),
                    "tr": float(label) + 0.2 * rng.normal(),
                }
            )
            y.append(label)
    return pd.DataFrame(rows), np.asarray(y)


def _estimator() -> _DummySequenceClassifier:
    return _DummySequenceClassifier(
        task_type="binary",
        tabular_config=TabularConfigParams(
            time_varying_real_cols=("tr",), lookback=3, min_periods=1, min_periods_predict=1
        ),
        scheduler=SchedulerParams(name="constant", warmup_steps=0),
        max_epochs=2,
        batch_size=16,
        val_fraction=0.2,
        cal_fraction=0.0,
        precision="32-true",
        verbose=False,
        seed=42,
    )


_SUBPROCESS = textwrap.dedent(
    """
    import sys, numpy as np, pandas as pd
    model_dir, x_pkl, out_npy = sys.argv[1:4]
    from tests._test_models._dummy_estimator import _DummySequenceClassifier
    est = _DummySequenceClassifier.load(model_dir)
    x = pd.read_pickle(x_pkl)
    np.save(out_npy, est.predict_proba(x))
    """
)


def test_subprocess_reload_byte_equal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel()
    est = _estimator().fit(x, y)
    before = est.predict_proba(x)

    model_dir = tmp_path / "model"
    est.save(model_dir)
    x_pkl = tmp_path / "x.pkl"
    x.to_pickle(x_pkl)
    out_npy = tmp_path / "preds.npy"

    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS, str(model_dir), str(x_pkl), str(out_npy)],
        capture_output=True,
        text=True,
        cwd=_REPO,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    after = np.load(out_npy)
    assert np.array_equal(after, before)


def test_load_version_mismatch_warning_via_estimator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import seq_sklearn

    _force_cpu(monkeypatch)
    x, y = _panel()
    est = _estimator().fit(x, y)
    model_dir = tmp_path / "model"
    est.save(model_dir)

    state_path = model_dir / "state.json"
    state = json.loads(state_path.read_text())
    state["seq_sklearn_version"] = "0.0.0-ancient"
    state_path.write_text(json.dumps(state))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _DummySequenceClassifier.load(model_dir)
    mismatch = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(mismatch) == 1
    msg = str(mismatch[0].message)
    assert "0.0.0-ancient" in msg
    assert seq_sklearn.__version__ in msg
