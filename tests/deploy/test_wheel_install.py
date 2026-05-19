"""Wheel-install deploy smoke (Phase 10).

Install the built wheel WITH the `[onnx]` extra in a fresh venv, then
import + fit + predict + import the onnx wrapper in a subprocess. This
catches packaging breakage (missing package data, broken extras
metadata, import errors) that no in-tree test sees. Prefers the
CI-prebuilt wheel via `SEQ_SKLEARN_WHEEL` (Step 3b sets it) and only
builds locally otherwise; trivially small so it stays in the per-PR
`deploy` budget (not marked `slow`).
"""

import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

pytestmark = pytest.mark.deploy

_SMOKE = """
import numpy as np
import seq_sklearn  # noqa: F401
import seq_sklearn.inference.onnx  # noqa: F401
from seq_sklearn.config._adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier

gen = SyntheticPanelGenerator(
    target_kind="binary", num_entities=4, periods_per_entity=6,
    lookback=3, seed=0,
)
panel, y = gen.generate(seed=0)
est = TFTClassifier(
    task_type="binary",
    tabular_config=TabularConfigParams(
        id_col=gen.id_col, time_col=gen.time_col,
        static_categorical_cols=tuple(gen.static_categorical_cols),
        static_real_cols=tuple(gen.static_real_cols),
        time_varying_real_cols=tuple(gen.time_varying_real_cols),
        time_varying_categorical_cols=tuple(gen.time_varying_categorical_cols),
        lookback=gen.lookback, min_periods=1, min_periods_predict=1,
        max_categorical_cardinality=10_000,
    ),
    scheduler=SchedulerParams(name="constant", warmup_steps=0),
    hidden_size=8, attention_heads=2, max_epochs=1, batch_size=16,
    val_fraction=0.25, cal_fraction=0.0, precision="32-true",
    seed=0, verbose=False,
).fit(panel, y)
proba = est.predict_proba(panel)
assert proba.shape == (len(panel), 2), proba.shape
print("WHEEL_SMOKE_OK")
"""


def _build_wheel(tmp_path: Path) -> Path:
    pytest.importorskip("build")
    dist = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist)],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    wheels = sorted(dist.glob("*.whl"))
    assert wheels, "no wheel produced by python -m build"
    return wheels[0]


def test_wheel_installs_and_predicts(tmp_path: Path) -> None:
    env_wheel = os.environ.get("SEQ_SKLEARN_WHEEL")
    wheel = Path(env_wheel) if env_wheel else _build_wheel(tmp_path)
    assert wheel.is_file(), f"wheel not found: {wheel}"

    venv_dir = tmp_path / "venv"
    venv.create(venv_dir, with_pip=True)
    py = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"

    install = subprocess.run(
        [str(py), "-m", "pip", "install", "--quiet", f"{wheel}[onnx]"],
        capture_output=True,
        text=True,
    )
    if install.returncode != 0:
        pytest.fail(f"pip install <wheel>[onnx] failed:\n{install.stderr}")

    smoke = subprocess.run([str(py), "-c", _SMOKE], capture_output=True, text=True)
    if smoke.returncode != 0 or "WHEEL_SMOKE_OK" not in smoke.stdout:
        pytest.fail(
            f"wheel smoke failed (rc={smoke.returncode}):\n"
            f"STDOUT:\n{smoke.stdout}\nSTDERR:\n{smoke.stderr}"
        )
