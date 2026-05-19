"""Runnable seq-sklearn quickstart (the file the README quickstart mirrors).

`tests/e2e/test_quickstart.py` imports `run_quickstart` and asserts the
N1 binary-classifier threshold (accuracy >= 0.75) at `seed=42`,
`dgp_version=1` per architecture A14, so the documented example and the
CI assertion never drift.

The README snippet is the legible shape; this is the executable form:
the panel and split are built from the F6 synthetic generator so the
example is self-contained and reproducible.
"""

from sklearn.metrics import accuracy_score

from seq_sklearn.config._adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier


def run_quickstart(seed: int = 42) -> float:
    """Fit a TFT binary classifier on a synthetic panel; return accuracy.

    Deterministic for a fixed ``seed`` on the CPU FP32 path (the panel
    is pinned to ``dgp_version=1``, ``signal_strength=0.7`` per N1).
    """
    gen = SyntheticPanelGenerator(
        target_kind="binary",
        num_entities=160,
        periods_per_entity=24,
        signal_strength=0.9,
        lookback=12,
        seed=seed,
    )
    panel, y = gen.generate(seed=seed)

    tabular_config = TabularConfigParams(
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
    )
    clf = TFTClassifier(
        task_type="binary",
        tabular_config=tabular_config,
        # Default cosine_with_warmup converges far better than a flat
        # constant LR in a bounded epoch budget.
        scheduler=SchedulerParams(name="cosine_with_warmup", warmup_steps=50),
        hidden_size=64,
        attention_heads=4,
        max_epochs=60,
        batch_size=64,
        val_fraction=0.2,
        cal_fraction=0.0,
        precision="32-true",
        verbose=False,
        seed=seed,
    )
    clf.fit(panel, y)
    preds = clf.predict(panel)
    return float(accuracy_score(y, preds))


if __name__ == "__main__":  # pragma: no cover - manual run
    acc = run_quickstart()
    print(f"quickstart binary accuracy: {acc:.3f}")
