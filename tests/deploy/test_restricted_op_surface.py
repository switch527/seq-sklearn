"""Restricted ONNX op-surface guard (Phase 10, architecture A21 / R2).

`ONNX_SAFE_OPS` is the architecture-enumerated allowlist (copied
VERBATIM from docs/architecture.md "A21: ONNX restricted op surface",
which derives it by static analysis of the backbone/head/attention
modules, NOT from a sample export). The exported graph's op_type set
(recursing subgraphs) must be a subset. An export emitting anything
outside it (e.g. `ScaledDotProductAttention`, `Loop`, `If`) fails
here at first pin; that is the R2 regression guard. Growing the set
requires a deliberate edit here AND in the A21 subsection.
"""

import pytest

pytest.importorskip("onnx")
import onnx

from seq_sklearn.config._adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier

pytestmark = [pytest.mark.deploy, pytest.mark.onnx]

# VERBATIM from docs/architecture.md "A21: ONNX restricted op surface".
ONNX_SAFE_OPS: frozenset[str] = frozenset(
    {
        "Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "MatMul", "Gemm",
        "LSTM", "Softmax", "Sigmoid", "Tanh", "Elu", "Erf", "ReduceMean",
        "ReduceSum", "LayerNormalization", "Gather", "GatherElements",
        "ScatterElements", "ArgMax", "Range", "Greater", "GreaterOrEqual",
        "Less", "Where", "Expand", "Cast", "Concat", "Slice", "Reshape",
        "Transpose", "Squeeze", "Unsqueeze", "Shape", "Constant",
        "ConstantOfShape", "Identity", "Flip", "Neg", "Equal", "And",
        "Not", "IsNaN", "IsInf", "Mod", "Split", "GatherND",
    }
)  # fmt: skip


def _collect_op_types(graph: object) -> set[str]:
    """All node op_types in a graph, recursing control-flow subgraphs."""
    ops: set[str] = set()
    for node in graph.node:  # type: ignore[attr-defined]
        ops.add(node.op_type)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                ops |= _collect_op_types(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    ops |= _collect_op_types(g)
    return ops


def test_exported_graph_uses_only_safe_ops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    gen = SyntheticPanelGenerator(
        target_kind="binary",
        num_entities=12,
        periods_per_entity=10,
        signal_strength=0.8,
        lookback=5,
        seed=42,
    )
    panel, y = gen.generate(seed=42)
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
        hidden_size=16,
        attention_heads=2,
        max_epochs=2,
        batch_size=32,
        val_fraction=0.2,
        cal_fraction=0.0,
        precision="32-true",
        seed=42,
        verbose=False,
    ).fit(panel, y)

    out = str(tmp_path / "ops.onnx")  # type: ignore[operator]
    est.export_onnx(out, panel)
    model = onnx.load(out)
    used = _collect_op_types(model.graph)
    extra = used - ONNX_SAFE_OPS
    assert not extra, (
        f"exported graph emits ops outside the A21 restricted surface: "
        f"{sorted(extra)}. If intentional, update docs/architecture.md "
        f"A21 AND ONNX_SAFE_OPS here in one reviewed edit."
    )
