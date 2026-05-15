# PyTorch 2026 best practices for seq-sklearn

Scope: pin choices, behavior changes, and library hooks that affect the v1 TFT
classifier/regressor, training, ONNX export, and reproducibility.

## Source citations

- PyTorch 2.12 release announcement, key dates: https://dev-discuss.pytorch.org/t/pytorch-release-2-12-key-dates/3329
- PyTorch on PyPI (version index): https://pypi.org/project/torch/
- `torch.load` weights_only flip (2.6), BC-breaking notice: https://dev-discuss.pytorch.org/t/bc-breaking-change-torch-load-is-being-flipped-to-use-weights-only-true-by-default-in-the-nightlies-after-137602/2573
- `torch.serialization.add_safe_globals` docs (linked from the flip notice).
- HF accelerate weights_only fallout: https://github.com/huggingface/accelerate/issues/3539
- `torch.nn.attention.sdpa_kernel` (current API, 2.12): https://docs.pytorch.org/docs/2.12/generated/torch.nn.attention.sdpa_kernel.html
- `torch.backends.cuda.sdp_kernel` deprecation tracker: https://github.com/fpgaminer/joytag/issues/11
- SDPA tutorial (backend selection, math fallback): https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
- SDPA fails to export to ONNX: https://github.com/pytorch/pytorch/issues/135615
- Reproducibility docs (2.12): https://docs.pytorch.org/docs/stable/notes/randomness.html
- `torch.use_deterministic_algorithms` op list (2.12): https://docs.pytorch.org/docs/2.12/generated/torch.use_deterministic_algorithms.html
- Mixed precision guidance: https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/
- Lightning N-Bit precision (2.6.1): https://lightning.ai/docs/pytorch/stable/common/precision_intermediate.html
- Lightning `torch.compile` guide: https://lightning.ai/docs/pytorch/stable/advanced/compile.html
- Dynamic shapes in `torch.compile`: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html
- ONNX export with `dynamo=True` (2.12 tutorial): https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
- `torch.export`-based ONNX exporter reference: https://docs.pytorch.org/docs/stable/onnx_export.html
- Migrating to `torch.func`: https://docs.pytorch.org/docs/stable/func.migrating.html
- `pack_padded_sequence` reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
- `MultiheadAttention` reference (2.11): https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html

## Version pin recommendation

PyTorch 2.12.0 shipped 2026-05-13 and is the current stable
(https://dev-discuss.pytorch.org/t/pytorch-release-2-12-key-dates/3329,
https://pypi.org/project/torch/). 2.12 supports CPython 3.10 through 3.14.

Pin `torch >= 2.6, < 3` for seq-sklearn v1:

- 2.6 is the floor where `weights_only=True` is the default for `torch.load`. Building the save/load format against that floor avoids forking behavior across user environments (https://dev-discuss.pytorch.org/t/bc-breaking-change-torch-load-is-being-flipped-to-use-weights-only-true-by-default-in-the-nightlies-after-137602/2573).
- 2.5 introduced the production `torch.onnx.export(..., dynamo=True)` path that we want for ONNX export (https://github.com/huggingface/optimum/issues/2026).
- The `torch.nn.attention.sdpa_kernel` API and `SDPBackend` enum are stable from 2.3+ and supported through 2.12 (https://docs.pytorch.org/docs/2.12/generated/torch.nn.attention.sdpa_kernel.html).

CI should run against 2.6 (floor), 2.8 (LTS-style midpoint), and 2.12 (head).

## `weights_only` policy

PyTorch 2.6 flipped `torch.load`'s default to `weights_only=True`. Loading any
pickle-embedded Python object (optimizer states with custom classes, RNG state
holders, dataclasses) now raises `UnpicklingError` unless the class is on the
safe-globals allowlist (https://dev-discuss.pytorch.org/t/bc-breaking-change-torch-load-is-being-flipped-to-use-weights-only-true-by-default-in-the-nightlies-after-137602/2573,
https://github.com/huggingface/accelerate/issues/3539).

Policy for seq-sklearn:

1. Weights file is `.safetensors` (tensors only, no pickle). See serialization section.
2. Metadata file is JSON sidecar (config, feature schema, fit-state scalars). Never a pickle.
3. If we must call `torch.load` for round-trip of RNG state or optimizer state during resumed training, set `weights_only=True` and register required classes via `torch.serialization.add_safe_globals([...])` at module import. Never default to `weights_only=False`. A trusted-source escape hatch can exist behind an explicit flag, off by default.

This keeps the save/load contract identical across torch 2.6, 2.8, and 2.12.

## SDPA backend selection and ONNX export

Three backends remain in 2.12: `FLASH_ATTENTION`, `EFFICIENT_ATTENTION` (mem-efficient),
`MATH`, plus `CUDNN_ATTENTION` on newer hardware
(https://docs.pytorch.org/docs/2.12/generated/torch.nn.attention.sdpa_kernel.html).
Of these, only `MATH` decomposes into ops the ONNX exporter understands cleanly. The
fused kernels lower to opaque CUDA calls that the exporter cannot trace
(https://github.com/pytorch/pytorch/issues/135615).

API migration: `torch.backends.cuda.sdp_kernel(...)` is deprecated in favor of
`torch.nn.attention.sdpa_kernel(...)` with the `SDPBackend` enum
(https://github.com/fpgaminer/joytag/issues/11,
https://docs.pytorch.org/docs/2.12/generated/torch.nn.attention.sdpa_kernel.html).
The new form takes a single backend or a list, and supports `set_priority=True`
for ordered fallback.

Recommended pattern for seq-sklearn:

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

# Training/inference: let the dispatcher pick.
# ONNX export only:
with sdpa_kernel([SDPBackend.MATH]):
    torch.onnx.export(model, sample, "model.onnx", dynamo=True)
```

Wrap the export call in the `MATH`-only context. Never globally disable the fused
backends; that costs throughput on training and live inference paths.

## Deterministic mode current status

Settings, in order of increasing strictness:

- `torch.backends.cudnn.benchmark = False`: stops cuDNN from picking different convolution algorithms run-to-run (https://docs.pytorch.org/docs/stable/notes/randomness.html).
- `torch.backends.cudnn.deterministic = True`: restricts cuDNN to deterministic conv algorithms only.
- `torch.use_deterministic_algorithms(True, warn_only=False)`: global switch. Some ops become deterministic, some raise `RuntimeError` when called.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or `:16:8`) environment variable, required on CUDA >= 10.2 to make cuBLAS matmuls reproducible (https://docs.pytorch.org/docs/stable/notes/randomness.html).

2.12 op coverage (from https://docs.pytorch.org/docs/2.12/generated/torch.use_deterministic_algorithms.html):

- Become deterministic under the switch: `index_add`, `scatter_add_`, `scatter_reduce` with `sum`/`mean`, `index_copy`, `scatter`, `gather`/`index_select`/`repeat_interleave` backward, conv forward/backward, sparse-dense bmm.
- Raise `RuntimeError`: `scatter_reduce` with `reduce='prod'`, `EmbeddingBag` backward with `mode='max'`, several pool/interpolate backward paths on CUDA, `NLLLoss` on CUDA, `CTCLoss` backward on CUDA, `MaxUnpool*`.
- LSTM: cuDNN LSTM has historically been nondeterministic on CUDA; the workaround documented in `torch.nn.LSTM` is to set `CUBLAS_WORKSPACE_CONFIG` and accept the cuDNN deterministic path or fall back to the non-cuDNN implementation (https://docs.pytorch.org/docs/stable/notes/randomness.html). TFT uses LSTM blocks, so this is load-bearing.

seq-sklearn policy: a `set_deterministic(strict: bool)` helper that

1. seeds Python, NumPy, and Torch RNGs,
2. sets cuDNN flags,
3. exports `CUBLAS_WORKSPACE_CONFIG=:4096:8` if not already set,
4. calls `torch.use_deterministic_algorithms(True, warn_only=not strict)`,
5. documents the LSTM caveat in the API docstring.

`scatter_reduce(prod)` is not used in TFT blocks; we can keep `warn_only=False`
by default if we audit our op surface. `EmbeddingBag` is not in TFT either
(we use `Embedding`, which is deterministic). The variable-selection network uses
`scatter_*` for nothing we have authored; quantile loss uses elementwise ops only.

## Mixed precision recommendations 2026

For Ampere and newer (A100, H100, RTX 30xx+, MI200+, L4), prefer `bf16-mixed`:

- bf16 matches fp32 dynamic range, so `GradScaler` is not needed (https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/).
- Tensor-core throughput on Ampere+ is the same for bf16 and fp16 (same source).
- Quantile loss and the softmax/attention paths in TFT have wide value ranges; bf16 avoids underflow that fp16 routinely hits there.

For pre-Ampere (V100, T4, RTX 20xx), bf16 is software-emulated and slow.
Fall back to `16-mixed` with `GradScaler` (https://lightning.ai/docs/pytorch/stable/common/precision_intermediate.html).

Lightning precision values map directly: `precision="bf16-mixed"`,
`precision="16-mixed"`, `precision="32-true"`. seq-sklearn exposes these via the
training config and defaults to `bf16-mixed` when CUDA is available and the
device compute capability is >= 8.0, else `32-true`. Never default to `16-mixed`;
the silent NaN risk on TFT-shaped models is real
(https://github.com/pytorch/pytorch/issues/119131).

## `torch.compile` viability for seq-sklearn

Verdict: support it behind an opt-in flag, do not default to it for v1.

- Lightning integrates cleanly: `model = torch.compile(model)` before `trainer.fit(...)` (https://lightning.ai/docs/pytorch/stable/advanced/compile.html).
- Dynamic shapes work but are the source of most real-world breakage. Seq-sklearn has two dynamic axes: batch size and sequence length. Use `torch._dynamo.mark_dynamic(x, 0)` for batch and the time dim, or `dynamic=True` on the compile call (https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html).
- Pydantic v2 frozen models: not stored on `nn.Module` state, so `torch.compile` does not see them at graph capture. Safe.
- Variable-length attention masks recompile per unique mask shape unless the time dim is marked dynamic. Mark it dynamic.

Recommended config knob: `compile: bool | dict[str, Any] = False`. When truthy, run `torch.compile(model, dynamic=True, fullgraph=False, **opts)`. Document that interpretability hooks (attention extraction) require `fullgraph=False` so the user can break out the introspection ops.

## ONNX export path

2.12 promotes `torch.onnx.export(..., dynamo=True)` as the supported path
(https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html,
https://docs.pytorch.org/docs/stable/onnx_export.html). The standalone
`torch.onnx.dynamo_export(...)` entry point is still labeled preview and is
folded into `export(dynamo=True)` for production use
(https://github.com/huggingface/optimum/issues/2026).

Path choice for seq-sklearn:

- Primary: `torch.onnx.export(model, sample, path, dynamo=True, opset_version=20)`.
- Wrap with `sdpa_kernel([SDPBackend.MATH])` so the attention op decomposes.
- Mark batch and time dims dynamic via `dynamic_shapes=` (the dynamo path uses `torch.export.Dim`).
- Fallback: legacy `dynamo=False` path for environments stuck on opset <= 17 or for ops the dynamo exporter still chokes on. The TFT-relevant trouble spots are the gated residual blocks (clean), GLU (clean), `LayerNorm` (clean), `MultiheadAttention` (only via `MATH` SDPA, otherwise opaque), and `LSTM` (works on both paths but watch sequence-length dynamism).

Opset target: 20. TensorRT and ORT 1.18+ accept it, and opset >= 18 is required for several `torch.export`-emitted ops (https://discuss.pytorch.org/t/why-does-torch-not-support-onnx-export-with-opset-17-dynamo-export-not-working-properly/211336).

## Variable-length sequence handling

RNN path (LSTM in TFT's locality enhancement):

- Use `pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)` then the LSTM, then `pad_packed_sequence` to recover (B, T, H). `enforce_sorted=False` has been the right setting since 1.1.0 and remains correct in 2.12 (https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html). Sorting is only required if you intend ONNX export through the legacy path; the dynamo path handles unsorted.

Attention path:

- `nn.MultiheadAttention.key_padding_mask`: boolean tensor, `True` means ignore that key (i.e., padding) (https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html).
- `attn_mask` in the same module: same `True = ignore` for boolean masks.
- `F.scaled_dot_product_attention(attn_mask=...)`: opposite for boolean masks. `True = participate`, `False = mask out`. This is the documented divergence between the two APIs (https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html and the MHA docs above).

seq-sklearn convention: internally store masks as `padding_mask` with `True = pad`,
matching `MultiheadAttention`. Convert with `~mask` at the SDPA call site. Add a
unit test that builds a known-shape padded batch and asserts the unpadded
positions get nonzero attention to non-pad keys only.

## Decisions implied for seq-sklearn

1. Pin `torch>=2.6,<3`; CI matrix on 2.6, 2.8, 2.12.
2. Save weights as `.safetensors`. Sidecar JSON for config/schema/fit state. No `torch.save` for the public artifact.
3. Wrap ONNX export in `sdpa_kernel([SDPBackend.MATH])` and use `torch.onnx.export(..., dynamo=True, opset_version=20)`. Mark batch and time dims dynamic.
4. Mixed precision default: `bf16-mixed` on CC >= 8.0 CUDA devices, else `32-true`. Never default to `16-mixed`.
5. Provide a `set_deterministic(strict: bool)` helper that wires all four switches (cuDNN benchmark/deterministic, `use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`).
6. `torch.compile` is opt-in via training config, off by default in v1. When on, pass `dynamic=True`.
7. Internal mask convention: `True = pad`. Convert at the SDPA boundary.
8. Use `pack_padded_sequence(..., enforce_sorted=False)` for the LSTM block.
9. Test isolation: prefer `torch.func.functional_call` over hand-rolled state surgery when a test needs to call a module with synthetic parameters; it is the supported replacement for `make_functional` (https://docs.pytorch.org/docs/stable/func.migrating.html).
