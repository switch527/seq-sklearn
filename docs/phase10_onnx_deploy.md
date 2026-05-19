# Phase 10 plan: ONNX export + deploy tests (CONSENSUS, R4)

Point-for-point implementation spec for Phase 10 (implementation_plan
"### Phase 10: ONNX export + deploy tests"), grounded in the current
code. Claude dual-model swarm consensus REACHED after 4 rounds (see
the ledger at the bottom). R1 caught the export-mechanism risk; R2 a
wrong LSTM-fallback numerics claim, a phantom op-surface source, an
unspecified import-guard test, a dead CI deploy gate; R3 the
fitted-only parity oracle; R4 all-APPROVE. Implement Steps 0..8;
then code `/review` consensus; then Gemini final-pass.

## Invariants

- I-1: `export_onnx(path, X)` is a public method on
  `BaseSequenceEstimator` (`models/_base.py`), inherited by both
  families; `X` is REQUIRED (matches `predict`); raises
  `NotFittedError` before fit; raises `OnnxExportError` with an
  actionable message if the optional `[onnx]` extra is absent.
- I-2: export is the two-step `torch.export.export(..., strict=
  False)` then `torch.onnx.export(ep, ..., dynamo=True,
  opset_version=20, external_data=False)` (see Export Mechanism).
- I-3: the exported graph is the model FORWARD only (backbone + head
  raw logits `(B, K)`). Calibration, thresholding, below-floor
  NaN-fill, caller-order restore are sklearn-side numpy and NOT in
  the graph; same torch/numpy seam `_raw_outputs` / `_predict_raw`
  already draw. Surfaced in the `export_onnx` docstring and the
  requirements.md:249 BETA row, not only here.
- I-4: parity within `atol=1e-4, rtol=1e-4` on a fixed batch with a
  masked variable-length entity; the wrapper output equals the
  production sorted-space raw logits `est._predict_raw(X)[0]` (the
  oracle is `_predict_raw`, NOT `_raw_outputs`: `_predict_raw` routes
  through `_forward_backbone` -> `_predict_module()` which is
  loaded-aware (`_base.py:613/629/580-584`), whereas `_raw_outputs`
  (`_base.py:422-428`) dereferences `self._module` and raises
  `AttributeError` on a `load()`-ed estimator; both `_OnnxForward`
  and `_predict_raw[0]` are in transform/sorted order so they are
  directly comparable); ONNX logits + the documented post-processing
  reproduce `est.predict_proba` (above-floor rows match; a
  below-floor entity is NaN in `predict_proba`).
- I-5: dynamic batch `B` only. Time dim is the fit-constant
  `lookback` (`tabular_to_sequence.py:511-522`); `L` is static.
- I-6: deploy tests pass against a freshly built wheel in CI with
  the `[onnx]` extra installed; the backbone emits only ops in the
  architecture-enumerated restricted op surface (Step 0b).

## Export mechanism (R1 C1, corrected per R2)

`torch.onnx.export(dynamo=True)` lowers through `torch.export`.
`TFTBackbone.forward` (`models/transformer/tft/backbone.py:187`) and
helpers contain data-dependent guards that raise
`GuardOnDataDependentSymNode` under strict tracing:
`backbone.py:151` (left-pad contract, `_run_lstm`), `:201`
(non-finite guard), `:216` (embedding-index bounds), `:224`
(all-padding-row guard); plus `:162-163`
`pack_padded_sequence(gathered, lengths.cpu(), ...)`. (`.item()` at
`:291/:295` is in `compute_training_metrics`, not the forward path.)

MECHANISM (mandatory two-step):
1. `ep = torch.export.export(module, args, dynamic_shapes={B
   dynamic}, strict=False)`. `strict=False` specializes the
   data-dependent boolean guards to the trace: on a VALID example
   batch each guard is not-taken, so it is erased from the exported
   graph. SOUND because these are eager input-validation checks with
   no place in a serialized inference graph (ONNX has no exception
   mechanism; validation stays the responsibility of the eager
   `predict` path and the serving layer). The `export_onnx`
   docstring states the exported graph trusts its inputs and omits
   the eager validation guards.
2. `torch.onnx.export(ep, args, str(path), dynamo=True,
   opset_version=20, external_data=False)`.

LSTM HANDLING (R2 CRITICAL B, corrected). `_run_lstm`
(`backbone.py:137-176`) ALREADY gathers each row valid-first via
`order = argsort((~valid).int(), dim=1, stable=True)` (`:156-158`),
runs `pack_padded_sequence -> lstm -> pad_packed_sequence` on the
GATHERED (valid-first) tensor (`:162-167`), then scatters back via
`inverse = argsort(order)` (`:172-175`). The pack/unpack is purely
an efficiency wrapper over the already-gathered sequence. The
export-only fallback in `_OnnxForward` (used only if
`pack_padded_sequence` does not lower under dynamo) reproduces
`_run_lstm` EXACTLY except it replaces
`pack_padded_sequence(gathered, lengths.cpu()) -> self.lstm ->
pad_packed_sequence` with a plain `self.lstm(gathered, (h_0, c_0))`
on the same valid-first gathered tensor, keeping the identical
`order` gather and `inverse` scatter. This is numerically identical
at every VALID position because the LSTM is causal left-to-right:
on the gathered sequence the valid steps are a contiguous PREFIX and
the trailing pad steps cannot affect the recurrent state of earlier
(valid) steps; `pack_padded_sequence` merely stops the recurrence
early, producing the same hidden states for the valid prefix. The
extra trailing-pad outputs are discarded by the `inverse` scatter
mapping them to masked positions, which both readouts ignore
(`_readout` last-valid index `backbone.py:184-185`; `mean_pool`
averages over `valid` only). It is NOT a full-left-padded-sequence
LSTM (that WOULD contaminate state with leading pad; the R2
finding); it is the same valid-first gathered sequence minus only
the pack wrapper. The Step 4 parity test (`atol=1e-4`, AND a
loaded-estimator case) is the objective gate; the
restricted-op-surface test (Step 5) fails loudly if anything outside
the enumerated surface is emitted.

## Key reconciliation: SDPA pin vs interpretable path

`TFTBackbone.forward` computes attention at
`models/transformer/tft/backbone.py:247` via
`self.attention.forward_interpretable(attn_in, mask)`
(`models/transformer/_interpretable_attention.py:90-96`), explicit
matmul -> masked_fill -> softmax -> matmul, NO
`scaled_dot_product_attention`. The SDPA fast path
(`_interpretable_attention.py:62-77`) is not on the export path.
Manual softmax is inherently ONNX-safe; the flash/mem-efficient
hazard the requirement warns about does not apply. RESOLUTION
(swarm to confirm): keep the interpretable path (the trained/
validated numeric path; snapshots + `predict_with_attention` depend
on it); wrap export in `with sdpa_kernel([SDPBackend.MATH]):` as
defense-in-depth (`from torch.nn.attention import SDPBackend,
sdpa_kernel`); the Step 0b op-surface enumeration + Step 5 test are
the real enforcement. Step 7b reconciles all three doc sites
(requirements.md:389-394, requirements.md:1702-1712 the N1 ONNX-
parity paragraph, implementation_plan.md:1399-1406 R2) to this
grounded reality (no behavior change; surfaced, not silent).

## Step 0 - errors.py + the import-guard test (R2 CRITICAL C)

0a. `src/seq_sklearn/errors.py`: add `class
OnnxExportError(SeqSklearnError)` AND add `"OnnxExportError"` to the
existing `__all__` (present at `errors.py:12-19`; mandatory, not
conditional).

0b. `src/seq_sklearn/inference/onnx.py` test: pin the import-guard
ENVIRONMENT-INDEPENDENTLY (not behind any `[onnx]` importorskip):
new `tests/unit/test_errors.py::test_export_onnx_raises_without_
extra` (or a unit test under `tests/unit/inference/`): fit a tiny
estimator, then with `onnx`/`onnxruntime` forced absent
(`monkeypatch`/`sys.modules` shadowing so `import onnx` raises
`ImportError` even when the extra IS installed), call
`export_onnx`; assert `OnnxExportError` is raised, the message
contains `"pip install seq-sklearn[onnx]"`, and
`exc.__cause__ is not None` (the `raise ... from e`). This runs in
the default `test-unit` CI (no extra needed) so the guard cannot
silently break.

## Step 0b - architecture.md restricted op-surface enumeration
## (R2 CRITICAL A)

requirements.md:1709-1712 explicitly delegates: "The architecture
phase enumerates the restricted PyTorch op surface the backbone is
allowed to use; ops outside the surface are caught by a
static-analysis check in the deploy job." This was never
discharged; there is no enumerated set anywhere (grep-confirmed).
Phase 10, when ONNX export is built, is the correct time to
discharge it.

Add an architecture.md subsection (under A20 / a new A21 "ONNX
restricted op surface") that ENUMERATES, by STATIC ANALYSIS of the
backbone/head/attention module source (NOT by running an export),
the ONNX op_types the exported graph is permitted to emit. Derived
from the actual modules (`tft/backbone.py`,
`_interpretable_attention.py`, the GRN/VSN/GLU blocks, `nn.LSTM`,
`nn.Linear`, `nn.LayerNorm`, `nn.Embedding`, the head):

`{"Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "MatMul", "Gemm",
"LSTM", "Softmax", "Sigmoid", "Tanh", "Elu", "Erf", "ReduceMean",
"ReduceSum", "LayerNormalization", "Gather", "GatherElements",
"ScatterElements", "ArgMax", "Range", "Greater", "GreaterOrEqual",
"Less", "Where", "Expand", "Cast", "Concat", "Slice", "Reshape",
"Transpose", "Squeeze", "Unsqueeze", "Shape", "Constant",
"ConstantOfShape", "Identity", "Flip", "Neg", "Equal", "And",
"Not", "IsNaN", "TopK"}`

Non-obvious entries (one-line static-analysis provenance, mirrored
as a comment in the architecture.md subsection so each entry is
auditable, not re-derived per reviewer):
- `IsNaN`/`Where`: `torch.nan_to_num` at
  `_interpretable_attention.py:93`. This is UNCONDITIONAL on the
  export path (it runs every `forward_interpretable` call, not
  gated by any A6-style reachability condition), so the op is
  always emitted and must be in the set; `IsNaN` was added in R3 to
  forestall a guaranteed Step-5 red-then-amend.
- `GatherElements`: the rank-3 `torch.gather` in `_run_lstm`
  (`backbone.py:159-161` valid-first gather, `:173-174` inverse
  scatter); the gather-preserving fallback (Step 1 GATE CONDITION)
  keeps both, so `GatherElements`/`ScatterElements` are
  unconditional on the export path.
- `TopK`/`ArgMax`: `torch.argsort` (`backbone.py:156,172`) and
  `argmax` (`backbone.py:184`); the dynamo exporter lowers
  `argsort` via the TopK/ArgMax family, so both are kept with this
  rationale rather than dropped.
- `Flip`: `valid.flip(dims=[1])` in `_readout` (`backbone.py:184`).
- `Range`/`Greater*`: `torch.arange` + the `>=` comparison in
  `_run_lstm` (`backbone.py:149-151`), erased only if the strict=
  False guard-specialization removes them; kept defensively.

This concrete set IS the swarm's review target (the agents check it
against the module source; the real independent check, not running
an export). The implementer copies the architecture.md set verbatim
into Step 5's test; any export op outside it fails the test. The set
grows only by a deliberate, reviewed architecture.md + test edit.
NOTE: `ScaledDotProductAttention`, `Attention`, `Loop`, `If`,
`Scan`, `NonZero` are DELIBERATELY EXCLUDED; their appearance is
exactly the R2 regression Step 5 must catch.

Blast radius: `docs/architecture.md` (one new subsection). This is a
design-doc addition the requirements explicitly delegate to the
architecture phase; it is reviewed by this plan's swarm and the
later code `/review` + Gemini pass.

## Step 1 - the ONNX-traceable wrapper (`inference/onnx.py`)

New file `src/seq_sklearn/inference/onnx.py`:
`class _OnnxForward(nn.Module)`:
- `__init__(self, backbone, head)`: store, `.eval()`.
- `forward(self, static_categorical, static_real,
  time_varying_real, time_varying_categorical, padding_mask)
  -> Tensor`: reassemble the dict with exactly these 5 keys (the
  backbone forward reads only these; `entity_id` not read on the
  forward path, `target`/`input_row_order` post-hoc; verified
  against `backbone.py:187` body), return
  `head(backbone(batch).representation)` raw logits `(B, K)`,
  K-agnostic, no calibration.
- GATE CONDITION (pins the fallback so it is never dead/untested):
  `_OnnxForward` ALWAYS uses the gather-preserving plain-LSTM path
  (the corrected Export-mechanism fallback: keep `_run_lstm`'s exact
  `order` gather + `inverse` scatter, replace only
  `pack_padded_sequence -> lstm -> pad_packed_sequence` with
  `self.lstm(gathered, (h_0,c_0))`). It does NOT branch on whether
  `pack_padded_sequence` lowers. Consequence: Step 4 (a) at
  `atol=1e-6` (`_OnnxForward` vs the production packed
  `_predict_raw[0]`) IS the objective numeric proof that the
  gather-preserving fallback equals the packed production path, on
  every run, for binary/multiclass/quantile and the loaded
  estimator. The eager TRAINED forward (`TFTBackbone._run_lstm`) is
  untouched; only the export wrapper takes this path.

Blast radius: 1 new file.

## Step 2 - `export_onnx` on BaseSequenceEstimator

`src/seq_sklearn/models/_base.py`, new public method on
`BaseSequenceEstimator` (family-agnostic; nothing family-specific,
no calibrator, no attention_weights leaks):
`def export_onnx(self, path: str | Path, X: pd.DataFrame) -> None`:
1. `self._check_fitted()` (raises `NotFittedError`).
2. `try: import onnx, onnxruntime  # noqa: F401
   except ImportError as e: raise OnnxExportError("ONNX export
   requires the optional extra: pip install seq-sklearn[onnx]")
   from e`.
3. `backbone, head = self._predict_module()` (the ONLY accessor;
   `_base.py:580-584`; works for fitted AND `load()`-ed estimators,
   which never set `self._module`). Never `self._module.backbone`.
4. `batch = self.transformer_.transform(X)`; take the 5 wrapper
   inputs.
5. `module = _OnnxForward(backbone, head)`.
6. `B = torch.export.Dim("batch")` on dim 0 of all 5 inputs; time
   dim static `lookback`; per-feature counts static.
7. `with sdpa_kernel([SDPBackend.MATH]): ep =
   torch.export.export(module, args, dynamic_shapes=...,
   strict=False); torch.onnx.export(ep, args, str(path),
   dynamo=True, opset_version=20, external_data=False)`.
8. Docstring: graph = raw head logits only (no calibration/
   threshold/caller-order; sklearn-side); graph trusts inputs
   (eager validation erased); `X` required; BETA.

Blast radius: `_base.py` (one method) + Step 1 import.

## Step 3 - extras, markers, `build` dep, and the CI gate
## (R2 CRITICAL D)

3a. pyproject.toml ALREADY has `onnx = ["onnx>=1.18",
"onnxruntime>=1.21"]` and the registered `deploy`/`onnx` markers.
ADD `build` to the `dev` optional extra (Step 6 needs
`python -m build`; not currently declared).

3b. `.github/workflows/pr.yml` `test-deploy` job is currently a
DEAD gate for Phase 10: it does `uv build --wheel`, installs the
wheel into a throwaway `.venv-deploy`, then `uv run pytest
tests/deploy/ -m "deploy"` in the PROJECT env that has NEITHER the
`[onnx]` extra NOR `build` (no `uv sync --extra ...`), so every
Phase 10 deploy/onnx test `importorskip`s away. Fix the job:
- `uv sync --extra dev --extra onnx` (so `onnx`, `onnxruntime`,
  `build` are present in the env `uv run pytest` uses).
- `uv build --wheel` then `export SEQ_SKLEARN_WHEEL="$(ls
  dist/*.whl)"`.
- `uv run pytest tests/deploy/ tests/integration/test_onnx_parity.py
  -m "deploy or onnx"` (drop the `|| echo skip` Phase-0 escape; the
  directory is no longer empty).
nightly.yml's `full-matrix` job (`nightly.yml:28-29`) is
`uv sync --extra dev` + `pytest -m "not gpu and not perf"`; add
`--extra onnx` to its `uv sync` so the `@pytest.mark.onnx` Step 4
tests (not gpu/perf) are collected and run nightly, and explicitly
add `tests/deploy/` to the nightly pytest invocation (or confirm it
is already collected via `testpaths`) so Steps 5/6 also run nightly.
Blast radius: `.github/workflows/pr.yml`,
`.github/workflows/nightly.yml`,
`pyproject.toml`.

## Step 4 - `tests/integration/test_onnx_parity.py`

New file. `pytest.importorskip("onnxruntime")` and
`pytest.importorskip("onnx")` at module top (both fully qualified);
every test `@pytest.mark.onnx`. CPU-forced tiny TFT mirroring
`tests/integration/test_gpu_cpu_parity.py` (`hidden_size=16,
attention_heads=2, max_epochs=3, precision="32-true", seed=42`).
Tests:
- `test_export_onnx_parity_binary`: fit a tiny `TFTClassifier` with
  a non-trivial calibrator on a panel that includes
  padding-edge entities (parametrize the panel over a
  multi-valid-step entity, a single-valid-step `T=1` entity, and an
  all-but-one-padded entity) so the `order`/`inverse`
  gather-scatter contract of the gather-preserving LSTM path is
  exercised at its boundaries, not only a generic short entity.
  Assert: (a) `_OnnxForward(*inputs)` == the production
  sorted-space oracle `est._predict_raw(panel)[0]` within
  `atol=1e-6` (NOT `_raw_outputs`; see I-4, `_predict_raw` is
  loaded-aware, both sides are transform/sorted order); (b)
  onnxruntime == `_OnnxForward` within `atol=1e-4, rtol=1e-4`; (c)
  sigmoid/softmax(ONNX logits) + the documented post-processing ==
  `est.predict_proba(panel)` for above-floor rows within
  `atol=1e-4`, AND a below-floor entity's `predict_proba` rows are
  NaN (bridges ONNX -> deployed prediction including the floor
  contract); (d) sanity that the calibrator is non-trivial (the
  fitted calibration parameter is not the identity) so (c) is
  actually sensitive to calibration being OUTSIDE the graph.
- `test_export_onnx_parity_loaded_estimator`: `save()` then
  `load()`, `export_onnx` from the loaded estimator, assert (a)+(b)
  (the (a) oracle `est._predict_raw(panel)[0]` works on the loaded
  estimator precisely because it routes through `_predict_module()`;
  this is what makes `_raw_outputs` the wrong oracle and is the
  whole point of this test). ALSO assert the loaded estimator's
  `predict_proba(panel)` equals the in-memory estimator's
  `predict_proba(panel)` (transitively carries the (c)
  calibration-bridge contract onto the loaded path, so a post-`load`
  calibrator-serialization bug is caught). This exercises the
  `_predict_module()` loaded path that `export_onnx` actually uses
  (production path coverage, qa-opus I1).
- `test_export_onnx_parity_multiclass` and
  `..._regressor_quantile`: repeat (a)+(b), asserting `(B, K)` shape
  (K=n_classes, K=n_quantiles) so a binary-only head-routing bug is
  caught.
- `test_export_onnx_dynamic_batch`: load the ALREADY-EXPORTED
  `.onnx` file (do NOT re-export) into a single `onnxruntime`
  session, then run it at a different batch size `B2 != B` and
  assert NUMERIC parity at B2 vs `_OnnxForward(*inputs_B2)` within
  `atol=1e-4` (proves the exported artifact's dynamic-batch dim is
  real, not that re-exporting at B2 happens to work). L not varied
  (I-5).
- `test_export_onnx_raises_not_fitted`: unfitted -> `NotFittedError`
  (the env-independent guard is Step 0b; this is the in-file
  companion).

## Step 5 - `tests/deploy/test_restricted_op_surface.py` (R2 C-A)

New file, `@pytest.mark.deploy` AND `@pytest.mark.onnx`,
`pytest.importorskip("onnx")`. Copy the Step 0b architecture.md
op-surface set VERBATIM into a module `ONNX_SAFE_OPS:
frozenset[str]` (with a comment citing the architecture.md
subsection as the source of truth). Export a tiny fitted model,
load the `ModelProto`, recurse every `graph.node` including subgraph
attributes of any control-flow nodes, collect the op_type set,
assert it is a SUBSET of `ONNX_SAFE_OPS`; on failure, the message
lists the offending op_type(s). Because the allowlist is the
independently-authored architecture.md enumeration (from module
static analysis, NOT from a sample export), an export emitting
`ScaledDotProductAttention`/`Attention`/`Loop`/`If` fails AT FIRST
PIN; the R2 guard is real.

## Step 6 - `tests/deploy/test_wheel_install.py`

New file, `@pytest.mark.deploy` (NOT `slow`: deploy must run in
per-PR CI; kept trivially small). `pytest.importorskip("build")`.
Prefer a CI-prebuilt wheel: if `os.environ.get(
"SEQ_SKLEARN_WHEEL")` is set (Step 3b sets it), use that path and
DO NOT rebuild; else `python -m build --wheel` into `tmp_path`.
Create a fresh venv, `pip install "<wheel>[onnx]"` (WITH the extra
so broken extra metadata is caught), then a subprocess that imports
`seq_sklearn`, imports `seq_sklearn.inference.onnx`, builds a
2-entity / few-period / 1-epoch tiny `TFTClassifier`, fits,
predicts, asserts the prediction shape. Subprocess non-zero exit ->
`pytest.fail` with captured stderr. Skips only if `build` is absent
AND `SEQ_SKLEARN_WHEEL` unset.

## Step 7 - docstring + three-site reconciliation

7a. `export_onnx` docstring carries the I-3 contract (raw logits;
no calibration/threshold/caller-order; inputs trusted; X required;
BETA).
7b. Reconcile, surfaced not silent, no behavior change:
requirements.md:389-394; requirements.md:1702-1712 (N1 ONNX-parity
paragraph, reword "the architecture phase enumerates ..." to
reference the now-discharged Step 0b architecture.md subsection, and
the SDPA clause to the interpretable-path reality);
implementation_plan.md:1399-1406 (R2: guard is the architecture-
enumerated op-surface test; SDPA context retained as a no-op
defense-in-depth); requirements.md:249 BETA row (raw-logits note).

## Step 8 - gate

Single background job, gpu deselected: ruff format + ruff check +
pyright (0 errors), then `pytest -m "not gpu"` full suite +
coverage (new `inference/onnx.py`, `export_onnx`, the Step 0b error
path, the architecture.md set referenced by Step 5) + 3
randomized-seed passes. Locally the `[onnx]` extra + `build` are
installed (dev env), so Steps 4/5/6 EXECUTE (not skip) in the gate;
the gate log records executed-vs-skipped per file and FAILS if any
of Steps 4/5/6 is skipped while `onnx`/`onnxruntime`/`build` are
importable (tripwire against silent skip with deps present). pgrep
orphan sweep; specific `git add` of changed paths only (never `-A`;
never the untracked root PDF); commit + push on
`phase-10-onnx-deploy`.

## Sequencing & risk

0/0b (errors + architecture op-surface) -> 1 (wrapper) -> 2
(export_onnx) -> 3 (extras + CI gate) -> 4/5/6 (tests) -> 7
(docstring + 3-site reconcile) -> 8 (gate). The Step 1 GATE
CONDITION removes the lowering question from the risk surface:
`_OnnxForward` ALWAYS uses the gather-preserving plain-LSTM path, so
the export never depends on `pack_padded_sequence` lowering at all.
The dominant remaining risk is therefore whether the
gather-preserving path is numerically faithful to the packed
production `_run_lstm`; Step 4 (a) (`_OnnxForward` ==
`est._predict_raw(panel)[0]` at `atol=1e-6`) plus the loaded-
estimator case prove that on every run, not whether packed lowering
was tested. The SDPA-vs-interpretable reconciliation (Key
reconciliation + Step 7b, all three sites) is the surfaced design
choice for the swarm.

## Claude consensus ledger (R1-R4)

Dual-model swarm (code/architecture/qa/style, opus+sonnet).

- R1: 6 REQUEST_CHANGES. CRITICALs: data-dependent guards block
  `torch.export(dynamo=True)`; dynamic-L contradicts the transform
  contract; op-allowlist tautology; parity-vs-production gap;
  not-fitted/import-guard untested; loaded-estimator accessor;
  `build` not a dep. -> full R2 rewrite (two-step strict=False
  export, dynamic-B-only, etc.).
- R2: still REQUEST_CHANGES. CRITICALs: LSTM-fallback "full
  left-padded sequence" numerics WRONG (leading-pad contaminates
  causal state); op-surface source phantom (requirements.md:1702-
  1712 enumerates zero ops); Step 0 import-guard test unspecified;
  CI `test-deploy` job a DEAD gate (never installs `[onnx]`/`build`).
  -> R3: gather-preserving fallback (drop only the pack wrapper);
  Step 0b discharges the op-surface enumeration into architecture.md
  with the concrete inlined set; env-independent import-guard unit
  test; pr.yml/nightly.yml `--extra dev --extra onnx` + prebuilt-
  wheel env var.
- R3: 5 APPROVE, 1 REQUEST_CHANGES (arch-opus). Sole CRITICAL: the
  parity oracle `_raw_outputs` is fitted-only and `AttributeError`s
  on the loaded-estimator test. -> R4: oracle repointed to the
  loaded-aware `est._predict_raw(panel)[0]`; fallback gate pinned
  always-on; `IsNaN` + provenance block; loaded `predict_proba`
  bridge; dynamic-batch loads the exported file; nightly job named.
- R4: ALL 4 reviewers APPROVE, ZERO CRITICAL. Remaining IMPROVEMENTs
  resolved in-plan: stale "Sequencing & risk" framing corrected
  (the GATE CONDITION removes packed-lowering from the risk
  surface); `nan_to_num`/`IsNaN` annotated UNCONDITIONAL;
  `GatherElements` provenance added; binary parity parametrized over
  padding-edge entities. style 0/0/0 all four rounds' final state.

CONSENSUS REACHED after 4 rounds (the `/review` hard cap): zero
CRITICAL, every IMPROVEMENT resolved, NITPICKs optional. The
SDPA-vs-interpretable reconciliation (keep interpretable path,
sdpa_kernel as no-op defense-in-depth, op-surface test as
enforcement, three-site doc reconcile in Step 7b) is the one
surfaced design choice, swarm-confirmed. Cleared to implement
(Steps 0..8), then code `/review` consensus, then Gemini final-pass.
