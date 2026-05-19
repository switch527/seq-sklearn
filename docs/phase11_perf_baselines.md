# Phase 11: Performance baselines (plan)

## Requirements

The grading rubric for this plan. Every swarm finding must trace to one
of these, or to a fundamental correctness concern.

- **P1** Ship the three perf benchmarks named in
  `docs/implementation_plan.md` Phase 11: `tests/perf/test_train_step_time.py`,
  `tests/perf/test_peak_memory.py`, `tests/perf/test_inference_latency.py`,
  measuring N7's metric set (median + P95 train step time, peak memory,
  per-sample inference latency).
- **P2** Ship checked-in baselines `tests/perf/_baselines/cpu-x86.json`
  and `tests/perf/_baselines/t4.json`, the two public A13 cells, each
  carrying every measured metric.
- **P3** A regression gate per A13: fail at **>15%** over baseline
  median step time and **>10%** over baseline peak memory. The gate is
  nightly; on PR-CI it warns and does not block (A13, implementation_plan
  Phase 11 "Deliverable tests").
- **P4** Done-when (implementation_plan Phase 11): the nightly job
  produces a baseline JSON for the two public cells AND the regression
  gate fires correctly on a contrived 20% slowdown.
- **P5** Repo rules hold: pydantic v2 for structured data, type hints,
  `ruff`/`pyright` clean, no `print`, tests mirror layout, the perf
  suite is `-m perf` and excluded from the default + coverage runs
  (acceptance criterion 2: `pytest -m "not slow and not perf"`).

## Scope and non-goals

- **Scope**: a *relative* regression gate on a fixed, scaled proxy
  workload, captured per hardware cell, run nightly. Phase 11 owns
  step-time / memory / latency regression with checked-in per-cell
  baselines (`docs/benchmark_suite_design.md` B0.1 confirms this
  ownership boundary).
- **Non-goal NG1**: Phase 11 is NOT the N7 *absolute* budget validation.
  N7's reference workload (100k entities x 24 months x 30 features,
  < 8 GB GPU, < 30 min on A100/T4/4090) cannot run inside a nightly
  ubuntu/T4 job. Acceptance criterion 9 ("all N1-N7 met") is discharged
  by a NAMED owning artifact, not a bare deferral (arch-C1). The
  cross-doc wiring is LANDED in this same change set, not asserted:
  `docs/requirements.md` criterion 9 now names
  `tests/perf/test_n7_absolute.py` as the evidence, and
  `docs/implementation_plan.md` Phase 12 carries the release-checklist
  step that runs it. Phase 11 ships `tests/perf/test_n7_absolute.py`,
  marked `@pytest.mark.gpu` AND `@pytest.mark.slow`, asserting the four
  N7 numeric budgets on the N7 reference config. It is excluded from
  PR CI and from the nightly GPU job: `nightly.yml`'s gpu job is
  changed in this set to `-m "gpu and not slow"` so this slow N7-scale
  test does NOT run nightly (arch-I, the marker-overlap fix). It runs
  ONLY when invoked manually as the Phase 12 release-checklist step.
  Phase 11's *gate* (P3) measures a small deterministic proxy and
  gates *change*; `test_n7_absolute.py` is the separate
  absolute-conformance check. Authoring the gpu+slow test stub and the
  three doc/CI edits is the Phase 11 N7-absolute deliverable; actually
  running it at full N7 scale pre-release is D1.
- **Non-goal NG2**: not the comparative benchmark suite
  (`docs/benchmark_suite_design.md`); that is separate science, does not
  gate CI, and is not part of v1 Phase 11.
- **Non-goal NG3**: no multi-GPU / distributed perf. Single-cell only.

## P-A: Benchmarked workload (the proxy)

One fixed config, shared by all three benchmarks, small enough that the
CPU cell finishes in the nightly budget yet exercises the real
TFT train/predict path through the public estimator API.

- **PA.1** `SyntheticPanelGenerator(target_kind="binary",
  num_entities=N, periods_per_entity=P, lookback=L, seed=11)` with a
  single pinned `(N, P, L)` proxy size declared as module constants in
  `tests/perf/_workload.py`. Initial proposal: `N=256`, `P=24`, `L=12`
  (L matches the N7 reference lookback; N/P scaled for a CPU nightly
  envelope, tunable in review). The same panel + `TFTClassifier`
  hyperparameters (`hidden_size=128`, `attention_heads=4` per the N7
  reference architecture; `max_epochs` small, fixed `batch_size`) are
  reused by all three tests via a session-scoped fixture in
  `tests/perf/conftest.py`.
- **PA.2** Determinism mode ON for every perf run (the library's strict
  determinism path), `precision="32-true"`, hardware forced through the
  real `detect()` (NOT monkeypatched: the perf run must measure the
  path the cell actually executes). Seeds fixed.
- **PA.3** The workload builder is one function
  `build_proxy_estimator_and_panel()` returning
  `(estimator, panel, y)`; the three tests differ only in WHAT they
  measure on it, not in the workload. This guarantees the three metrics
  describe the same execution.

## P-B: The three benchmarks

- **PB.1 `test_train_step_time.py`** uses `pytest-benchmark` (already a
  dev dep, `perf` marker already registered). Benchmarks a single
  training step (one optimizer step on one batch) via the Lightning
  path, `--benchmark-only`. Recorded metrics: median and P95 step
  seconds from the pytest-benchmark stats object (`stats.stats.median`,
  and the 95th percentile computed from `stats.stats.data` with a
  pinned numpy `percentile(..., 95, method="linear")` so the P95 has a
  non-arbitrary oracle).
- **PB.2 `test_peak_memory.py`** measures peak memory of a full
  fit+predict of the proxy. CPU cell: `tracemalloc` peak (Python
  allocations) PLUS `resource.getrusage(RUSAGE_SELF).ru_maxrss` (RSS
  high-water), the gated metric being `ru_maxrss` (the N7 "GPU memory"
  analog on CPU). CUDA cell: `torch.cuda.reset_peak_memory_stats()` then
  `torch.cuda.max_memory_allocated()` after fit+predict. The JSON
  records the metric name actually gated per cell so a CPU/GPU mix is
  explicit, never silently comparing RSS to CUDA bytes.
- **PB.3 `test_inference_latency.py`** measures per-sample inference
  latency on a batch of 1024 windows (the N7 latency reference size),
  reported as median and P95 over a fixed repeat count, measured
  separately from a throughput-batched predict (warm-up iterations
  excluded, count pinned). Uses `time.perf_counter`, not
  pytest-benchmark, because the unit is per-sample latency over a fixed
  batch, not a microbench round.

## P-C: Baseline file contract

- **PC.1** `tests/perf/_baselines/<cell>.json` schema is a pydantic v2
  model `PerfBaseline` (serialized via `model_dump_json`) with
  `model_config = ConfigDict(extra="forbid")` so an unknown key is a
  hard `ValidationError`, never silently dropped (qa-C5). Fields:
  `cell` (`Literal["cpu-x86", "t4"]`), `captured_git_sha: str`,
  `torch_version: str`, `python_version: str`, `device_name: str`
  (the resolving device string, `"cpu"` for the CPU cell or
  `torch.cuda.get_device_name(0)` for a CUDA cell; recorded so a
  Volta/Turing-tier collision is auditable per PC.2 / arch-I1),
  `provisional: bool` (default `False`; PE.3 / qa-N3, listed here so
  PC.1 is the single field-set source of truth), and the metric
  fields, each a `float` (NOT optional, so a missing one raises at
  load):
  `train_step_median_s`, `train_step_p95_s`, `peak_memory_value`,
  `peak_memory_metric: Literal["ru_maxrss_kb",
  "cuda_max_alloc_bytes"]`, `inference_latency_median_s`,
  `inference_latency_p95_s`. The `Literal` typing makes a wrong
  metric-name string (e.g. `cuda_bytes`) a load-time
  `ValidationError`, not a silent mis-gate.
- **PC.1a (module-import boundary, arch-C3).** `PerfBaseline`, the cell
  resolver (PC.2), the gate helper (PD.1), and `PerfRegressionError`
  live in `tests/perf/_gate.py`, which imports ONLY pydantic + stdlib
  at module load (no `torch`, no `seq_sklearn` estimator import). The
  heavy proxy (`tests/perf/_workload.py`, which imports torch +
  `TFTClassifier`) is imported only inside the `perf`-marked benchmark
  bodies and the capture CLI, never at `_gate.py` import time. The
  default + coverage suite (acceptance criterion 2) runs PG.1/PG.2,
  which import only `_gate.py`; they therefore never construct a
  `TFTClassifier` or pull torch through this path, keeping the fast PR
  job fast (this boundary is asserted by
  `test_gate_module_has_no_heavy_imports`, PG.3).
- **PC.2** Cell identity is resolved in two steps, NOT by tier alone
  (arch-I1: `HardwareTier.VOLTA_TURING` spans both V100, CC 7.0, and
  T4, CC 7.5, so a tier-only map would mis-resolve a V100 runner to
  `t4.json`). Step 1: `HardwareTier.detect()` is the coarse gate,
  `CPU -> cpu-x86`. Step 2 for CUDA: the cell is `t4` only if
  `HardwareTier.detect() == VOLTA_TURING` AND
  `"T4" in torch.cuda.get_device_name(0)`. Any CUDA device that is not
  a name-confirmed T4 (V100, A10, H100, ...) resolves to no baseline
  and the perf path `pytest.skip`s with a reason that includes BOTH
  the detected tier name AND `torch.cuda.get_device_name(0)`, so a
  Volta/Turing collision is diagnosable from the skip message, not
  silent. `device_name` (PC.1) is written on every captured row so the
  exact resolving device is auditable. The two public cells are the
  only v1 baselines per A13/P2; other devices are optional contributor
  cells (D2), not a Phase 11 deliverable. The cell resolver lives in
  `_gate.py` but the `torch.cuda` calls are lazily imported INSIDE the
  resolver function (not at module top) to preserve PC.1a's
  no-torch-at-import boundary; PG.3 asserts this.

## P-D: The regression gate

- **PD.1** A `conftest.py` helper `assert_within_baseline(cell, measured)`
  loads `PerfBaseline` for the resolved cell and checks:
  `measured.train_step_median_s <= baseline * 1.15` and
  `measured.peak_memory_value <= baseline * 1.10` (A13 thresholds; P3).
  Latency and P95 are recorded and reported but NOT hard-gated in v1
  (A13 names only step-time 15% and memory 10% as gate thresholds;
  latency is observational). The peak-memory comparison asserts
  `measured.peak_memory_metric == baseline.peak_memory_metric` first, so
  an RSS-vs-CUDA mismatch fails loudly rather than comparing
  incommensurable units.
- **PD.1a (metric-name guard, qa-C1).** Before the numeric memory
  comparison the helper asserts
  `measured.peak_memory_metric == baseline.peak_memory_metric`; on
  mismatch it raises `PerfRegressionError` naming both metrics,
  regardless of the enforce/warn mode (a unit mismatch is a
  configuration bug, not a measured regression, so it must fail loudly
  even in warn mode). This blocks an RSS-kilobytes vs CUDA-bytes
  numeric compare that could otherwise read as a spurious pass.
- **PD.1b (precedence, qa NEW-C1).** The helper's first action is the
  provisional short-circuit (PE.3): if `baseline.provisional is True`
  it warns and returns BEFORE the metric-name guard, the mode read,
  and any numeric comparison. Rationale: a provisional baseline's
  `peak_memory_metric` is itself a hand-seeded placeholder, so
  comparing it is meaningless and a "mismatch" against it is not a
  real misconfiguration signal. Precedence is therefore strictly:
  (0) load+validate the baseline file; on load/validation failure
  apply the PD.1c missing/corrupt branch and stop; else
  (1) provisional => warn+return; else (2) metric-name mismatch =>
  raise (mode-independent); else (3) mode dispatch (enforce raises on
  breach, warn/unset warns). Step (0) necessarily precedes (1) because
  `provisional` cannot be read from a file that failed to parse (arch
  R3-I; the enumeration now names the load step PD.1c depends on).
  PE.3 and PD.1a do not conflict because (1) strictly precedes (2).
  `test_provisional_precedes_mismatch`
  (PG.1) pins this exact order: provisional=True AND a metric-name
  mismatch AND enforce mode => no raise, only the provisional warning.
- **PD.1c (missing/corrupt baseline, arch-I4).** If the resolved
  cell's baseline file is absent or fails `PerfBaseline` validation
  (truncated, bad JSON, missing/extra key), the helper does NOT
  silently pass: under `enforce` it raises `PerfRegressionError`
  ("baseline for cell <cell> missing or invalid: <detail>"); under
  `warn`/unset it emits the warning and returns. A corrupt baseline is
  never treated as "no regression". This is ordered AFTER the
  provisional short-circuit only when a file exists and parses; an
  unparseable file cannot be read for `provisional`, so the
  missing/corrupt branch is evaluated first when load itself fails.
  PG.2 pins all four branches with individually named tests:
  `test_missing_baseline_enforce_raises`,
  `test_corrupt_baseline_enforce_raises`,
  `test_missing_baseline_warn_logs`,
  `test_corrupt_baseline_warn_logs` (qa R3-I1: the warn-mode variants
  are named, not folded into "the same two").
- **PD.2 Nightly vs PR.** Mode is read from `SEQ_SKLEARN_PERF_GATE`:
  `enforce` => a breach raises `PerfRegressionError` (set by the
  nightly perf job); `warn` OR UNSET OR any other value => the helper
  emits a `logging.warning` + a structured log record and returns
  without raising (qa-C2: unset defaults to `warn`, the safe default,
  so a nightly job that forgets the var degrades to a non-blocking
  alert rather than silently never-firing or unexpectedly blocking).
  This matches A13 ("CPU PR regression fires ... but does NOT block
  merge in v1 ... appears as a nightly alert"). The
  metric-name-mismatch raise (PD.1a) is the one condition that ignores
  the mode. The enforce path, the warn path, and the unset-default
  path each have a named PG.1 test.
- **PD.3 Noise control.** Shared GitHub CPU runners are timing-noisy.
  Mitigations: median (not mean) over a `_workload.py` module constant
  `BENCH_MIN_ROUNDS` (named, like `INFERENCE_WARMUP`/`INFERENCE_REPEATS`,
  so it is falsifiable not prose-only, arch R3-N; PG.8 asserts it is
  the value passed to pytest-benchmark); pytest-benchmark
  `warmup=True`; the gate is on median step time vs a 15% band
  (already wide); the CPU cell is nightly-alert-only (PD.2), so a
  noisy false-positive pages no one and blocks no merge. P95 is
  recorded for trend visibility but not gated (PD.1).

## P-E: Baseline capture / update protocol

- **PE.1** Baselines are captured, not hand-written: a CLI entry
  `python -m tests.perf.capture --cell <cell>` runs the three
  benchmarks and writes `_baselines/<cell>.json` with the current git
  SHA. The same code path the gate measures produces the baseline (no
  divergence between capture and check).
- **PE.2** A baseline change is reviewed exactly like the Phase 9
  snapshot precedent, mirroring `scripts/check_snapshots.sh` in FULL
  (arch-C2), via a new `scripts/check_perf_baselines.sh` wired as a
  `perf-baseline-guard` job in `pr.yml`. It replicates all three
  snapshot-guard behaviors against the `tests/perf/_baselines/` glob:
  (i) a **bot-authored PR** (`PR_USER_TYPE=Bot`, injected from
  `github.event.pull_request.user.type` exactly as the snapshot job
  does) touching `_baselines/` **hard-fails unconditionally**, no
  marker can override it. This is load-bearing in an
  agent-authored-PR repo: without it an automated PR could move the
  perf goalposts and self-approve, the precise failure PE.2 exists to
  prevent. (ii) `_baselines/*.json` modified alongside non-baseline
  files requires a `PERF_BASELINE_REVIEWED: <reason>` marker in some
  PR commit message. (iii) no `_baselines/` change => the guard is a
  no-op pass. The script is a near-verbatim adaptation of
  `check_snapshots.sh` with the glob and marker string swapped, so the
  bot hard-fail is not accidentally dropped.
- **PE.3** The `t4.json` baseline is captured by the nightly self-hosted
  GPU runner (A13: "Google Colab T4 captured by a nightly self-hosted
  runner"); `cpu-x86.json` is captured on `ubuntu-latest`. Until a real
  T4 capture lands, `t4.json` is committed with `provisional: true`
  and the GPU perf job is `continue-on-error` so a missing/stale T4
  baseline never blocks (it is nightly-only anyway). **Provisional
  gate behavior (I4):** when `PerfBaseline.provisional is True`, the
  gate helper (PD.1) NEVER raises regardless of mode or measured
  value; it emits a `logging.warning` ("gating skipped: provisional
  baseline for cell <cell>") and returns. A provisional baseline holds
  hand-seeded, not measured, numbers; gating against them would
  produce false regressions or false passes. The first real
  self-hosted T4 capture overwrites `t4.json` with
  `provisional: false` (a `PERF_BASELINE_REVIEWED:` change). This
  behavior has a named PG.1 test.

## P-F: CI wiring

- **PF.1** Replace the placeholder nightly `perf` job body
  (`uv run pytest -m perf --benchmark-only || echo "empty in Phase 0"`)
  with: `SEQ_SKLEARN_PERF_GATE=enforce uv run pytest -m perf
  --benchmark-only -p no:cacheprovider`. The job stays
  `runs-on: ubuntu-latest` (the cpu-x86 cell) and remains in `nightly.yml`
  (not `pr.yml`): perf is nightly-only (implementation_plan Phase 11).
- **PF.2** Add a nightly `perf-gpu` job under the existing
  `[self-hosted, gpu]` runner (guarded by the same
  `vars.GPU_RUNNER_AVAILABLE` condition) running the same command; it
  resolves the `t4` cell. `continue-on-error: true` per PE.3.
- **PF.3** Add the `perf-baseline-guard` job to `pr.yml` (PE.2), the
  only Phase 11 addition to PR-CI; it is a fast git-diff/marker check,
  not a perf run, so it respects the 5-minute PR budget.

## P-G: Done-when proof (P4) and the named test roster

Every test below is in `tests/perf/` but, except where it says
`perf`-marked, is NOT `perf`-marked, so it runs in the default
`pytest -m "not slow and not perf"` suite (acceptance criterion 2) and
imports only `_gate.py` (PC.1a), never the heavy workload. This makes
the P4 done-when ("gate fires correctly on a contrived 20% slowdown")
and the gate's correctness offline, deterministic, and fast.

- **PG.1 `test_gate_logic.py`** (the P4 discharge + every gate branch):
  - `test_enforce_step_breach_raises`: `measured` step =
    baseline * 1.20, `SEQ_SKLEARN_PERF_GATE=enforce` => raises
    `PerfRegressionError`. (P4 step arm.)
  - `test_enforce_step_within_band_passes`: step = baseline * 1.05
    (< 15%) => no raise. (Non-vacuous polarity.)
  - `test_enforce_memory_breach_raises` / `..._within_band_passes`:
    memory = baseline * 1.20 vs baseline * 1.05 against the 10% band.
    (P4 memory arm + the asymmetric 10% threshold.)
  - `test_warn_mode_breach_does_not_raise`: 1.20 breach,
    `SEQ_SKLEARN_PERF_GATE=warn` => no raise, asserts a
    `logging.warning` record is emitted (caplog). (qa-C3.)
  - `test_unset_env_defaults_to_warn`: 1.20 breach,
    `provisional=False` AND a matching `peak_memory_metric` (qa R3-I2:
    pinned explicitly so the no-raise is reached via the mode-dispatch
    default, not the provisional or load-failure short-circuits), env
    cleared via
    `monkeypatch.delenv("SEQ_SKLEARN_PERF_GATE", raising=False)` (qa
    NEW-I1: `raising=False` so the test is correct whether or not the
    var was set in the test process) => no raise + warning emitted.
    (qa-C2; pins the safe default.)
  - `test_metric_name_mismatch_raises_even_in_warn`: baseline
    `peak_memory_metric="ru_maxrss_kb"`, measured
    `"cuda_max_alloc_bytes"`, mode `warn`, `provisional=False` =>
    raises (PD.1a; the one mode-independent raise). (qa-C1.)
  - `test_provisional_baseline_never_gates`: `provisional=True`,
    1.20 breach, mode `enforce` => no raise + the
    "gating skipped: provisional" warning. (I4.)
  - `test_provisional_precedes_mismatch`: `provisional=True` AND a
    metric-name mismatch AND mode `enforce` => no raise, only the
    provisional warning (pins PD.1b precedence; qa NEW-C1).
- **PG.2 `test_baseline_schema.py`** (PC.1 schema is enforced, not
  assumed):
  - `test_perf_baselines_present_and_valid`: both
    `_baselines/cpu-x86.json` and `t4.json` exist and
    `PerfBaseline.model_validate_json` succeeds with the full key set.
    (P2/P4 checked-in-artifact clause.)
  - `test_missing_metric_key_raises`: a dict without
    `train_step_p95_s` => `ValidationError` (qa-C5; pins non-optional).
  - `test_extra_key_raises`: an unknown key => `ValidationError`
    (pins `extra="forbid"`).
  - `test_bad_peak_memory_metric_literal_raises`:
    `peak_memory_metric="cuda_bytes"` => `ValidationError` (pins the
    `Literal`).
  - `test_tracemalloc_key_rejected`: a JSON with a `tracemalloc_peak_kb`
    key => `ValidationError` (pins Q2/qa NEW-C2: tracemalloc is logged,
    not persisted).
  - `test_missing_baseline_enforce_raises`,
    `test_corrupt_baseline_enforce_raises`: a nonexistent path and a
    truncated-JSON file, mode `enforce` => `PerfRegressionError`.
  - `test_missing_baseline_warn_logs`,
    `test_corrupt_baseline_warn_logs`: the same two inputs, mode
    `warn` => no raise + a `logging.warning` record (qa R3-I1: the
    warn variants are individually named, four total for PD.1c).
  - `test_capture_writes_device_name_cpu` (qa R3-C1): drive the CPU
    capture path (PG.6's monkeypatched-measurement harness, cell
    `cpu-x86`) and assert the round-tripped `PerfBaseline.device_name
    == "cpu"`. `device_name` is a free `str`, so schema validation
    alone cannot catch a capture bug that copies the CUDA branch on
    CPU or leaves it empty; this is the only assertion that pins the
    auditability claim in PC.2.
- **PG.3 `test_gate_module_boundary.py`**:
  - `test_gate_module_has_no_heavy_imports`: in a subprocess launched
    with `sys.executable` (qa NEW-I2: same interpreter/venv as pytest,
    not ambient `python`), `import tests.perf._gate`, THEN call the
    cell resolver with a monkeypatched `detect()` returning `CPU`
    (qa R3-I3: exercises the resolver's internal lazy-import contract,
    not just the module-level boundary, so a top-of-function
    `import torch` in the resolver is caught), then assert `"torch"`
    and `"seq_sklearn"` are absent from `sys.modules`. Pins PC.1a /
    arch-C3 so the fast PR suite cannot regress into pulling torch.
  - `test_cell_resolver_mapping`: monkeypatch `detect()` and
    `torch.cuda.get_device_name` => `CPU` resolves `cpu-x86`;
    `VOLTA_TURING` + device name containing `"T4"` resolves `t4`;
    `VOLTA_TURING` + a `"V100"` device resolves to "no baseline"
    (arch-I1 collision); `PASCAL`, `AMPERE_ADA`, `HOPPER`, `BLACKWELL`
    each resolve to "no baseline". For every "no baseline" case assert
    the `pytest.skip` reason string CONTAINS both the tier name and
    the device name (qa NEW-I3: the skip is diagnosable, not opaque;
    qa Q5 silent-skip gap; PC.2).
- **PG.4 `test_workload.py`** (`perf`-marked where it constructs the
  estimator; the budget test is subprocess-isolated):
  - `test_proxy_estimator_config_matches_spec` (`perf`-marked):
    `build_proxy_estimator_and_panel()` returns a `TFTClassifier` with
    `hidden_size==128`, `attention_heads==4`, and a panel of exactly
    `N*P` rows over the `_workload.py` `(N, P, L)` constants (qa-C4;
    all three benchmarks measure the spec'd workload, not a
    misconfigured one).
  - `test_proxy_builds_within_cpu_budget`: run
    `build_proxy_estimator_and_panel()` + one fit in a subprocess with
    a hard `timeout` (constant, proposed 180 s); assert exit 0 within
    the timeout, so an oversized proxy fails loudly instead of silently
    timing the nightly job out (qa-I1 / R1).
- **PG.5 `conftest.py` fixture guard (I3)**: the session-scoped perf
  fixture asserts `torch.are_deterministic_algorithms_enabled() is
  True` at setup and raises `RuntimeError` otherwise, so a perf run in
  non-deterministic mode fails loudly rather than producing
  unreproducible numbers that the gate then trusts. The fixture is
  `autouse=False` and explicitly requested only by the `perf`-marked
  workload tests (qa R3-I4): a session `autouse` fixture would import
  torch at the start of the fast PR suite and violate the PC.1a
  no-torch boundary for the non-`perf` PG.1/PG.2/PG.3 tests.
  `test_perf_fixture_not_autouse` (in `test_gate_module_boundary.py`,
  non-`perf`) asserts the fixture object has `autouse is False` so the
  boundary cannot silently regress.
- **PG.6 `test_capture_cli.py`** (qa-I2): monkeypatch the three
  benchmark measurement functions to return known constants, run
  `python -m tests.perf.capture --cell cpu-x86` against `tmp_path`,
  assert the written JSON round-trips through `PerfBaseline` and the
  metric values equal the monkeypatched constants. The
  `captured_git_sha == git rev-parse HEAD` assertion is guarded by
  `shutil.which("git")` and skipped (not failed) when git is absent or
  HEAD is unresolvable (qa NEW-N1: avoids an environmental false
  failure unrelated to capture logic). Catches a capture/gate schema
  divergence before any real nightly run.
- **PG.7 `test_check_perf_baselines_script.py`** (qa-I5): drive
  `scripts/check_perf_baselines.sh` in a `tmp_path` git repo over
  these cases: bot author + baseline change => exit 1 AND stdout
  contains the bot-fail message (qa NEW-N2; the load-bearing arch-C2
  case); baseline + source change without marker => exit 1; with
  marker => exit 0; no baseline change => exit 0. Pins the guard
  regex/branches offline, no CI required.
- **PG.8 metric-helper oracles** (qa-N2): `test_p95_matches_numpy`
  asserts the P95 helper on a known data list equals
  `np.percentile(data, 95, method="linear")`; the warm-up/repeat counts
  (PB.3) are module constants `INFERENCE_WARMUP`, `INFERENCE_REPEATS`
  asserted applied (qa-N1); `test_bench_min_rounds_applied` asserts
  the train-step benchmark is invoked with `BENCH_MIN_ROUNDS` (PD.3 /
  arch R3-N).
- **PG.9 `test_latency_breach_does_not_raise`** (qa R3-N1): a measured
  `inference_latency_median_s` of baseline * 3.0 under mode `enforce`
  (matching metric, non-provisional) => no raise. Pins PD.1's
  "latency is observational, not gated" so an implementation that
  accidentally hard-gates latency is caught.

## Resolved questions (round 1)

- **Q1 RESOLVED** Proxy size stays `(N=256, P=24, L=12)` as the v1
  proposal, now falsifiable rather than asserted: `PG.4
  test_proxy_builds_within_cpu_budget` enforces a hard subprocess
  timeout (180 s constant), so an oversized proxy fails loudly. The
  size is a `_workload.py` constant; tuning it is a
  `PERF_BASELINE_REVIEWED:` change since it invalidates baselines.
- **Q2 RESOLVED** The CPU cell gates `ru_maxrss` (RSS high-water): it
  is the closest analog to N7's GPU-memory budget because it includes
  torch C++/MKL allocations that `tracemalloc` (Python-only) misses.
  `tracemalloc` peak is emitted only in the structured log line at
  measurement time for human trend inspection; it is NOT persisted in
  `PerfBaseline` (qa NEW-C2: with `extra="forbid"` an unschema'd
  `tracemalloc` JSON key would be a hard `ValidationError`, so it must
  not be written). The single persisted+gated memory metric is
  `peak_memory_value` tagged by the `peak_memory_metric` `Literal`.
  `PG.2 test_tracemalloc_key_rejected` pins this: a JSON carrying a
  `tracemalloc_peak_kb` key fails `PerfBaseline` validation.
- **Q3 RESOLVED** Median-only step-time gate for v1, matching A13
  verbatim (it names only median step + peak memory as gate
  thresholds). P95 step + both latency stats are recorded for trend
  visibility but not hard-gated (PD.1). Revisit in v2.
- **Q4 RESOLVED** Commit a hand-seeded `t4.json` with
  `provisional: true` NOW so P2 ("baseline JSON for the two public
  cells") is literally satisfied at Phase 11 merge as a schema-valid
  checked-in artifact, while PE.3's provisional gate behavior
  guarantees those un-measured numbers never gate. First real
  self-hosted T4 capture flips it to `provisional: false`.

## Risks

- **R1** Shared-runner timing noise causes false regressions. Mitigated
  by median + wide band + nightly-alert-only CPU gate (PD.3).
- **R2** No self-hosted T4 at Phase 11 merge => `t4.json` cannot be
  truly captured. Mitigated by PE.3 provisional flag + continue-on-error;
  worst case P2's t4 cell is provisional and Q4-resolved.
- **R3** Determinism mode changes perf characteristics vs a real user
  run with determinism off. Accepted: the gate measures change in the
  determinism-on path consistently; absolute N7 is NG1/D1.

## Deferred

- **D1** N7 absolute-budget GPU-scale validation (100k x 24 x 30,
  < 8 GB, < 30 min): separate periodic GPU run, not Phase 11 (NG1).
- **D2** Optional contributor cells `(ampere-a10)`, `(hopper-h100)`
  (A13 "optional"): registry admits them by adding a tier->file mapping
  and a captured JSON, no code change; not a v1 deliverable.
- **D3** Multi-GPU / distributed perf (NG3).

## Tracking (review loop)

Addressed and deferred items are maintained here so successive swarm
runs see prior decisions and do not re-raise resolved points.

Round 1 (architecture 3C/4I/2N REQUEST_CHANGES; qa 5C/4I/3N
REQUEST_CHANGES; style 0/0/0 APPROVE):

- Addressed (CRITICAL):
  - arch-C1: NG1 now names an owning artifact for acceptance criterion
    9 (`tests/perf/test_n7_absolute.py`, gpu+slow, release-checklist
    procedure), not a bare deferral.
  - arch-C2: PE.2 mirrors `check_snapshots.sh` in full, including the
    unconditional bot-PR hard-fail; PG.7 pins that branch offline.
  - arch-C3: PC.1a states the `_gate.py` import boundary (pydantic +
    stdlib only); PG.3 `test_gate_module_has_no_heavy_imports`
    enforces it so the fast PR suite never pulls torch.
  - qa-C1: PD.1a + `test_metric_name_mismatch_raises_even_in_warn`.
  - qa-C2: PD.2 pins unset == `warn` (safe default) +
    `test_unset_env_defaults_to_warn`.
  - qa-C3: warn path `test_warn_mode_breach_does_not_raise` (caplog).
  - qa-C4: PG.4 `test_proxy_estimator_config_matches_spec`.
  - qa-C5: PC.1 `extra="forbid"` + non-optional floats + `Literal`s;
    PG.2 missing/extra/bad-literal rejection tests.
- Addressed (IMPROVEMENT):
  - qa-I1: PG.4 `test_proxy_builds_within_cpu_budget` (subprocess
    timeout).
  - qa-I2: PG.6 `test_capture_cli.py` round-trip.
  - qa-I3 / arch-I3: PG.5 conftest fixture asserts determinism active.
  - qa-I4 / arch-I4: PE.3 defines provisional => never-gates; PG.1
    `test_provisional_baseline_never_gates`.
  - qa-I5: PG.7 drives `check_perf_baselines.sh` offline.
  - arch-I1/I2: folded into the arch-C1/C2 doc fixes above.
- Addressed (NITPICK):
  - qa-N1: PB.3 warm-up/repeat are named constants
    (`INFERENCE_WARMUP`, `INFERENCE_REPEATS`), asserted in PG.8.
  - qa-N2: PG.8 `test_p95_matches_numpy` oracle.
  - qa-N3: `provisional` is now listed in the PC.1 field set (single
    source of truth), resolving the PC.1/PE.3 inconsistency.
- Resolved: Q1 (size + budget test), Q2 (RSS gated, tracemalloc
  observational), Q3 (median-only per A13), Q4 (provisional t4.json
  committed now, never gates).

Round 2 (architecture 1C/3I/1N REQUEST_CHANGES; qa 2C/3I/2N
REQUEST_CHANGES; style 0/0/0 APPROVE). All round-1 findings confirmed
discharged; round 2 surfaced revision-introduced issues:

- Addressed (CRITICAL):
  - arch-C1 (re-raised, not closed in R1): the cross-doc wiring is now
    actually LANDED in this change set, not asserted. `requirements.md`
    criterion 9 names `tests/perf/test_n7_absolute.py`;
    `implementation_plan.md` Phase 12 carries the release-checklist
    step; `nightly.yml` gpu job changed to `-m "gpu and not slow"` so
    the gpu+slow N7 test never runs nightly. NG1 reworded to cite the
    landed edits, not claim a pre-existing one.
  - qa NEW-C1: PD.1b pins gate precedence (provisional short-circuit
    strictly precedes the metric-name guard and mode dispatch), so
    PE.3 and PD.1a no longer conflict; `test_provisional_precedes_mismatch`.
  - qa NEW-C2: Q2 reworded, `tracemalloc` is logged, NOT persisted in
    `PerfBaseline` (would violate `extra="forbid"`);
    `test_tracemalloc_key_rejected` pins it.
- Addressed (IMPROVEMENT):
  - arch-I (marker overlap): `nightly.yml` gpu job excludes `slow`
    (above); NG1 states it.
  - arch-I4 (corrupt/missing baseline): PD.1c, missing/corrupt under
    enforce raises, under warn warns; PG.2 tests both.
  - arch-I1 (V100/T4 CC-7.x collision): PC.2 resolves the t4 cell by
    `detect()==VOLTA_TURING` AND `"T4" in get_device_name`; non-T4
    Volta/Turing => no baseline + diagnostic skip; `device_name` added
    to PC.1; torch lazily imported in the resolver to keep PC.1a.
  - qa NEW-I1: PG.1 unset test mandates
    `monkeypatch.delenv(..., raising=False)`.
  - qa NEW-I2: PG.3 boundary test runs the subprocess via
    `sys.executable`.
  - qa NEW-I3: PG.3 `test_cell_resolver_mapping` asserts the skip
    reason contains both tier and device name.
- Addressed (NITPICK):
  - qa NEW-N1: PG.6 git-sha assertion guarded by `shutil.which("git")`.
  - qa NEW-N2: PG.7 asserts the bot-fail stdout message.
  - arch NIT (R2): folded into the arch-C1 doc rewrite.

Round 3 (architecture 0C/1I/2N APPROVE; qa 1C/4I/1N
REQUEST_CHANGES; style 0/0/0 APPROVE). All round-2 findings verified
genuinely closed against the landed files; round 3 surfaced
audit-level gaps:

- Addressed (CRITICAL):
  - qa R3-C1: `device_name` on the CPU cell had no asserting test (it
    is a free `str`, schema cannot catch a wrong value). Added
    `PG.2 test_capture_writes_device_name_cpu` pinning the PC.2
    auditability claim.
- Addressed (IMPROVEMENT):
  - arch R3-I: PD.1b precedence list now names step (0) load+validate
    => PD.1c on failure, so PD.1b no longer contradicts PD.1c read in
    isolation.
  - qa R3-I1: PD.1c warn-mode variants individually named
    (`test_missing_baseline_warn_logs`,
    `test_corrupt_baseline_warn_logs`), four named tests total.
  - qa R3-I2: `test_unset_env_defaults_to_warn` pins
    `provisional=False` + matching metric so the no-raise is reached
    via mode dispatch, not a short-circuit.
  - qa R3-I3: PG.3 boundary test now also calls the resolver
    (monkeypatched `detect()==CPU`) in the subprocess, exercising the
    resolver's internal lazy-import contract, not just the module
    boundary.
  - qa R3-I4: PG.5 fixture stated `autouse=False`, perf-requested
    only; `test_perf_fixture_not_autouse` pins it so the PC.1a
    boundary cannot silently regress.
- Addressed (NITPICK):
  - qa R3-N1: `PG.9 test_latency_breach_does_not_raise` pins
    latency-is-observational.
  - arch R3-N1 (min_rounds prose-only): PD.3 now names
    `BENCH_MIN_ROUNDS` as a `_workload.py` constant with PG.8
    `test_bench_min_rounds_applied` asserting it. arch R3-N2 (NG1
    density): left as-is, pure prose polish with no behavioral content
    (cosmetic; deferred, not blocking).

Gemini final pass: not yet run (runs after Claude consensus).
