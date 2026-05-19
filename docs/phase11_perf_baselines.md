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
  dev dep, `perf` marker already registered) via the `benchmark`
  fixture. Benchmarks a single training step (one optimizer step on
  one batch) via the Lightning path. NOTE (Gemini-NIT): on CPU a
  single `Trainer.fit` carries dataloader/callback init overhead that
  is non-trivial vs one step; this is acceptable for a RELATIVE gate
  because the overhead is identical run-to-run on a fixed proxy and
  the baseline captures it too, but the doc states it so the absolute
  number is never read as pure step time. Recorded metrics: median and
  P95 step seconds from the pytest-benchmark stats object
  (`stats.stats.median`, and the 95th percentile computed from
  `stats.stats.data` with a pinned numpy
  `percentile(..., 95, method="linear")` so the P95 has a
  non-arbitrary oracle). The suite is selected by the `-m perf`
  MARKER, never by pytest-benchmark's `--benchmark-only` flag
  (Gemini-C1: `--benchmark-only` skips every test that does not
  request the `benchmark` fixture, which would SILENTLY drop PB.2 and
  PB.3 and leave the gate measuring step time alone).
- **PB.2 `test_peak_memory.py`** measures peak memory of a full
  fit+predict of the proxy. CRITICAL isolation (Gemini-C3):
  `resource.getrusage(RUSAGE_SELF).ru_maxrss` is a
  PROCESS-LIFETIME high-water mark that CANNOT be reset, so measuring
  it in the shared pytest worker would report the peak of whatever ran
  before (PB.1's benchmark rounds, collection, other tests), not the
  isolated proxy. PB.2 therefore runs the fit+predict payload in a
  FRESH child process (`multiprocessing.get_context("spawn").Process`,
  spawn not fork so the parent's already-allocated heap is not
  inherited into the child's `ru_maxrss` baseline) that calls
  `build_proxy_estimator_and_panel()`, does fit+predict, reads its OWN
  `ru_maxrss` (and `tracemalloc` peak, logged-only per Q2), and
  returns the value over a `Queue`. The parent asserts against the
  baseline. This makes the measurement order-independent and
  attributable to the proxy alone. CUDA cell: inside the same spawned
  child, `torch.cuda.reset_peak_memory_stats()` then
  `torch.cuda.max_memory_allocated()` after fit+predict (CUDA stats
  ARE resettable, but the child keeps CPU and CUDA paths symmetric and
  isolated). The JSON records the metric name actually gated per cell
  so a CPU/GPU mix is explicit, never silently comparing RSS to CUDA
  bytes. The child returns a small record over the `Queue`:
  `{value, peak_memory_metric, pid, start_method}` where
  `start_method` is the child's `multiprocessing.get_start_method()`.
  The parent does a BLOCKING `Queue.get(timeout=...)` (a pinned
  `PEAK_MEM_CHILD_TIMEOUT_S` constant) and treats an empty queue /
  timeout / non-zero child exitcode as a hard test failure
  ("peak-memory child did not return"), never a silent skip
  (post-Gemini qa-I1: a crashed/OOM-killed child must fail loudly,
  not hang or pass vacuously). `test_peak_memory_payload_runs_in_child_process`
  (a `perf`-marked PG.4 test) asserts (i) the queue record is present
  (child completed), (ii) `pid != os.getpid()` (a regression that
  inlines the payload back into the pytest process fails), AND (iii)
  `start_method == "spawn"` (post-Gemini qa-I1: on Linux the default
  context is `fork`; a regression from `get_context("spawn")` to the
  default keeps the PID distinct but silently reinherits the parent
  heap into `ru_maxrss`, so the PID check alone is insufficient and
  the start-method must be pinned explicitly).
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
- **PC.1a (module-import boundary, arch-C3; tightened by Gemini-C2).**
  `PerfBaseline`, the cell resolver (PC.2), the gate helper (PD.1),
  and `PerfRegressionError` live in `tests/perf/_gate.py`, which
  imports ONLY pydantic + stdlib at module load (no `torch`, no
  `seq_sklearn` import). CRITICAL subtlety Gemini caught:
  `seq_sklearn.hardware` does an UNCONDITIONAL module-level
  `import torch` (`src/seq_sklearn/hardware.py:14`). Therefore
  `_gate.py` must NOT do a module-level
  `from seq_sklearn.hardware import detect`; that single line would
  transitively pull torch at `_gate.py` import and silently void this
  whole boundary (and make PG.3 check (a) structurally impossible, not
  just check (c)). The cell resolver (PC.2) instead does its
  `from seq_sklearn.hardware import detect` LAZILY, inside the
  resolver function body, called only by `perf`-marked benchmark
  tests at run time. The heavy proxy (`tests/perf/_workload.py`,
  torch + `TFTClassifier`) is likewise imported only inside the
  `perf` benchmark bodies and the capture CLI. The default + coverage
  suite (acceptance criterion 2) runs PG.1/PG.2 and PG.3 checks
  (a)/(b), which import only `_gate.py` and `conftest` and NEVER call
  the resolver, so they never pull torch, keeping the fast PR job
  fast. The boundary is what PG.3 (a)/(b) assert; the resolver
  function legitimately needs torch (via `detect`) when actually
  called, so PG.3 (c) tests the resolver's branch LOGIC only and does
  NOT assert `sys.modules` after calling it (Gemini-C2 fix).
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
  `_gate.py` but BOTH `from seq_sklearn.hardware import detect` AND the
  `torch.cuda` calls are lazily imported INSIDE the resolver function
  body (not at module top), because `seq_sklearn.hardware` itself
  module-imports torch (Gemini-C2); this keeps `_gate.py`'s module
  import torch-free per PC.1a. The resolver is only ever called from
  `perf`-marked benchmark tests at run time (where torch is loaded
  anyway), never from PG.1/PG.2 or PG.3 (a)/(b).

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
  -p no:cacheprovider`. The `--benchmark-only` flag is DROPPED
  (Gemini-C1): it restricts the run to tests requesting the
  `benchmark` fixture, which would silently skip PB.2
  (`tracemalloc`/`ru_maxrss`) and PB.3 (`time.perf_counter`), leaving
  the gate measuring only PB.1 step time. The `-m perf` marker alone
  isolates the perf suite while collecting all three. The job stays
  `runs-on: ubuntu-latest` (the cpu-x86 cell) and remains in
  `nightly.yml` (not `pr.yml`): perf is nightly-only
  (implementation_plan Phase 11).
- **PF.2** Add a nightly `perf-gpu` job under the existing
  `[self-hosted, gpu]` runner (guarded by the same
  `vars.GPU_RUNNER_AVAILABLE` condition) running the same command
  (also WITHOUT `--benchmark-only`, Gemini-C1); it resolves the `t4`
  cell. `continue-on-error: true` per PE.3.
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
  (`test_capture_writes_device_name_cpu` is NOT here; it is a
  capture-path test and lives in PG.6, not in this schema file, qa
  R4-C1, see PG.6.)
- **PG.3 `test_gate_module_boundary.py`** (restructured per
  Gemini-C2: the import-boundary guard and the resolver-logic test
  are now SEPARATE tests, because calling the resolver legitimately
  imports torch via `seq_sklearn.hardware` and cannot assert it
  absent):
  - `test_gate_module_has_no_heavy_imports`: in a subprocess launched
    with `sys.executable` (qa NEW-I2: same interpreter/venv as pytest,
    not ambient `python`), run TWO checks asserting `"torch"` and
    `"seq_sklearn"` absent from `sys.modules` after each: (a)
    `import tests.perf._gate` (the module-level boundary, valid only
    because `detect` is lazy-imported inside the resolver per PC.1a /
    Gemini-C2, NOT at `_gate` module scope); (b)
    `import tests.perf.conftest` (qa R4-I2: the conftest defers its
    `import torch` into the fixture body, see PG.5, so importing it
    for PG.5's `autouse` introspection does not pull torch). The
    resolver is deliberately NOT called here; (a)/(b) are the entire
    fast-PR boundary, and PG.1/PG.2 only ever import `_gate`, never
    resolve a cell.
  - `test_cell_resolver_branch_logic` (in-process, NOT asserting
    `sys.modules`; Gemini-C2): monkeypatch `detect()` and
    `torch.cuda.get_device_name` and assert resolution only: `CPU ->
    cpu-x86`; `VOLTA_TURING` + `"T4"` device -> `t4`; `VOLTA_TURING` +
    `"V100"` -> "no baseline" (`pytest.skip` with a reason containing
    both the tier name and the device name, qa NEW-I3 / R4-I1 V100
    collision); `PASCAL`/`AMPERE_ADA`/`HOPPER`/`BLACKWELL` -> "no
    baseline". This subsumes the old `test_cell_resolver_mapping`.
    The earlier "run the resolver in a no-torch subprocess" construction
    (R3-I3 / R4-I1 / confirming-I1) is RETRACTED: its premise (the
    resolver can execute without importing torch) is false because
    `seq_sklearn.hardware:14` module-imports torch (Gemini-C2). The
    no-torch boundary is enforced solely by the two
    never-call-the-resolver import checks above.
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
- **PG.5 `conftest.py` fixture guard (I3; corrected per
  Gemini-IMPROVEMENT).** The session-scoped perf fixture ENABLES
  determinism then asserts it: it calls the library's strict-mode
  entrypoint and ONLY THEN asserts
  `torch.are_deterministic_algorithms_enabled() is True`, raising
  `RuntimeError` if the assert fails. The import path is
  `from seq_sklearn.training._determinism import enable_strict_mode`
  (post-Gemini arch-I: `seq_sklearn/training/__init__.py` has an
  empty `__all__` and does NOT re-export the symbol;
  `enable_strict_mode` is defined at
  `src/seq_sklearn/training/_determinism.py:21`; the
  `_determinism` direct import is exactly how `trainer.py:47` reaches
  it, zero source change, lower blast radius than adding a
  re-export). The earlier wording asserted the flag without anyone
  setting it, which (Gemini noted) would crash every perf run
  unconditionally since nothing in the plan turned determinism on
  before the assert. Enabling-then-asserting both guarantees PA.2's
  "determinism ON for every perf run" and still fails loudly if
  `enable_strict_mode` itself regresses. `tests/perf/conftest.py`
  does its `import torch` AND its
  `from seq_sklearn.training._determinism import enable_strict_mode`
  INSIDE the fixture function body, NOT at module level (qa R4-I2 +
  Gemini-C2: `seq_sklearn.training._determinism` module-imports torch
  at `_determinism.py:12`), so importing the conftest module to
  introspect the fixture does not pull torch; PG.3 check (b) pins
  this. The fixture is `autouse=False` and explicitly
  requested only by the `perf`-marked workload tests (qa R3-I4): a
  session `autouse` fixture would run torch at the start of the fast
  PR suite and violate the PC.1a no-torch boundary for the non-`perf`
  PG.1/PG.2/PG.3 tests. `test_perf_fixture_not_autouse` (in
  `test_gate_module_boundary.py`, non-`perf`) asserts the fixture
  object has `autouse is False` so the boundary cannot silently
  regress.
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
  - `test_capture_writes_device_name_cpu` (qa R3-C1 / R4-C1, lives
    HERE in the capture-path file, not in PG.2's schema file, because
    it must exercise the capture code, not schema construction): run
    the SAME capture entrypoint as above with cell `cpu-x86` and
    monkeypatched measurements, assert the round-tripped
    `PerfBaseline.device_name == "cpu"`. `device_name` is a free
    `str`, so schema validation alone cannot catch a capture bug that
    copies the CUDA branch on CPU or writes an empty string; running
    the capture path is the only thing that pins the PC.2
    auditability claim. The CUDA-cell analog
    (`device_name == get_device_name(0)`) is GPU-only and Deferred
    (D4) since it cannot run on CPU CI.
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
- **PG.9** `test_latency_breach_does_not_raise` lives in
  `test_gate_logic.py` (PG.1's file; it is a gate-dispatch behavior
  test, qa R4-N1): a measured `inference_latency_median_s` of
  baseline * 3.0 under mode `enforce` (matching metric,
  non-provisional) => no raise. Pins PD.1's "latency is
  observational, not gated" so an implementation that accidentally
  hard-gates latency is caught.
- **PG.10 `test_all_three_perf_tests_collected`** (Gemini-C1
  regression guard, NON-`perf`-marked so it runs in the fast suite
  via pytest collection only, no benchmark execution): use the
  `pytester` plugin (or `pytest.main(["--collect-only", "-q", "-m",
  "perf", "tests/perf/"])`) and assert the collected node ids include
  all three of `test_train_step_time`, `test_peak_memory`,
  `test_inference_latency`. This single positive assertion IS the
  load-bearing regression guard: it fails if any of the three loses
  its `perf` marker or its file is renamed, which is the exact
  Gemini-C1 silent-drop. The earlier proposed second assertion
  (collect WITH `--benchmark-only`, expect fewer than three) is
  RETRACTED (post-Gemini qa-I2): `pytest-benchmark`'s
  `--benchmark-only` filters at the RUN phase (it skips non-`benchmark`
  -fixture tests at execution), not at collection, so under
  `--collect-only` it may still collect all three and the assertion
  would be mechanistically false / flaky. WHY `--benchmark-only` is
  forbidden in PF.1/PF.2 is instead documented as a one-line code
  comment in the test referencing PB.1's run-phase rationale, not a
  mechanically asserted clause. Collection-only, so PG.10 neither
  builds the proxy nor imports torch.

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
- **D4** CUDA-cell `device_name == get_device_name(0)` capture test
  (the GPU analog of `test_capture_writes_device_name_cpu`): GPU-only,
  cannot run on CPU CI, so deferred to the nightly self-hosted GPU
  perf job's first real run. The CPU-cell value IS pinned (PG.6); the
  CUDA value is auditable on every captured row via `device_name`
  (PC.1) even without a dedicated test (qa R4-N2).

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
  - qa R3-I3: [SUPERSEDED by Gemini-C2, see "Gemini final-pass
    report" below] originally added a subprocess resolver call to
    PG.3; that construction was RETRACTED because
    `seq_sklearn.hardware:14` module-imports torch, making a
    torch-free resolver call impossible. PG.3 is now split into a
    boundary test that never calls the resolver and an in-process
    `test_cell_resolver_branch_logic`.
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

Round 4 (architecture 0C/0I/2N APPROVE; style 0/0/0 APPROVE; qa
1C/2I/2N REQUEST_CHANGES). arch + style reached consensus and judged
the doc Gemini-eligible; qa surfaced placement/precision fixes that
are mechanical (not a design disagreement), addressed in a round-4
follow-up rather than stopping at the nominal cap, since the fixes
are non-contentious and the reviewer's intent is unambiguous:

- Addressed (CRITICAL):
  - qa R4-C1: `test_capture_writes_device_name_cpu` moved out of
    PG.2 (`test_baseline_schema.py`, where a schema-only construction
    would be vacuous) into PG.6 (`test_capture_cli.py`), reusing the
    same capture entrypoint+harness so it exercises the capture code,
    not schema construction. PG.2 now carries an explicit pointer
    that the test is NOT there and why.
- Addressed (IMPROVEMENT):
  - qa R4-I1: PG.3 `test_gate_module_has_no_heavy_imports` adds a
    `VOLTA_TURING` + `get_device_name->"V100"` sub-case so a torch
    import inside the resolver's CUDA branch is caught, not only a
    bare function-top import.
  - qa R4-I2: PG.5 states `conftest.py` does `import torch` inside
    the fixture body, not at module level; PG.3 check (b) asserts
    importing `tests.perf.conftest` does not pull torch.
- Addressed (NITPICK):
  - qa R4-N1: PG.9 assigned to `test_gate_logic.py` (PG.1's file).
  - qa R4-N2: CUDA-cell `device_name` capture test recorded as
    Deferred D4 with reason (GPU-only, value still auditable via the
    `device_name` field on every captured row).

Consensus: round 4. architecture-reviewer and style-reviewer APPROVE
with zero CRITICAL/IMPROVEMENT; qa-test-coverage's round-4 CRITICAL
+ IMPROVEMENTs were mechanical doc-placement/precision items, all
resolved in-doc above (a confirming qa pass verifies closure). Two
NITPICKs (PG.3 subprocess-monkeypatch mechanism unstated; NG1 prose
density) remain, permitted by the consensus rule.

Confirming qa pass (after the round-4 follow-up): qa
0C/1I/0N APPROVE. All round-4 CRITICAL + IMPROVEMENTs verified
genuinely closed. The single new IMPROVEMENT (PG.3 V100 sub-case:
the resolver's `pytest.skip` would abort the subprocess before the
`sys.modules` snapshot) is addressed: PG.3 now specifies the
subprocess wraps the resolver call in
`try/except _pytest.outcomes.Skipped`.

CONSENSUS REACHED: architecture-reviewer APPROVE (0C/0I), style-
reviewer APPROVE (0/0/0), qa-test-coverage APPROVE (0C/0I after this
fix). Every CRITICAL resolved, every IMPROVEMENT resolved or
deferred-with-reason (D1-D4; arch R3-N2 NG1-density cosmetic
deferral). NITPICKs permitted to remain per the consensus rule. The
plan is eligible for the Gemini design final-pass.

## Gemini final-pass report (architecture-reviewer, design)

Run 2026-05-19 on the consensus'd plan (commit `fb4a39a`). Tally:
**CRITICAL 3 / IMPROVEMENT 1 / NITPICK 1, REQUEST_CHANGES.** All
findings verified by reading the cited files; none hallucinated.

- **Gemini-C1** (`PF.1/PF.2`, PB.1/PB.3): the CI command used
  `pytest -m perf --benchmark-only`. `pytest-benchmark`'s
  `--benchmark-only` collects ONLY tests requesting the `benchmark`
  fixture, silently skipping PB.2 (`tracemalloc`/`ru_maxrss`) and
  PB.3 (`time.perf_counter`). The gate would have measured step time
  alone while reporting green. VERIFIED VALID.
- **Gemini-C2** (`PC.1a`/`PC.2`/`PG.3`, `src/seq_sklearn/hardware.py:14`):
  `seq_sklearn.hardware` does an unconditional module-level
  `import torch`. A module-level `from seq_sklearn.hardware import
  detect` in `_gate.py` would transitively pull torch at `_gate`
  import, voiding the entire PC.1a fast-PR boundary (not just PG.3
  check (c)). VERIFIED VALID, and deeper than Gemini framed: it also
  invalidates the round-3/4 "run the resolver in a no-torch
  subprocess" construction, whose premise was impossible.
- **Gemini-C3** (`PB.2`): `resource.getrusage(...).ru_maxrss` is a
  non-resettable process-lifetime high-water mark; measured in the
  shared pytest worker it reports the peak of whatever ran before,
  not the proxy. Order-contaminated. VERIFIED VALID.
- **Gemini-IMPROVEMENT** (`PG.5`): the fixture asserted determinism
  was on but nothing in the plan ENABLED it; the assert would crash
  every perf run. VERIFIED VALID.
- **Gemini-NITPICK** (`PB.1`): a single Lightning `fit` step on CPU
  is dominated by trainer/dataloader init overhead. VALID; acceptable
  for a relative gate, now documented.

## Claude perspective

- **Agreed / addressed (all five):** C1 (drop `--benchmark-only`,
  add PG.10 collection guard), C2 (lazy `detect` import inside the
  resolver, retract the no-torch-subprocess resolver test, PG.3 split
  into a boundary test that never calls the resolver + an in-process
  logic test), C3 (PB.2 runs the payload in a spawned child process
  with its own `ru_maxrss`, PG.4 PID-distinctness test), IMPROVEMENT
  (PG.5 enables strict mode then asserts), NITPICK (PB.1 documents
  the overhead as relative-gate-acceptable).
- **Disagreed:** none. All findings are correct cross-family catches
  the same-family Claude swarm systematically missed (CI-tool
  interaction, a transitive module-level import, an OS-metric
  lifetime semantic, a missing enable call). This is exactly the
  class of gap the Gemini pass exists to find.
- **Missed by Gemini:** none material; the swarm-era contracts it
  did not flag remain sound under the C2 redesign.
- **Hallucinated:** none. Every `path:line` (incl.
  `hardware.py:14`) was verified accurate.

CONSENSUS INVALIDATED by three new valid CRITICALs (per the
gemini-final-pass protocol). The five findings are addressed in-doc
above; one more `/design-review` Claude round is required on the
revised plan before Phase 11 is cleared for S6 implementation. The
prior round-1..4 + confirming consensus stands for everything Gemini
did not touch; the re-review is scoped to the C1/C2/C3/IMPROVEMENT/
NITPICK revisions (PB.1, PB.2, PC.1a, PC.2, PF.1, PF.2, PG.3, PG.5,
PG.10).

## Post-Gemini re-review round (Claude swarm)

architecture 0C/1I/1N, qa 0C/2I/1N, style 0/0/0 APPROVE. Zero
CRITICAL: all three Gemini CRITICALs verified correctly and
completely closed, no new layering/import/data-flow violation. The
IMPROVEMENTs were valid doc-precision defects in the Gemini fixes
themselves, all addressed:

- Addressed (IMPROVEMENT):
  - arch-I: PG.5 named `seq_sklearn.training.enable_strict_mode`,
    but `seq_sklearn/training/__init__.py` has an empty `__all__`;
    the symbol is at `_determinism.py:21`. Repointed to
    `from seq_sklearn.training._determinism import enable_strict_mode`
    (matches the `trainer.py:47` precedent, zero source change).
    Conftest lazy-import + Gemini-C2 torch-transitivity note
    repointed to `_determinism.py:12` accordingly.
  - qa-I1: PG.4/PB.2 PID-distinctness test did not pin the
    `multiprocessing` start method; on Linux the default is `fork`,
    which reinherits the parent heap into `ru_maxrss` while keeping
    the PID distinct. The child now returns `start_method` (and the
    parent fails loudly on empty queue / timeout / nonzero exitcode);
    PG.4 asserts queue-present + distinct PID + `start_method ==
    "spawn"`.
  - qa-I2: PG.10's second assertion (collect with `--benchmark-only`,
    expect < 3) was mechanistically uncertain (`--benchmark-only`
    filters at run phase, not collection). Retracted; the positive
    "all three collected under `-m perf`" assertion is the
    load-bearing guard, the `--benchmark-only` rationale moved to a
    code comment.
- Addressed (NITPICK):
  - arch-NIT / qa-N1: the round-3 tracking entry for qa R3-I3 (which
    described the now-retracted subprocess resolver call) is marked
    SUPERSEDED-by-Gemini-C2 so an implementer reading the ledger
    cannot implement the retracted construction. style NITPICK: none
    (style 0/0/0).

POST-GEMINI CONSENSUS REACHED. The Gemini design final-pass found 3
CRITICAL + 1 IMPROVEMENT + 1 NITPICK (all valid, all addressed); the
mandatory follow-up Claude `/design-review` round on the revised
plan returned zero CRITICAL with every IMPROVEMENT resolved (arch +
qa + style). The Phase 11 plan is FINAL and cleared for S6
implementation. NITPICKs permitted to remain per the consensus rule.
