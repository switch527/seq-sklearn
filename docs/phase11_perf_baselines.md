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
  by a NAMED owning artifact, not a bare deferral (arch-C1): Phase 11
  ships `tests/perf/test_n7_absolute.py`, marked `@pytest.mark.gpu` and
  `@pytest.mark.slow` so it never runs in PR or nightly CPU CI, which
  asserts the four N7 numeric budgets on the N7 reference config. It is
  the documented evidence for acceptance criterion 9 and is run manually
  on an A100/T4/4090 as a release-checklist step (added to
  `docs/requirements.md` acceptance criteria as the criterion-9
  procedure). Phase 11's *gate* (P3) measures a small deterministic
  proxy and gates *change*; `test_n7_absolute.py` is the separate
  absolute-conformance check. The release-checklist wiring of that test
  is the only N7-absolute work; running it at N7 scale is D1.
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
  `torch_version: str`, `python_version: str`, `provisional: bool`
  (default `False`; PE.3 / qa-N3, listed here so PC.1 is the single
  field-set source of truth), and the metric fields, each a
  `float` (NOT optional, so a missing one raises at load):
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
- **PC.2** Cell identity is resolved from `HardwareTier.detect()`:
  `CPU -> cpu-x86`, `VOLTA_TURING -> t4` (T4 is Turing). Any other tier
  resolves to no baseline file and the perf tests `pytest.skip` with a
  reason naming the unmapped tier (the two public cells are the only v1
  baselines per A13/P2; other tiers are optional contributor cells, not
  a Phase 11 deliverable).

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
  Mitigations: median (not mean) over a pinned `min_rounds`;
  pytest-benchmark `warmup=True`; the gate is on median step time vs a
  15% band (already wide); the CPU cell is nightly-alert-only (PD.2), so
  a noisy false-positive pages no one and blocks no merge. P95 is
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
  - `test_unset_env_defaults_to_warn`: 1.20 breach, env var deleted =>
    no raise + warning emitted. (qa-C2; pins the safe default.)
  - `test_metric_name_mismatch_raises_even_in_warn`: baseline
    `peak_memory_metric="ru_maxrss_kb"`, measured
    `"cuda_max_alloc_bytes"`, mode `warn` => raises (PD.1a; the one
    mode-independent raise). (qa-C1.)
  - `test_provisional_baseline_never_gates`: `provisional=True`,
    1.20 breach, mode `enforce` => no raise + the
    "gating skipped: provisional" warning. (I4.)
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
- **PG.3 `test_gate_module_boundary.py`**:
  - `test_gate_module_has_no_heavy_imports`: import `tests.perf._gate`,
    assert `"torch"` and `"seq_sklearn"` are absent from `sys.modules`
    attributable to that import (subprocess-isolated, fresh
    interpreter). Pins PC.1a / arch-C3 so the fast PR suite cannot
    regress into pulling torch.
  - `test_cell_resolver_mapping`: monkeypatch `detect()` => `CPU`
    resolves `cpu-x86`, `VOLTA_TURING` resolves `t4`; `PASCAL`,
    `AMPERE_ADA`, `HOPPER`, `BLACKWELL` each resolve to "no baseline"
    and the perf path `pytest.skip`s with a reason naming the tier
    (qa Q5 silent-skip gap; PC.2).
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
  unreproducible numbers that the gate then trusts.
- **PG.6 `test_capture_cli.py`** (qa-I2): monkeypatch the three
  benchmark measurement functions to return known constants, run
  `python -m tests.perf.capture --cell cpu-x86` against `tmp_path`,
  assert the written JSON round-trips through `PerfBaseline` and the
  metric values equal the monkeypatched constants and the embedded
  `captured_git_sha` is the real `git rev-parse HEAD`. Catches a
  capture/gate schema divergence before any real nightly run.
- **PG.7 `test_check_perf_baselines_script.py`** (qa-I5): drive
  `scripts/check_perf_baselines.sh` in a `tmp_path` git repo over
  three cases: bot author + baseline change => exit 1 (the load-bearing
  arch-C2 case); baseline + source change without marker => exit 1;
  with marker => exit 0; no baseline change => exit 0. Pins the guard
  regex/branches offline, no CI required.
- **PG.8 metric-helper oracles** (qa-N2): `test_p95_matches_numpy`
  asserts the P95 helper on a known data list equals
  `np.percentile(data, 95, method="linear")`; the warm-up/repeat counts
  (PB.3) are module constants `INFERENCE_WARMUP`, `INFERENCE_REPEATS`
  asserted applied (qa-N1).

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
  `tracemalloc` peak is recorded as an observational field but not
  gated. The `peak_memory_metric` `Literal` makes the gated metric
  explicit on every row (PC.1).
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

Gemini final pass: not yet run (runs after Claude consensus).
