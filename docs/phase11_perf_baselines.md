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
  ubuntu/T4 job. Acceptance criterion 9 ("all N1-N7 met") is validated
  separately by a periodic GPU N7-scale run, out of Phase 11 scope and
  tracked as Deferred D1. Phase 11 measures a small deterministic proxy
  and gates *change*, not absolute conformance.
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
  model `PerfBaseline` (serialized via `model_dump_json`): fields
  `cell` (str: `cpu-x86` | `t4`), `captured_git_sha`,
  `torch_version`, `python_version`, and a `metrics` map
  `{metric_name -> {value: float, unit: str}}` with the fixed key set
  `train_step_median_s`, `train_step_p95_s`, `peak_memory_value`,
  `peak_memory_metric` (`ru_maxrss_kb` | `cuda_max_alloc_bytes`),
  `inference_latency_median_s`, `inference_latency_p95_s`. Loading
  validates through the pydantic model; a missing/extra key is a typed
  failure, not a silent skip.
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
- **PD.2 Nightly vs PR.** The gate raises (test failure) only under
  `SEQ_SKLEARN_PERF_GATE=enforce` (set by the nightly perf job). On
  PR-CI and locally the helper emits a warning + a structured log line
  and the test passes (A13: CPU PR regression "fires ... but does NOT
  block merge in v1 ... appears as a nightly alert"). This env gate is
  the single switch between "warn" and "block".
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
  snapshot precedent: the diff to a `_baselines/*.json` requires a
  commit-message marker `PERF_BASELINE_REVIEWED: <reason>` and a CI
  guard job (mirroring `snapshot-guard` in `pr.yml`) that fails a PR
  touching `_baselines/` without the marker. Prevents silently moving
  the goalposts to make a regression "pass".
- **PE.3** The `t4.json` baseline is captured by the nightly self-hosted
  GPU runner (A13: "Google Colab T4 captured by a nightly self-hosted
  runner"); `cpu-x86.json` is captured on `ubuntu-latest`. Until a real
  T4 capture lands, `t4.json` is committed with a `provisional: true`
  flag in the model and the GPU perf job is `continue-on-error` so a
  missing/stale T4 baseline never blocks (it is nightly-only anyway).

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

## P-G: Done-when proof (P4)

- **PG.1** `test_gate_fires_on_contrived_slowdown` (a normal unit test
  under `tests/perf/`, NOT `perf`-marked, so it runs in the default
  suite): construct a `PerfBaseline` in `tmp_path`, call
  `assert_within_baseline` with a `measured` whose step time is
  baseline * 1.20 and `SEQ_SKLEARN_PERF_GATE=enforce`; assert it raises
  the typed `PerfRegressionError`. Symmetric case: baseline * 1.05
  (under the 15% band) does NOT raise. A memory variant at baseline *
  1.20 vs the 10% band. This proves the gate logic without needing a
  real regression and discharges the P4 "fires correctly on a contrived
  20% slowdown" clause deterministically and offline.
- **PG.2** `test_perf_baselines_present_and_valid` (default suite,
  offline): both `_baselines/cpu-x86.json` and `t4.json` load and
  validate through `PerfBaseline` with the full key set, discharging the
  P2/P4 "produces a baseline JSON for the two public cells" clause as a
  checked-in artifact (the nightly job refreshes them; their existence
  and schema are pinned by a fast offline test).

## Open questions for the swarm

- **Q1** Proxy size `(N=256, P=24, L=12)`: large enough to be
  representative and stable under CPU noise, small enough for nightly?
  Trade-off between signal and runner-time. (PA.1)
- **Q2** Is RSS (`ru_maxrss`) the right CPU analog for N7's
  GPU-memory budget, or should the CPU cell gate `tracemalloc` peak
  (Python-only, more stable but ignores torch C++/MKL allocations)?
  (PB.2)
- **Q3** Should P95 step time also be gated (a second band), or is
  median-only the right v1 gate with P95 observational? A13 names only
  median step + peak memory. (PD.1)
- **Q4** `t4.json` provisional bootstrap (PE.3): commit a
  hand-seeded provisional T4 baseline now, or commit only after a real
  self-hosted capture and skip the t4 gate until then? Affects whether
  P2 is satisfied at Phase 11 merge or deferred to first nightly GPU.

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

(populated by `/design-review`)
