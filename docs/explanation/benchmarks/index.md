# Benchmark-suite methodology

The `benchmarks/` package ships a four-experiment comparative
suite for seq-sklearn against documented baselines. Each
experiment lives behind a CLI dispatch arm and produces its own
Markdown report; the cross-experiment assembler stitches them
into `report.md` next to a fingerprinted `run_manifest.json`.

This page is the methodology index. The full design + impl plan
live under `docs/benchmark_suite_design.md` and
`docs/benchmark_suite_implementation_plan.md`.

## Experiments

The harness runs four experiments in topological order
(`raw_loss` first so the report-only kinds have data to read):

| Order | Kind | Report | Spec |
|---|---|---|---|
| 1 | `raw_loss` (B5 / B6.1) | `leaderboard.md` | per-cell B4 metrics with primary-loss ranking |
| 2 | `ensemble` (B6 / B6.2) | `pairwise.md` | per-pair complementarity from cached predictions |
| 3 | `training_time` (B7 / B6.3) | `training_time.md` | per-(dataset, model, hardware_tier) fit wall-clock + memory |
| 4 | `hpo_uplift` (B8 / B6.4) | `hpo_uplift.md` | per-(dataset, model) tuned-vs-default Δ + Friedman/Holm |

Topological order is enforced at CLI dispatch
(`benchmarks/run.py::_DISPATCH_ORDER`); a config that declares
the kinds out of order still runs in the canonical sequence.

## Reproducibility (B8.1 + B9)

Every CLI invocation writes a `run_manifest.json` to
`output_root/` BEFORE any experiment touches the filesystem. The
manifest records the library + benchmarks SHAs, the dependency
versions, the environment fingerprint (CPU + GPU model, CUDA
runtime + driver), the dataset + model roster, the resolved
experiment specs, and the run timestamps.

A crashed run leaves the manifest of intent on disk. A clean exit
rewrites the manifest with `completed_at_utc` populated and
assembles `report.md` from the four per-experiment Markdown files
plus the manifest's exec-summary fields.

Re-running from the same manifest on the same code (library SHA +
dependency versions) reproduces the same outputs within the GPU
parity tolerance pinned at `tests/integration/test_gpu_cpu_parity.py`
(atol=1e-5, rtol=1e-5).

## Cell-level vs run-level state

Two manifest layers coexist under `output_root/`:

- **Per-cell** (`results/*.parquet` + `results/_done/*.json`):
  one `ResultRow` per `(dataset, model, seed, variant, fold)`
  cell, written atomically via the B7.2 shard-then-sentinel
  pattern. `benchmarks.manifest.load_run(output_root)`
  concatenates the shards into a single DataFrame.
- **Per-run** (`run_manifest.json`): one `RunManifest` per CLI
  invocation. Carries everything the per-cell rows can't, in
  particular the dependency versions + env fingerprint.

The two layers join on `run_id`: `RunManifest.run_id` equals
every cell's `ResultRow.run_id` for cells written by the same
invocation.

## Search-space parity disclosure (B6.4.0)

The HPO uplift report names the search-space dimensionality for
each model family next to the Δ. v1 does NOT normalize the
dimensionality across families (a deep-model search space and a
GBM search space are structurally different); the asymmetry is
reported, not hidden.

## Per-experiment design notes

Each per-experiment renderer's section in `report.md` quotes the
underlying Markdown verbatim. The full per-cell rows remain under
`results/` for downstream audit (e.g., re-running statistical
tests against the raw cell distribution rather than the seed-mean
the report aggregates).
