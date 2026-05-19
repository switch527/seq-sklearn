"""Phase 11 perf constants (PA.1 / PD.3).

stdlib-only, NO torch, NO seq_sklearn (PC.1a / Gemini-C2). Anything
that needs only a tuning constant imports from HERE, never from
`_workload` (which pulls torch). This keeps the fast PR suite's
non-`perf` tests (the proxy-budget and constants-applied tests)
torch-free, the boundary the post-Gemini code review flagged.
"""

__all__ = [
    "BENCH_MIN_ROUNDS",
    "INFERENCE_BATCH",
    "INFERENCE_REPEATS",
    "INFERENCE_WARMUP",
    "PEAK_MEM_CHILD_TIMEOUT_S",
    "PROXY_ATTENTION_HEADS",
    "PROXY_BUILD_TIMEOUT_S",
    "PROXY_HIDDEN_SIZE",
    "PROXY_L",
    "PROXY_N",
    "PROXY_P",
    "PROXY_SEED",
]

# Proxy size (PA.1): L matches the N7 reference lookback; N/P scaled
# for a CPU nightly envelope. Changing any of these invalidates the
# checked-in baselines and is a PERF_BASELINE_REVIEWED: change.
PROXY_N = 256
PROXY_P = 24
PROXY_L = 12
PROXY_SEED = 11
PROXY_HIDDEN_SIZE = 128  # N7 reference architecture
PROXY_ATTENTION_HEADS = 4  # N7 reference architecture

# Named tuning constants (PD.3 / arch R3-N; asserted applied by PG.8).
BENCH_MIN_ROUNDS = 5
INFERENCE_WARMUP = 3
INFERENCE_REPEATS = 20
INFERENCE_BATCH = 1024  # N7 latency reference size

# Subprocess timeouts (qa-I1 / R1): an oversized proxy or a
# crashed/OOM-killed child must fail loudly, never hang.
PROXY_BUILD_TIMEOUT_S = 180
PEAK_MEM_CHILD_TIMEOUT_S = 240
