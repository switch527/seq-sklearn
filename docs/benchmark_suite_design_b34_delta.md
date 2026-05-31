# B34 delta: BootstrapResult dataclass refactor (D-B21.4)

## Requirements

R-B34-1 (closes D-B21.4): replace the
`entity_block_bootstrap_ci` 4-tuple return with a frozen
`BootstrapResult` pydantic BaseModel. The model carries 4
named fields matching the current tuple positions: `mean`,
`ci_lo`, `ci_hi`, `fallback_reason`. Production callers
and tests access via attribute names. Improves code-clarity
by removing positional-tuple coupling without changing any
behavior.

## Non-requirements

- v1 does NOT add the `ci_method` field to the result that
  the original D-B21.4 text mentioned. Adding ci_method
  would echo back the caller-supplied parameter; no
  consumer needs it (D-B21.3 + D-B21.2 are YAGNI-deferred
  so per-method dispatch doesn't exist).
- v1 does NOT add `__iter__` for backward-compat tuple
  unpacking. All ~25 call sites get updated to attribute
  access; no transition period needed.
- v1 does NOT change behavior. Pure refactor: same returns,
  same fallback semantics, same numerical values.

## B34.0 Background

`entity_block_bootstrap_ci` at
`benchmarks/metrics/bootstrap.py:129-246` currently returns
`tuple[float, float, float, str | None]`. 23 call sites
across the repo do tuple-unpack:
- 8 production sites in `benchmarks/report/` (bootstrap_rollup,
  bootstrap_pairwise, bootstrap_training_time,
  bootstrap_hpo_uplift, bootstrap_ensemble_lift; some
  modules call twice for main + oracle).
- 15+ test sites in
  `tests/benchmarks/test_b21_bca_ci.py`,
  `test_bootstrap.py`, and 5 other phase test files.

The original D-B21.4 deferral framed this as "becomes
useful when D-B21.1 surfaces additional audit fields". Now
that B24 closed D-B21.1 and surfaced the BCa health
footnote, the additional clarity has real value (callers
don't need to know which positional slot carries the
fallback_reason).

## B34.1 BootstrapResult schema

In `benchmarks/metrics/bootstrap.py`:

```python
class BootstrapResult(BaseModel):
    """Frozen result type for entity_block_bootstrap_ci.

    Replaces the prior 4-tuple return per B34 / D-B21.4.
    Fields preserve the order and semantics of the tuple
    positions exactly: mean, ci_lo, ci_hi, fallback_reason.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean: float
    ci_lo: float
    ci_hi: float
    fallback_reason: str | None = None
```

Lives alongside `_bca_percentile_points` and the other
private helpers in the same module.

## B34.2 entity_block_bootstrap_ci signature change

Return type annotation changes from
`tuple[float, float, float, str | None]` to
`BootstrapResult`. The 3 return statements update:

- Degenerate `n_entities <= 1`:
  `return BootstrapResult(mean=ground_truth_mean,
  ci_lo=ground_truth_mean, ci_hi=ground_truth_mean,
  fallback_reason=None)`
- Percentile method path:
  `return BootstrapResult(mean=ground_truth_mean,
  ci_lo=ci_lo, ci_hi=ci_hi, fallback_reason=None)`
- BCa method path:
  `return BootstrapResult(mean=ground_truth_mean,
  ci_lo=ci_lo, ci_hi=ci_hi,
  fallback_reason=fallback_reason)`

Docstring "Returns" section updated to name the
`BootstrapResult` type.

## B34.3 Call-site updates

23+ sites change from:
```python
mean, ci_lo, ci_hi, fallback_reason = entity_block_bootstrap_ci(...)
```
to:
```python
result = entity_block_bootstrap_ci(...)
mean = result.mean
ci_lo = result.ci_lo
ci_hi = result.ci_hi
fallback_reason = result.fallback_reason
```

OR more concisely (preserving the original 4-name unpack):
```python
result = entity_block_bootstrap_ci(...)
# ... use result.mean, result.ci_lo, ... directly ...
```

Each call site picks the form that matches its existing
variable usage. Some sites only use 1-2 fields and will
inline (`mean = entity_block_bootstrap_ci(...).mean` or
similar).

### B34.3.1 Per-file site count

| File | Sites |
|---|---|
| `benchmarks/report/_bootstrap_aggregate.py` | 1 |
| `benchmarks/report/bootstrap_rollup.py` | 1 |
| `benchmarks/report/bootstrap_pairwise.py` | 1 |
| `benchmarks/report/bootstrap_training_time.py` | 1 |
| `benchmarks/report/bootstrap_hpo_uplift.py` | 1 |
| `benchmarks/report/bootstrap_ensemble_lift.py` | 2 (main + oracle) |
| `tests/benchmarks/test_bootstrap.py` | ~10 |
| `tests/benchmarks/test_b21_bca_ci.py` | ~9 |
| Other test files | 5+ |

Total: ~30 sites.

## B34.4 Tests

Baseline (post-B33): 1084 tests.

### B34.4.1 BootstrapResult schema tests

1. `test_bootstrap_result_accepts_required_fields`: construct
   with all 4 fields; assert reads back correctly.
2. `test_bootstrap_result_accepts_none_fallback_reason`:
   construct with `fallback_reason=None`; default is None
   so omitting it works too.
3. `test_bootstrap_result_rejects_extra_field`: construct
   with an unknown kwarg; assert `pytest.raises(ValidationError)`
   per `extra="forbid"`.
4. `test_bootstrap_result_is_frozen`: assert attempting to
   set `result.mean = 999` raises (frozen).
5. `test_entity_block_bootstrap_ci_returns_bootstrap_result_instance`:
   call the primitive; assert
   `isinstance(result, BootstrapResult)`.

### B34.4.2 Test-suite migration

All existing tests that unpacked the 4-tuple migrate to
attribute access. The migration preserves test semantics;
no test assertions change in content (only the unpack
form). Net test count delta: +5 named (B34.4.1).

### B34.4.3 Expected test delta

Baseline: 1084.
- 5 new BootstrapResult tests.
- 23+ migrated call sites (no test count change).
- Total: 1084 + 5 = 1089.

## B34.5 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B34-Risk-1 | A call site missed during migration produces a TypeError at runtime (BootstrapResult doesn't support tuple-unpack). | Low | Full test suite is the gate; ruff/pyright catches type mismatches. Migrating in a single commit means no partial state. |
| R-B34-Risk-2 | The new BaseModel adds parquet round-trip surface. | None | BootstrapResult is the in-memory primitive return; it never gets persisted to parquet (the aggregators extract numeric fields and store them in the rollup row schemas). No new write/load contract. |
| R-B34-Risk-3 | The `frozen=True` config breaks any caller that mutates the result. | Low | No current caller mutates; mutation would have been a code smell pre-refactor anyway. |

## Deferred

(None added.)
