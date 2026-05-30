# B20 design delta: bootstrap CI on the per-sample-best oracle Δ (D-B16.1)

**Scope**: D-B16.1 adds a bootstrap CI on the per-sample-best
oracle Δ for the B16 ensemble-lift aggregator. The B11 driver
already computes a scalar `oracle_loss_mean` and
`oracle_delta_loss_mean` on `PerDatasetLift` (the ceiling
that "if we always picked the right ensemble per-row, we
would have achieved this loss"). The B16 aggregator
bootstraps the GBM-vs-GBM+seq Δloss but not the oracle Δ;
the renderer's `Δloss(oracle)` column is therefore a scalar
without a confidence interval, while the achieved Δloss
column carries a `[mean lo, hi]` CI.

The fix: add 3 new fields on `EnsembleLiftRollupRow`
(`oracle_metric_mean`, `oracle_metric_ci_lo`,
`oracle_metric_ci_hi`) plus 1 audit field
(`n_oracle_cells_paired`) for the bootstrap-CI count. The
aggregator builds a per-cell oracle Δ array using each
cell's `loss_gbm - oracle_loss` (dropping cells where
`oracle_loss is None`), bootstraps it with the same
`entity_block_bootstrap_ci` primitive, and writes the
results. The renderer replaces the bare `Δloss(oracle)`
column with `Δloss(oracle) [95% CI]`.

The existing `loss(oracle)` column stays as a scalar (it is
not a Δ; it's an absolute loss ceiling).

## Requirements

The grading rubric for every reviewer finding traces back to
one of these.

- **R-B20-1** `EnsembleLiftRollupRow` gains FOUR new fields,
  placed to match the schema's existing group structure
  (arch-R1-I1 closure):
  - `n_oracle_cells_paired: int = Field(ge=0)` joins the
    COUNTS group immediately after `n_pair_grid`
    (before `n_skipped_cells`); it is a per-row count of
    cells that contributed to the oracle bootstrap,
    semantically parallel to `n_cells_paired` and
    `n_pair_grid`.
  - `oracle_metric_mean: float | None = None`,
    `oracle_metric_ci_lo: float | None = None`,
    `oracle_metric_ci_hi: float | None = None` join the
    METRIC-STATS group immediately after the
    `primary_metric_*` triple (before `bootstrap_seed`);
    they are CI statistics on a per-row metric, parallel
    to the `primary_metric_*` triple.
  The four other RollupRow schemas (B5/B6/B7/B15) are NOT
  touched.
- **R-B20-2** The B16 aggregator
  (`aggregate_bootstrap_ensemble_lift_rollup`) computes the
  per-cell oracle Δ array as
  `[loss_gbm - oracle_loss for cell in computed.cells if
  cell.oracle_loss is not None]` and bootstraps it with the
  same primitive used for the main `delta_loss` array
  (`entity_block_bootstrap_ci` with cell-index as
  entity-id, same `bootstrap_n_resamples` resolved from the
  same spec). Reuses the same `BOOTSTRAP_ROW_COUNT_CEILING`
  defensive guard.
- **R-B20-2a** (arch-R1-C1 closure): the two bootstrap
  invocations use DERIVED seeds, NOT the same
  `BOOTSTRAP_DEFAULT_SEED`. The main Δloss bootstrap keeps
  `seed=BOOTSTRAP_DEFAULT_SEED` (B16 contract); the oracle
  Δ bootstrap uses
  `seed=BOOTSTRAP_DEFAULT_SEED ^ BOOTSTRAP_ORACLE_SEED_OFFSET`
  where `BOOTSTRAP_ORACLE_SEED_OFFSET` is a new module-
  level constant in `benchmarks/report/_bootstrap_aggregate.py`
  (a fixed prime XOR mask, e.g., `0xB20_07A_C7E`). This is
  REQUIRED because
  `benchmarks/metrics/bootstrap.py:138` constructs
  `rng = np.random.Generator(np.random.PCG64(seed))`: the
  PCG64 stream depends ONLY on `seed`. Two invocations
  with the same seed AND the same `n_entities` produce
  IDENTICAL resampled indices, which would make the main
  Δloss and oracle Δ CIs correlated on shared cells. The
  XOR offset gives the two bootstraps independent PCG64
  streams.
- **R-B20-3** Sentinel rows pass all 4 new fields as
  documentary defaults: `oracle_metric_mean=None`,
  `oracle_metric_ci_lo=None`, `oracle_metric_ci_hi=None`,
  `n_oracle_cells_paired=0`. The renderer's
  `bootstrap_skipped_reason is not None` short-circuit
  surfaces `(no CI)` before any oracle-column read, so the
  defaults are documentary.
- **R-B20-4** When no per-cell oracle Δ is computable
  (e.g., every cell has `oracle_loss=None` because all
  pairs are classification with mismatched proba columns),
  the aggregator emits the row with the bootstrap fields
  filled for the main Δloss (as today) AND
  `oracle_metric_*=None`, `n_oracle_cells_paired=0`. NO
  new sentinel string is added; the renderer falls back to
  `(no CI)` in the oracle column only.
- **R-B20-5** The renderer
  (`_render_complete_table_with_ci`) replaces the bare
  `Δloss(oracle)` column with `Δloss(oracle) [95% CI]` and
  formats it via the shared `format_ci_cell` helper. The
  `loss(oracle)` column stays as a scalar (no CI on the
  absolute loss). The std variant is NOT touched.
- **R-B20-6** A separate partial-coverage asterisk on the
  oracle CI cell fires when `n_oracle_cells_paired <
  n_pair_grid` (i.e., the oracle bootstrap covered fewer
  cells than the intersection grid). The existing
  `n_cells_paired < n_pair_grid` asterisk on the main
  Δloss column is unchanged.
- **R-B20-7** The B17 Guard A invariant (no
  `primary_loss_*` field on any RollupRow other than
  `primary_loss_column`) continues to hold. The four new
  oracle fields use `oracle_metric_*` and
  `n_oracle_cells_paired` naming, not `oracle_loss_*` or
  any `primary_loss_*` form.

## B20.0 What the change actually adds

Schema (after B19) currently groups fields as:

```python
class EnsembleLiftRollupRow(BaseModel):
    # IDENTITY
    dataset_name: str
    task_type: str
    primary_metric: str
    primary_loss_column: str
    # COUNTS
    n_seeds: int = Field(ge=0)
    n_folds: int = Field(ge=0)
    n_cells_paired: int = Field(ge=0)
    n_pair_grid: int = Field(ge=0)
    n_skipped_cells: int = Field(ge=0)
    # METRIC STATS
    primary_metric_mean: float | None = None
    primary_metric_ci_lo: float | None = None
    primary_metric_ci_hi: float | None = None
    # BOOTSTRAP CONFIG
    bootstrap_seed: int
    ...
```

New: insert ONE field at the end of the COUNTS group AND
THREE fields at the end of the METRIC STATS group
(arch-R1-I1 closure):

```python
class EnsembleLiftRollupRow(BaseModel):
    # IDENTITY (unchanged)
    ...
    # COUNTS
    n_seeds: int = Field(ge=0)
    n_folds: int = Field(ge=0)
    n_cells_paired: int = Field(ge=0)
    n_pair_grid: int = Field(ge=0)
    # B20 / D-B16.1: per-row count of cells that contributed to the
    # oracle bootstrap (i.e., cells whose oracle_loss was not None).
    # Parallel to n_cells_paired which counts cells that contributed
    # to the main delta_loss bootstrap. May be less than or equal to
    # n_cells_paired.
    n_oracle_cells_paired: int = Field(ge=0)
    n_skipped_cells: int = Field(ge=0)
    # METRIC STATS
    primary_metric_mean: float | None = None
    primary_metric_ci_lo: float | None = None
    primary_metric_ci_hi: float | None = None
    # B20 / D-B16.1: CI on the per-sample-best oracle delta. The
    # aggregator computes per-cell oracle delta as
    # `loss_gbm - oracle_loss` (dropping cells where oracle_loss
    # is None) and bootstraps with the same primitive but a
    # DERIVED seed (BOOTSTRAP_DEFAULT_SEED ^ BOOTSTRAP_ORACLE_SEED_OFFSET)
    # so the two PCG64 streams are independent. Sentinel rows leave
    # these as None; happy-path rows with no computable per-cell
    # oracle leave these as None and set n_oracle_cells_paired=0.
    oracle_metric_mean: float | None = None
    oracle_metric_ci_lo: float | None = None
    oracle_metric_ci_hi: float | None = None
    # BOOTSTRAP CONFIG (unchanged)
    bootstrap_seed: int
    ...
```

The other 4 RollupRow schemas (B5/B6/B7/B15) are NOT
touched.

## B20.1 Aggregator changes

In `aggregate_bootstrap_ensemble_lift_rollup` (after the
existing per-cell Δ-loss bootstrap):

```python
# Existing: bootstrap on the main delta_loss array
deltas = np.array([c.delta_loss for c in computed.cells], dtype=float)
...
mean, ci_lo, ci_hi = entity_block_bootstrap_ci(
    deltas, entity_ids, n_resamples=n_resamples, ...
)

# NEW B20 / D-B16.1 (post Build R1 arch-I3 collapse):
# single-pass np.array materialisation matching the main path's
# idiom; n_oracle_cells_paired = oracle_deltas.shape[0].
oracle_deltas = np.array(
    [
        cell.loss_gbm - cell.oracle_loss
        for cell in computed.cells
        if cell.oracle_loss is not None
    ],
    dtype=float,
)
n_oracle_cells_paired = int(oracle_deltas.shape[0])
if n_oracle_cells_paired == 0:
    oracle_mean = None
    oracle_ci_lo = None
    oracle_ci_hi = None
else:
    if not np.isfinite(oracle_deltas).all():
        raise RawRollupError(
            f"aggregate_bootstrap_ensemble_lift_rollup: dataset="
            f"{dataset_name!r} has a paired cell with a non-finite "
            "oracle delta; the upstream predictions shard is corrupt"
        )
    # arch-R1-I2 closure: defensive even though
    # n_oracle_cells_paired <= n_cells_paired in production (the
    # main bootstrap's gate above fires first on the equal-or-
    # larger main array). Test #5 forces this branch by mutating
    # `BOOTSTRAP_ROW_COUNT_CEILING` between the main and oracle
    # calls so the main gate is suppressed.
    if (
        n_oracle_cells_paired * n_resamples
        > _bootstrap_aggregate.BOOTSTRAP_ROW_COUNT_CEILING
    ):
        raise RawRollupError(...)
    oracle_entity_ids = np.arange(n_oracle_cells_paired, dtype=np.int64)
    oracle_mean, oracle_ci_lo, oracle_ci_hi = entity_block_bootstrap_ci(
        oracle_deltas, oracle_entity_ids,
        n_resamples=n_resamples, confidence=BOOTSTRAP_CONFIDENCE,
        seed=BOOTSTRAP_DEFAULT_SEED ^ _bootstrap_aggregate.BOOTSTRAP_ORACLE_SEED_OFFSET,
    )
```

Where `BOOTSTRAP_ORACLE_SEED_OFFSET` is a new module-level
constant exported from
`benchmarks/report/_bootstrap_aggregate.py` (a fixed XOR
mask, e.g., `0xB20_07A_C7E`, chosen to be a high-entropy
hex token so the resulting seed differs in every bit from
the main bootstrap seed). The same OOM ceiling and same
n_resamples; the two bootstraps now have INDEPENDENT PCG64
streams because `seed` is the only parameter that
determines the `np.random.PCG64(seed)` state and the XOR
ensures the two seeds differ.

The `_emit_sentinel_row` helper hardcodes the 4 oracle fields
inline (`n_oracle_cells_paired=0` and `oracle_metric_*=None`)
per Build R1 arch-I2 closure: the helper originally took the
4 oracle fields as defaulted keyword parameters, but every
caller used the defaults and the renderer's
`bootstrap_skipped_reason is not None` short-circuit discards
any non-default value, so widening the signature added no
production value. Test #12 still pins the sentinel-emit
contract end-to-end via `aggregate_bootstrap_ensemble_lift_rollup`.

## B20.2 Renderer changes

In `_render_complete_table_with_ci` at
`benchmarks/report/ensemble_lift.py:116-172`:

Old:

```python
header_cells = [
    ...
    "loss(oracle)",
    "Δloss(oracle)",
]
...
row_cells = [
    ...
    _format_value(row.oracle_loss_mean),
    _format_value(row.oracle_delta_loss_mean),
]
```

New:

```python
header_cells = [
    ...
    "loss(oracle)",
    "Δloss(oracle) [95% CI]",
]
...
# Oracle CI cell: same shape contract as the main Δloss CI cell.
# (no CI) when the rollup row has a sentinel OR when
# n_oracle_cells_paired=0 (no computable per-cell oracle).
# Partial-coverage asterisk on `n_oracle_cells_paired < n_pair_grid`
# (independent from the main Δloss column's asterisk).
if rollup_row is None or rollup_row.bootstrap_skipped_reason is not None:
    oracle_ci_cell = "(no CI)"
elif rollup_row.n_oracle_cells_paired == 0:
    oracle_ci_cell = "(no CI)"
else:
    oracle_partial = (
        rollup_row.n_oracle_cells_paired < rollup_row.n_pair_grid
        and rollup_row.n_pair_grid > 0
    )
    oracle_ci_cell = format_ci_cell(
        rollup_row.oracle_metric_mean,
        rollup_row.oracle_metric_ci_lo,
        rollup_row.oracle_metric_ci_hi,
        partial=oracle_partial,
    )

row_cells = [
    ...
    _format_value(row.oracle_loss_mean),
    oracle_ci_cell,
]
```

The std variant
(`_render_complete_table_std` and `_render_dataset_table`)
is NOT touched; it keeps the bare `Δloss(oracle)` scalar
column reading from `row.oracle_delta_loss_mean`.

**Module docstring update** (arch-R1-I4 closure): the
`benchmarks/report/ensemble_lift.py` module docstring at
the file head currently describes the CI variant as
replacing only the bare `Δloss` + `Δstd` columns. B20
updates this docstring to mention that the CI variant
also adds a `Δloss(oracle) [95% CI]` column (replacing the
bare `Δloss(oracle)` scalar) when the rollup carries
oracle bootstrap fields.

**Footnote-precedence note** (arch-R1-I5 closure): the
CI-variant body acquires a sixth implicit footnote source
for the oracle CI when `n_oracle_cells_paired <
n_cells_paired` (the oracle bootstrap covered fewer cells
than the main bootstrap, e.g., one or more cells had
`oracle_loss=None`). At v1 the renderer surfaces this only
as the trailing asterisk on the oracle CI cell; an
explicit footnote table block is deferred under D-B20.1
(coordinated with the existing partial-coverage footnote
infrastructure).

**Naming clarity** (arch-R1-I3 closure): the new schema
fields use the `oracle_metric_*` prefix to parallel the
existing `primary_metric_*` block (a B17 rename pattern).
The `oracle_metric_mean` field carries the mean of the
bootstrap statistic, which IS the oracle delta
(`loss_gbm - oracle_loss` mean over the resampled cells).
This is intentionally a parallel of `primary_metric_mean`
which carries the main delta. The pre-B16 scalar
`oracle_delta_loss_mean` field on `PerDatasetLift` stays
unchanged for the std-variant renderer; the new B20
fields are on the rollup row and feed the CI-variant
renderer only.

## B20.3 Test surface

### Existing tests touched

Each fixture site that constructs `EnsembleLiftRollupRow`
needs 4 new kwargs (qa-R1-N2 closure: concrete defaults
enumerated below so byte-pin tests get reproducible
values):

1. **`tests/benchmarks/test_bootstrap_manifest.py`**: the
   `_make_ensemble_lift_row` factory defaults
   `oracle_metric_mean=0.10`, `oracle_metric_ci_lo=0.08`,
   `oracle_metric_ci_hi=0.12`, `n_oracle_cells_paired=6`
   (matches the symmetric-roster happy path; same value as
   `n_cells_paired`).
2. **`tests/benchmarks/test_bootstrap_ensemble_lift.py`**:
   the 17 aggregator tests stub
   `compute_per_cell_lift_deltas` to return controlled
   `PerCellLiftDelta` records; they don't construct
   rollup rows directly. The aggregator tests pass through
   to the aggregator's new oracle-bootstrap code path; the
   existing assertions are unchanged. New B20-specific
   tests pin the new behavior (B20.4 above).
3. **`tests/benchmarks/test_run_bootstrap_ensemble_lift_wrapper.py`**:
   13 wrapper tests; don't construct rollup rows directly.
   Pass byte-equivalent.
4. **`tests/benchmarks/test_ensemble_lift_report_b16.py`**:
   the `_make_rollup_row` factory adds the 4 new kwargs
   with the same documentary defaults
   (`oracle_metric_mean=0.10, oracle_metric_ci_lo=0.08,
   oracle_metric_ci_hi=0.12`). The
   `n_oracle_cells_paired` kwarg uses the
   PYTHON-LEVEL-NONE factory pattern that mirrors the B19
   `n_pair_grid` factory pattern (qa-R2-N1 closure: the
   factory signature is `n_oracle_cells_paired: int |
   None = None` AND the factory body does `if
   n_oracle_cells_paired is None: n_oracle_cells_paired =
   n_cells_paired` before calling the pydantic
   constructor; the SCHEMA field at R-B20-1 stays
   `n_oracle_cells_paired: int = Field(ge=0)` with no
   Python-level default, so passing None to the schema
   would fail validation; the fallback lives in the
   factory body only). 24 renderer tests continue to
   pass; the existing "Δloss(oracle)" column header
   check is updated to "Δloss(oracle) [95% CI]" (one
   test).
5. **`tests/benchmarks/test_b17_byte_identity_pins.py`**:
   the `_make_ensemble_lift_rollup` direct construction
   gets `oracle_metric_mean=0.20, oracle_metric_ci_lo=0.15,
   oracle_metric_ci_hi=0.25, n_oracle_cells_paired=2`
   (matching the existing `primary_metric_*` values and
   `n_pair_grid=2` from B19's byte-pin closure). The
   byte-pin's mandatory-asterisk regex on the MAIN Δloss
   CI cell continues to fire; the oracle CI cell is a
   separate cell on the same row.
6. **`tests/benchmarks/test_b19_n_pair_grid.py`**: the 4
   helper-constructed rollup rows (in tests #3, #4, #5,
   #8) get
   `oracle_metric_mean=0.10, oracle_metric_ci_lo=0.08,
   oracle_metric_ci_hi=0.12,
   n_oracle_cells_paired=<test-specific>` (mirroring the
   test-specific `n_pair_grid` value where applicable).
   The 8 aggregator-output tests continue to pass; the
   existing assertions don't touch the oracle fields.

### NEW B20 tests

`tests/benchmarks/test_b20_oracle_delta_ci.py` (NEW; 13
tests):

1. `test_aggregator_writes_oracle_ci_on_happy_path`:
   fixture with 4 cells where `oracle_loss` is finite on
   all 4. Assert `n_oracle_cells_paired == 4`, oracle_metric_mean
   is finite, oracle_metric_ci_lo <= oracle_metric_mean <=
   oracle_metric_ci_hi.
2. `test_aggregator_skips_cells_with_none_oracle_loss`:
   fixture with 4 cells where 2 have `oracle_loss=None`.
   Assert `n_oracle_cells_paired == 2` (only the 2
   finite-oracle cells contributed); oracle_metric_mean
   computed from those 2 cells.
3. `test_aggregator_writes_oracle_none_when_all_cells_lack_oracle_loss`:
   fixture with 4 cells where ALL have `oracle_loss=None`.
   Assert `n_oracle_cells_paired == 0` AND
   `oracle_metric_mean is None` AND
   `oracle_metric_ci_lo is None` AND
   `oracle_metric_ci_hi is None` AND the row's main
   `primary_metric_*` fields are still populated (the main
   Δloss bootstrap is independent).
4. `test_aggregator_oracle_delta_sign_convention_is_loss_gbm_minus_oracle`:
   asymmetric fixture: `loss_gbm=0.50, oracle_loss=0.30,
   loss_gbm_plus_seq=0.45` per cell so per-cell oracle Δ =
   0.20 AND per-cell main Δloss = 0.05 (qa-R1-N1 closure:
   the main Δloss = 0.05 is intentionally DISTINCT from
   the oracle Δ = 0.20 so a cross-wire mutation where the
   aggregator writes the main bootstrap into the oracle
   fields would yield oracle_metric_mean = 0.05 and fail
   the bare-equality `== 0.20` assertion). Assert
   `oracle_metric_mean == pytest.approx(0.20, abs=1e-9)`
   AND `oracle_metric_mean != pytest.approx(0.05, abs=1e-9)`
   (cross-wire mutation pin) AND `primary_metric_mean ==
   pytest.approx(0.05, abs=1e-9)` (the main bootstrap
   wrote the right value to the right field).
5. `test_aggregator_oracle_oom_gate_raises`:
   monkeypatch `BOOTSTRAP_ROW_COUNT_CEILING=1` AND set
   `bootstrap_n_resamples=2` on the spec (qa-R2-N2
   closure: explicit value so the gate condition
   `n_oracle_cells * 2 > 1` fires deterministically);
   fixture with 2 finite-oracle cells AND
   `compute_per_cell_lift_deltas` stubbed to return only
   1 record on the MAIN delta (so `1 * 2 = 2 > 1` would
   fire the MAIN gate first; the test must arrange this
   so the MAIN bootstrap is suppressed entirely; the
   simplest path is to stub the MAIN gate out of band by
   ensuring `n_cells_paired == 0` while
   `n_oracle_cells_paired == 2`, which requires a
   carefully-built stub). Assert
   `pytest.raises(RawRollupError, match=r"oracle delta")`
   so the message clearly distinguishes which bootstrap
   tripped the gate.
6. `test_aggregator_oracle_nan_delta_raises_via_stub`
   (qa-R1-C1 closure: name the injection seam): stub
   `compute_per_cell_lift_deltas` (the SAME monkeypatch
   pattern used by the existing B16 aggregator tests at
   `test_bootstrap_ensemble_lift.py:_stub_per_cell_*`) to
   return a `ComputePerCellLiftDeltasResult` whose cells
   list contains a single record with
   `loss_gbm=float('nan'), oracle_loss=0.30,
   delta_loss=0.20` (delta_loss finite, oracle delta
   non-finite). Assert
   `pytest.raises(RawRollupError, match=r"non-finite oracle delta")`
   (qa-R2-I1 closure: explicit `match=` clause so a guard
   message rename is caught) AND confirm the main
   delta_loss bootstrap was not the one that tripped (the
   main delta is finite). This structurally exercises the
   new `np.isfinite(oracle_deltas).all()` guard in the
   aggregator; the stub-the-helper seam is the only path
   that can inject a NaN given that
   `PerCellLiftDelta.loss_gbm: float` is required.
7. `test_renderer_oracle_ci_cell_renders_mean_and_interval`:
   construct a rollup row with finite oracle CI values
   AND `n_oracle_cells_paired == n_pair_grid` (no oracle
   asterisk); assert the rendered CI cell matches the
   `0.NNNN [0.NNNN, 0.NNNN]` shape (no trailing asterisk)
   AND the header includes `Δloss(oracle) [95% CI]`.
8. `test_renderer_oracle_ci_cell_partial_asterisk_fires_independently`:
   construct a row with `n_pair_grid=4,
   n_cells_paired=4` (main asterisk does NOT fire),
   `n_oracle_cells_paired=3` (oracle asterisk SHOULD
   fire). Assert the main Δloss CI cell has no
   asterisk AND the oracle Δloss CI cell has a
   trailing asterisk.
9. `test_renderer_oracle_ci_cell_renders_no_ci_when_n_oracle_cells_paired_is_zero`
   (qa-R1-C2 closure: covers the renderer's distinct
   `elif rollup_row.n_oracle_cells_paired == 0` branch):
   construct a NON-sentinel rollup row
   (`bootstrap_skipped_reason=None`) with
   `n_oracle_cells_paired=0` AND
   `oracle_metric_mean=None`. Assert the rendered oracle
   CI cell is `"(no CI)"` AND the main Δloss CI cell
   still renders the finite interval (the two columns
   are independent).
10. `test_renderer_oracle_ci_cell_renders_no_ci_when_sentinel_row`
    (qa-R1-I3 closure: covers the renderer's
    `bootstrap_skipped_reason is not None` short-circuit
    on the oracle column specifically): construct a row
    with `bootstrap_skipped_reason="no_gbm_predictions"`,
    `n_oracle_cells_paired=0`, `oracle_metric_mean=None`,
    `primary_metric_mean=None`. Assert (qa-R2-I2 closure:
    cross-column simultaneous guarantee) BOTH the
    rendered oracle CI cell is `"(no CI)"` AND the main
    Δloss CI cell is `"(no CI)"`. A renderer bug that
    routes sentinel rows through the oracle-zero branch
    (test #9's branch) instead of the sentinel branch
    would produce the same oracle cell output but would
    leave the main CI cell rendering a numeric interval
    (the main bootstrap would have written
    primary_metric_mean on a non-sentinel row); the
    cross-column assertion catches this.
11. `test_renderer_oracle_partial_asterisk_suppressed_when_fully_paired`
    (qa-R1-I2 closure: the qa-R1-I1-from-B19 precedent
    applied to the oracle asterisk): construct a row
    with `n_pair_grid=4, n_oracle_cells_paired=4` (equal,
    full coverage). Assert the rendered oracle CI cell
    contains the numeric interval AND no trailing
    asterisk.
12. `test_aggregator_emits_sentinel_row_with_default_oracle_fields`
    (qa-R1-I4 closure: sentinel-emit path functional
    test): fixture with disjoint seed sets so the
    aggregator routes through `_emit_sentinel_row`.
    Assert the emitted row has `n_oracle_cells_paired ==
    0` AND `oracle_metric_mean is None` AND
    `oracle_metric_ci_lo is None` AND
    `oracle_metric_ci_hi is None`.
13. `test_aggregator_oracle_and_main_ci_differ_under_xor_seed_offset`
    (qa-R2-C1 closure: pins the
    `BOOTSTRAP_ORACLE_SEED_OFFSET` derived-seed mechanism;
    the Risk-2 mitigation column originally claimed test
    #1 carried this assertion, but R2 caught the drift):
    construct a fixture with AT LEAST 8 cells where
    `delta_loss == loss_gbm - oracle_loss` for every cell
    (so the main bootstrap and oracle bootstrap operate
    on IDENTICAL per-cell values) AND set
    `bootstrap_n_resamples=500` on the ExperimentSpec
    (qa-R3-N2 closure: enough resamples + cells that two
    independent PCG64 streams produce numerically
    distinct CI bounds with probability ~1; smaller
    sizes risk flaky false-negatives on the `!=`
    assertion). Run the aggregator twice on the same
    fixture: once with the real
    `BOOTSTRAP_ORACLE_SEED_OFFSET` and once with
    `BOOTSTRAP_ORACLE_SEED_OFFSET` monkeypatched to 0.
    Under the real offset, assert
    `(oracle_metric_ci_lo, oracle_metric_ci_hi) !=
    (primary_metric_ci_lo, primary_metric_ci_hi)` (the
    two PCG64 streams produced different resample
    sequences). Under offset=0, assert the two CIs are
    equal (collapsed seed correlates them). The test
    kills a regression where the XOR is dropped: the
    `!=` assertion fails because the two CIs collapse to
    identical bounds.

Expected test delta after the build:
- Existing tests: 871 to 871 (no count change; fixtures
  updated in place).
- B20-new: 13 tests (this section), plus 2 added in the
  Build R1 swarm closure (qa-R1-build-I1 + qa-R1-build-I2;
  see the closure block below).
- Total: 871 + 15 = 886 actual post-build.

## B20.4 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B20-Risk-1 | Adding 4 required fields breaks all existing fixture sites. | Medium | The 5 fixture sites are enumerated in B20.3. Each gets documentary defaults so the existing tests pass byte-equivalent. The new `n_oracle_cells_paired` field is `Field(ge=0)` with no default (matches `n_pair_grid` pattern from B19); fixtures must supply it. The three `oracle_metric_*` fields default to None on the schema. |
| R-B20-Risk-2 | The two bootstrap invocations originally shared `BOOTSTRAP_DEFAULT_SEED`; arch-R1-C1 surfaced that `benchmarks/metrics/bootstrap.py:138` constructs `np.random.Generator(np.random.PCG64(seed))` where the stream depends ONLY on `seed`. Two invocations with identical seed AND identical `n_entities` produce IDENTICAL resampled indices, so the main Δloss and oracle Δ CIs would correlate on shared cells. | High (closed) | R-B20-2a mandates a derived oracle seed via `BOOTSTRAP_DEFAULT_SEED ^ BOOTSTRAP_ORACLE_SEED_OFFSET` (a new module-level XOR mask). The two PCG64 streams are now independent. The aggregator-side test (B20.3 test #13, added in qa-R2-C1 closure) re-runs the same fixture under monkeypatched `BOOTSTRAP_ORACLE_SEED_OFFSET=0` and asserts the oracle CI bounds DIFFER from the seed-shared baseline, killing a regression where the offset is dropped. |
| R-B20-Risk-3 | The oracle bootstrap uses a subset of cells (only those with finite `oracle_loss`). The CI is on a different statistic than the main Δloss CI. A reader could conflate the two. | Low | The renderer surfaces the two CIs in two distinct columns with distinct headers (`Δloss [95% CI]` and `Δloss(oracle) [95% CI]`). The B16 module docstring already establishes the oracle = "per-sample best" ceiling; B20 only adds a CI on top of that established quantity. |
| R-B20-Risk-4 | A future B11 driver change that removes `oracle_loss` from `PerCellLiftDelta` would silently break the B20 aggregator. | Low | The aggregator's `cell.oracle_loss` attribute access fails at pyright time if the field is renamed/removed; the dropped-cells branch fires if oracle_loss is None on every cell. A B11 refactor would need to update the B20 aggregator in lockstep. |
| R-B20-Risk-5 | The 4 new fields are schema-additive on `EnsembleLiftRollupRow` only. Pre-B20 parquet shards fail validation on load. | Low | Bench-run shards are short-lived (B17 R-B17-3 precedent). A user with a stale shard re-runs the aggregator. No backward-compat reader. |

## B20.5 Implementation outline

1. **Constant**: add `BOOTSTRAP_ORACLE_SEED_OFFSET:
   int = 0xB20_07A_C7E` to
   `benchmarks/report/_bootstrap_aggregate.py` next to the
   existing `BOOTSTRAP_DEFAULT_SEED` constant. Export via
   `__all__`.
2. **Schema**: add the 4 new fields to
   `EnsembleLiftRollupRow` in
   `benchmarks/bootstrap_manifest.py` per the grouped
   placement at R-B20-1 (`n_oracle_cells_paired` after
   `n_pair_grid`; `oracle_metric_*` triple after the
   `primary_metric_*` triple).
3. **Aggregator**: in
   `benchmarks/report/bootstrap_ensemble_lift.py`'s
   `_build_dataset_rollup`, after the existing main-Δ
   bootstrap, compute the per-cell oracle Δ array and run
   the second bootstrap using
   `seed=BOOTSTRAP_DEFAULT_SEED ^ BOOTSTRAP_ORACLE_SEED_OFFSET`.
   The `_emit_sentinel_row` helper hardcodes the 4 oracle
   fields inline (Build R1 arch-I2 closure: the helper was
   originally specified to take 4 defaulted keyword params
   but no caller ever overrode them and the renderer
   discards any non-default value via the sentinel short-
   circuit; widening the signature added no production
   value).
4. **Renderer**: in
   `benchmarks/report/ensemble_lift.py`'s
   `_render_complete_table_with_ci`, replace
   `"Δloss(oracle)"` header with
   `"Δloss(oracle) [95% CI]"` and replace the bare
   `oracle_delta_loss_mean` cell with the oracle CI
   format. Update the module docstring at the file head
   to mention the new column. Std variant untouched.
5. **Update existing fixtures**: the 6 sites enumerated in
   B20.3 get concrete defaults for the 4 new fields per
   the qa-R1-N2 closure values.
6. **NEW tests**: add
   `tests/benchmarks/test_b20_oracle_delta_ci.py` with the
   13 tests (Build R1 qa-I1 + qa-I2 closures added test #14
   for parquet round-trip and test #15 for the renderer's
   `rollup_row is None` outer-branch arm; 15 tests total).
7. **Verify**: ruff + pyright clean; 886 tests pass.

## Addressed

R1 swarm: architecture-reviewer (1C / 5I / 3N
REQUEST_CHANGES), qa-test-coverage (2C / 4I / 2N
REQUEST_CHANGES), style-reviewer (0C / 0I / 1N APPROVE
accepted house style). Deduplicated total: 3 CRITICAL,
9 IMPROVEMENT, 5 NITPICK. Closures:

- **arch-R1-C1** (PCG64 stream depends ONLY on seed:
  `benchmarks/metrics/bootstrap.py:138` constructs
  `np.random.Generator(np.random.PCG64(seed))`. Two
  invocations with the same seed AND same `n_entities`
  produce IDENTICAL resampled indices, correlating the
  main Δloss and oracle Δ CIs on shared cells): added
  R-B20-2a mandating a derived seed via
  `BOOTSTRAP_DEFAULT_SEED ^ BOOTSTRAP_ORACLE_SEED_OFFSET`
  (new constant in `_bootstrap_aggregate.py`). The two
  PCG64 streams are now independent. R-B20-Risk-2
  rewritten with the actual mitigation.
- **qa-R1-C1** (test #6 inject seam unnamed): test #6
  renamed `test_aggregator_oracle_nan_delta_raises_via_stub`
  with the explicit stub-`compute_per_cell_lift_deltas`
  pattern naming the seam. The stub returns a
  `PerCellLiftDelta` record with `loss_gbm=float('nan')`,
  the only injection path that actually exercises the new
  `np.isfinite(oracle_deltas).all()` guard.
- **qa-R1-C2** (renderer `n_oracle_cells_paired == 0`
  branch untested): added test #9
  `test_renderer_oracle_ci_cell_renders_no_ci_when_n_oracle_cells_paired_is_zero`.
- **arch-R1-I1** (field placement): updated R-B20-1 to
  place `n_oracle_cells_paired` in the COUNTS group (after
  `n_pair_grid`) and `oracle_metric_*` triple in the
  METRIC STATS group (after `primary_metric_*`), matching
  the existing schema group structure. B20.0 example
  rewritten with the new grouped placement.
- **arch-R1-I2** (oracle OOM gate is dead in production
  because the main gate fires first on the equal-or-larger
  main array): added an inline comment at the gate
  acknowledging it is defensive; test #5 forces the gate
  to fire by stubbing the main delta path so the main
  gate is suppressed.
- **arch-R1-I3** (name drift between new `oracle_metric_*`
  fields and the existing `oracle_delta_loss_mean` scalar
  on `PerDatasetLift`): added a naming-clarity paragraph
  in B20.2 documenting the parallel with `primary_metric_*`
  and noting the pre-B16 scalar field stays for the
  std-variant renderer.
- **arch-R1-I4** (missing module-docstring update):
  B20.2 now mandates an update to
  `benchmarks/report/ensemble_lift.py`'s module docstring
  at the file head.
- **arch-R1-I5** (cross-CI clarifier footnote): B20.2 now
  documents the sixth implicit footnote source for the
  oracle CI asterisk and defers the explicit footnote
  table block under D-B20.1.
- **qa-R1-I2** (counter-test for oracle asterisk
  suppressed): added test #11
  `test_renderer_oracle_partial_asterisk_suppressed_when_fully_paired`
  (mirrors the B19 qa-R1-I1 precedent).
- **qa-R1-I3** (renderer sentinel-row oracle test): added
  test #10
  `test_renderer_oracle_ci_cell_renders_no_ci_when_sentinel_row`.
- **qa-R1-I4** (sentinel emit functional test): added test
  #12 `test_aggregator_emits_sentinel_row_with_default_oracle_fields`.
- **qa-R1-N1** (sign-convention test #4 vulnerable to
  cross-wire if `delta_loss == 0.20` coincidence): test #4
  fixture now explicitly sets `delta_loss=0.05` so the
  main bootstrap mean (0.05) differs from the oracle
  bootstrap mean (0.20); the test asserts both the
  `oracle_metric_mean == 0.20` AND the
  `oracle_metric_mean != 0.05` cross-wire pin AND
  `primary_metric_mean == 0.05`.
- **qa-R1-N2** (concrete fixture defaults): B20.3 now
  enumerates explicit oracle field values for each of the
  6 fixture sites.
- **style-R1-N1** (Title Case heading flagged as
  established house style): NOT changed. Same pattern as
  every prior benchmark design doc.

Test count after R1 closures: 12 new tests (was 8);
total 871 + 12 = 883. (R2 closures later added test #13
bringing the total to 13 / 884; see R2 closure block
below.)

### R2 swarm closure

R2 confirming swarm: architecture-reviewer (0C / 0I / 1N
APPROVE), qa-test-coverage (1C / 2I / 2N
REQUEST_CHANGES), style-reviewer (0C / 0I / 0N APPROVE).
Total: 1 CRITICAL, 2 IMPROVEMENT, 3 NITPICK. Closures:

- **qa-R2-C1** (the R-B20-Risk-2 mitigation column
  claimed test #1 carries a seed-independence assertion,
  but test #1's body asserts only shape and ordering;
  the seed-independence pin was unspecified in the
  12-test ledger): added test #13
  `test_aggregator_oracle_and_main_ci_differ_under_xor_seed_offset`
  explicitly running the aggregator twice (once with the
  real XOR offset, once with offset=0) and asserting CI
  bound inequality under the real offset + equality
  under offset=0. The test count is now 13 (was 12);
  total `871 + 13 = 884`.
- **arch-R2-N1** (stale "(NEW; 8 tests)" parenthetical
  at line 401): updated to "(NEW; 13 tests)".
- **qa-R2-I1** (test #6 error-message assertion was
  prose-only "with message containing"): now mandates
  `pytest.raises(RawRollupError, match=r"non-finite
  oracle delta")` explicitly so a guard rename is
  caught.
- **qa-R2-I2** (test #10 sentinel-row oracle test
  asserts only the oracle column, not the main column
  simultaneously; a renderer routing bug that produces
  the same oracle output via the n_oracle_cells_paired=0
  branch instead of the sentinel branch would not be
  caught): test #10 now asserts BOTH the oracle CI cell
  AND the main Δloss CI cell render as "(no CI)" under
  the same sentinel row.
- **qa-R2-N1** (factory None-fallback location
  ambiguous): B20.3 item 4 now explicitly states the
  fallback lives in the factory body, NOT on the schema
  field; pydantic schema stays `int = Field(ge=0)`.
- **qa-R2-N2** (test #5 doesn't specify n_resamples for
  the OOM gate): test #5 now mandates
  `bootstrap_n_resamples=2` so the gate condition
  `n_oracle_cells * 2 > 1` fires deterministically.
- **style-R2-N1** (Title Case headings accepted as house
  style, unchanged from R1): NOT changed.

Test count after R2 closures: 13 new tests; total
`871 + 13 = 884`.

### R3 swarm closure

R3 confirming swarm: architecture-reviewer (0C / 0I / 0N
APPROVE), qa-test-coverage (0C / 0I / 2N APPROVE),
style-reviewer (0C / 0I / 1N APPROVE accepted house
style). Total: 0 CRITICAL, 0 IMPROVEMENT, 3 NITPICK.
Closures:

- **qa-R3-N1** (stale "test #1" pointer in R-B20-Risk-2
  mitigation column after qa-R2-C1 moved the
  seed-independence pin to test #13): updated to "test
  #13, added in qa-R2-C1 closure" so a future reviewer
  reading the Risks table is sent to the right test.
- **qa-R3-N2** (test #13 spec did not protect against
  flaky false-negatives on minimal fixtures: with
  small cell counts and small `n_resamples`, two
  different PCG64 streams could coincidentally produce
  identical CI bounds): mandated AT LEAST 8 cells AND
  `bootstrap_n_resamples=500` so the `!=` assertion is
  reliable.
- **style-R3-N1** (Title Case headings accepted as house
  style; carried unchanged from R1/R2): NOT changed.

### Build R1 swarm closure

Build R1 swarm on commit `b6c6496` (the initial build):
code-reviewer (0C / 1I / 2N APPROVE), architecture-reviewer
(0C / 3I / 2N APPROVE), qa-test-coverage (0C / 2I / 1N
APPROVE), style-reviewer (0C / 0I / 1N APPROVE accepted
house style). Deduplicated total: 0 CRITICAL, 6 IMPROVEMENT,
6 NITPICK. Closures:

- **arch-R1-build-I1** (asymmetric monkeypatch seam between
  `BOOTSTRAP_ROW_COUNT_CEILING` by-name import and
  `BOOTSTRAP_ORACLE_SEED_OFFSET` late-bound via
  `_bootstrap_aggregate.<NAME>`; tests that monkeypatch the
  ceiling must target the consumer module while tests that
  monkeypatch the offset target the source module, which is
  a future-bug shape): converted the ceiling reads at
  `bootstrap_ensemble_lift.py:250` and `:299` to late-binding
  `_bootstrap_aggregate.BOOTSTRAP_ROW_COUNT_CEILING`. Both
  constants now share the same canonical-source monkeypatch
  seam; test #5 updated to monkeypatch
  `_bootstrap_aggregate.BOOTSTRAP_ROW_COUNT_CEILING` (not the
  consumer module).
- **arch-R1-build-I2** (`_emit_sentinel_row` widened its
  keyword surface with 4 oracle params that have only the
  default values exercised by any caller; production never
  passes non-default oracle values to a sentinel row and the
  renderer's `bootstrap_skipped_reason is not None`
  short-circuit would discard them anyway): dropped the 4
  new oracle keyword params; the sentinel emit now hardcodes
  `n_oracle_cells_paired=0` and `oracle_metric_*=None`
  inline. Test #12 still pins the contract because it
  exercises the aggregator end-to-end through
  `aggregate_bootstrap_ensemble_lift_rollup` (not the
  helper directly).
- **arch-R1-build-I3** (oracle delta materialised twice as a
  Python list + a numpy array, where the main bootstrap path
  materialises once): collapsed the oracle delta computation
  to a single `np.array([...], dtype=float)` matching the
  main path's idiom at `:241`. `n_oracle_cells_paired` is
  read from `oracle_deltas.shape[0]`.
- **code-R1-build-I1** (test #6 docstring claimed the test
  stubs `entity_block_bootstrap_ci` to a no-op when the body
  has no such stub; the test works because both injected
  cells have `delta_loss=0.20` finite so the real main
  bootstrap passes its `isfinite` guard, but a future
  reader following the docstring would be misled): rewrote
  the docstring to describe the actual mechanism (main
  guard sees finite `delta_loss` array and passes; oracle
  guard sees a NaN in the per-cell oracle delta array
  derived from the NaN `loss_gbm`).
- **qa-R1-build-I1** (parquet round-trip for the 4 new
  oracle fields never exercised end-to-end through
  `write_ensemble_lift_rollup` + `load_ensemble_lift_rollup`;
  a silent schema change or pd.NA -> None coercion break on
  the new nullable float fields would not be caught): added
  test #14
  `test_ensemble_lift_rollup_oracle_fields_survive_parquet_round_trip`
  covering both a finite-CI row and an all-None oracle row.
- **qa-R1-build-I2** (renderer `rollup_row is None` outer-
  branch arm never tested in isolation; test #10 covers only
  the `bootstrap_skipped_reason is not None` arm of the same
  outer branch, so a regression that moved the oracle `(no
  CI)` emission inside the sentinel sub-branch would be
  masked): added test #15
  `test_renderer_oracle_ci_cell_renders_no_ci_when_rollup_row_absent_from_index`
  asserting BOTH the main Δloss CI cell AND the oracle CI
  cell render as `(no CI)` when the dataset is missing from
  the rollup index.
- **code-R1-build-N1** (missing return annotation on
  `_make_rollup_row` helper in test_b20): added `->
  EnsembleLiftRollupRow` and hoisted the import to the top
  of the file.
- **code-R1-build-N2** (`and rollup_row.n_pair_grid > 0`
  clause in the oracle partial-flag computation is dead
  inside the `else` arm because the outer
  `n_oracle_cells_paired == 0` short-circuit guarantees
  `n_oracle_cells_paired >= 1`, which implies `n_pair_grid
  >= 1` by R-B20-6): clause kept for shape parallelism with
  the main `partial` calculation at line 166, with an inline
  comment documenting why it's defensive even though
  currently unreachable-False.
- **arch-R1-build-N1** (test #5's "AFTER the main bootstrap
  returns" comment overstates temporal sequencing; the
  actual mechanism is that the gate re-reads
  `BOOTSTRAP_ROW_COUNT_CEILING` on each invocation):
  rephrased the comment to describe the re-read mechanism
  explicitly.
- **arch-R1-build-N2** (oracle vs main raise messages
  distinguish via the "oracle delta" suffix tag which is
  load-bearing for test #5/#6's `match=`): NOT changed. The
  tag is consistent across both oracle raise sites
  (isfinite + OOM) and the design's qa-R2-I1 closure
  already mandates the explicit `match=` clause; deferred
  under D-B20.2 below.
- **qa-R1-build-N1** (schema does not validate the cross-
  field invariant `n_oracle_cells_paired <= n_cells_paired`;
  the aggregator enforces it structurally but no schema
  backstop exists): NOT changed. The aggregator's
  list-comprehension derives `oracle_deltas` from a subset
  of the same `computed.cells` that drives `n_cells_paired`,
  so the invariant is structural; adding a model validator
  would be redundant with the aggregator-side guarantee.
  Deferred under D-B20.3 below.
- **style-R1-build-N1** (`# Note:` substantive comment in
  test #6 fixture explaining direct construction of the
  NaN-injected `PerCellLiftDelta`): NOT changed; accepted
  as house style. The comment carries real explanatory
  weight and is not an AI-flavored placeholder.

Test count after R1 build closures: 15 tests (was 13;
qa-R1-build-I1 added test #14, qa-R1-build-I2 added test
#15); total `871 + 15 = 886`.

### Build R2 swarm closure

R2 confirming swarm on commit `4b44322` (post Build R1 fixes):
code-reviewer (0C / 0I / 1N APPROVE), architecture-reviewer
(0C / 0I / 2N APPROVE), qa-test-coverage (0C / 1I / 2N
APPROVE), style-reviewer (0C / 0I / 1N APPROVE accepted
house style). Deduplicated total: 0 CRITICAL, 1 IMPROVEMENT
(re-raised carry-forward), 5 NITPICK. Closures:

- **code-R2-build-N1** (test_b20_oracle_delta_ci.py module
  docstring still reads "Covers the 13 named tests" after
  R1 added tests #14 + #15): updated to "15 tests" with an
  explicit note that #14 + #15 are R1 build closures.
- **arch-R2-build-N1** (B20.1 pseudocode block at the
  design body still showed the two-pass list +
  np.array materialisation pattern and the bare
  `BOOTSTRAP_ROW_COUNT_CEILING` reference, both fixed by
  R1 arch-I3 and arch-I1): rewrote the B20.1 pseudocode to
  match the post-R1 single-pass `np.array(..., dtype=
  float)` shape and the `_bootstrap_aggregate.<NAME>` late-
  binding references.
- **arch-R2-build-N2** (B20.1 prose at "_emit_sentinel_row
  helper gets 4 new defaulted parameters" still described
  the pre-R1 widened signature; B20.5 step 3 + step 7 still
  pointed at 884 tests + 13 NEW tests): rewrote to describe
  the post-R1 hardcoded inline implementation and updated
  step 7 to "886 tests pass" + step 6 to "15 tests total".
- **qa-R2-build-I1** (re-raised the code-R1-build-N2
  observation about the `and rollup_row.n_pair_grid > 0`
  defensive clause being unreachable-False under the
  `else` arm; the qa agent reframed it as a coverage
  observation): the underlying clause is structurally
  unreachable-False BY THEOREM
  (`n_oracle_cells_paired >= 1` implies `n_pair_grid >= 1`
  by R-B20-6), so the branch cannot be exercised by any
  test fixture. The R1 code-R1-build-N2 closure documented
  this in-place with a clarifying inline comment at
  `benchmarks/report/ensemble_lift.py:182-188`; the qa
  agent itself acknowledged "presents no silent-failure
  risk in its current form (it evaluates to False
  regardless)." No further action; the closure stands as
  the durable artifact.
- **qa-R2-build-N2** (test #15's `table_row.count("(no
  CI)") == 2` assertion is brittle to a future column
  injection between the main Δloss CI cell and the oracle
  CI cell): NOT changed. The qa agent itself flagged this
  as below the action threshold: the negative assertions
  (`"0.2000 [0.1500, 0.2500]" not in table_row` AND
  `"0.1000 [0.0800, 0.1200]" not in table_row`) provide
  independent discriminators on each column's content. A
  future column addition would require a companion test
  update as a matter of course; the count assertion
  remains the simpler and more readable form for the v1
  9-column layout.
- **style-R2-build-N1** (pre-accepted house-style `#
  Note:` comment): unchanged.

Test count after R2 closures: 15 tests; total `871 + 15 =
886`.

## Deferred

- **D-B20.1**: explicit footnote table block for the
  oracle CI partial-coverage asterisk (coordinated with
  the existing partial-coverage footnote infrastructure).
  V1 surfaces this as the trailing asterisk only.
- **D-B20.2** (arch-R1-build-N2): tighten the oracle vs
  main raise-message discriminator to a stable token
  (e.g., the literal field name `oracle_metric_*` or
  `n_oracle_cells_paired`) instead of the prose "oracle
  delta" suffix. Carries to a future style/contract sweep
  on the bootstrap aggregators.
- **D-B20.3** (qa-R1-build-N1): pydantic-level model
  validator enforcing `n_oracle_cells_paired <=
  n_cells_paired` on `EnsembleLiftRollupRow`. Redundant
  with the aggregator's structural guarantee at v1; would
  add defense-in-depth for a future refactor that drops the
  list-comprehension shape.
