# B22 design delta: per-fold CIs (D-B16.3 / D-B13.6)

**Scope**: D-B16.3 (inherited from D-B13.6) adds per-fold
bootstrap CIs to the 5 rollup aggregators. v1 ships pooled
CIs aggregated across folds + seeds, producing one CI per
(dataset, model). B22 adds an optional opt-in path producing
ALSO one CI per (dataset, model, fold_index) so a reader can
inspect fold-level uncertainty.

Per-fold CIs are gated on a new
`ExperimentSpec.bootstrap_per_fold_cis_enabled: bool = False`
toggle. When False (the default) the aggregators behave
exactly as today (pooled CI only); when True, each rollup
row additionally carries a `per_fold_cis: list[FoldCI] | None`
attribute populated with one `FoldCI` record per fold. The
pooled CI is still computed and remains the primary surface
on the row; per-fold CIs are an audit-only addition at v1.

The renderer is NOT modified by B22 (deferred under D-B22.1)
to keep the diff scoped to schema + aggregator. The parquet
shard carries the new field; downstream consumers (notebooks,
ad-hoc analyses) can read per-fold CIs without any renderer
change.

## Requirements

The grading rubric for every reviewer finding traces back to
one of these.

- **R-B22-1** Add a new `FoldCI` pydantic `BaseModel` in
  `benchmarks/bootstrap_manifest.py` carrying per-fold audit
  fields. `FoldCI` is exported via the module's `__all__`
  (arch-R1-I1 closure: downstream consumers must import the
  class to round-trip rows). The fields:
  ```python
  class FoldCI(BaseModel):
      model_config = ConfigDict(extra="forbid", frozen=True)
      fold_index: int = Field(ge=0)
      # arch-R1-C1 closure: n_seeds is the count of UNIQUE
      # `seed` values contributing to this fold. n_entities
      # is the count of UNIQUE entity_ids (typically B5's
      # entity_col + time_col composite); the two fields
      # answer different questions and must be tracked
      # independently. For aggregators where each cell is its
      # own entity (B6 / B7 / B8 / B16's cell-as-entity
      # contract from B14.0), n_seeds tracks the seed axis
      # and n_entities equals n_cells in the fold.
      n_seeds: int = Field(ge=0)
      n_entities: int = Field(ge=0)
      metric_mean: float | None = None
      metric_ci_lo: float | None = None
      metric_ci_hi: float | None = None
      ci_method: str
      ci_fallback_reason: str | None = None
  ```
  `frozen=True` + `extra="forbid"` matches every existing
  RollupRow schema. `metric_mean` / `metric_ci_lo` /
  `metric_ci_hi` are nullable so a fold with insufficient
  data (e.g., single-seed degenerate primitive collapse) can
  surface as `(None, None, None)` rather than raising. The
  `ci_method` + `ci_fallback_reason` fields parallel the
  parent row's audit fields and let the per-fold path
  surface BCa fallbacks independently from the pooled path.
- **R-B22-2** Add a `per_fold_cis: list[FoldCI] | None = None`
  field to all 5 RollupRow schemas (B5 `RollupRow`, B6
  `PairwiseRollupRow`, B7 `TrainingTimeRollupRow`, B8
  `HPOUpliftRollupRow`, B16 `EnsembleLiftRollupRow`). The
  schema default `None` is the backward-compat marker for
  pre-B22 parquet shards AND for aggregator runs where
  per-fold CIs are not enabled. Aggregators that ARE
  computing per-fold CIs write a non-None list (which may be
  empty if no fold has computable data).
- **R-B22-3** Add `bootstrap_per_fold_cis_enabled: bool =
  False` to `ExperimentSpec` in `benchmarks/config.py`.
  Default False preserves v1 behavior on every existing
  config. The 5 aggregators each consult their respective
  `ExperimentSpec` kind to decide whether to populate
  `per_fold_cis`.
- **R-B22-4** Per-fold CI computation contract: when
  `bootstrap_per_fold_cis_enabled=True`, the aggregator
  groups the pooled `(losses, entity_ids)` arrays by
  `fold_index` and calls `entity_block_bootstrap_ci`
  ONCE PER FOLD with the same `n_resamples`, `confidence`,
  `seed=BOOTSTRAP_DEFAULT_SEED ^ BOOTSTRAP_PER_FOLD_SEED_OFFSET
  ^ fold_index`, `metric_fn` (per-task), and
  `ci_method=_bootstrap_aggregate.BOOTSTRAP_DEFAULT_CI_METHOD`.
  The fold-specific seed XOR ensures independent PCG64
  streams per fold (R-B20-2a precedent). The result tuple
  populates one `FoldCI` record.
- **R-B22-5** Add a new module constant
  `BOOTSTRAP_PER_FOLD_SEED_OFFSET: int = 0xB22_F01D_C1`
  in `benchmarks/report/_bootstrap_aggregate.py`. Exported
  via `__all__`. Aggregators read it via late-binding lookup
  (B20 R1 arch-I1 precedent).
- **R-B22-6** Per-fold OOM gate: each per-fold bootstrap
  is subject to the same `BOOTSTRAP_ROW_COUNT_CEILING` guard
  as the pooled bootstrap. A fold whose
  `n_fold_rows * n_resamples > _bootstrap_aggregate.
  BOOTSTRAP_ROW_COUNT_CEILING` causes a `RawRollupError`
  naming the fold:
  ```
  RawRollupError(
      f"aggregate_bootstrap_{kind}_rollup: dataset={dataset!r} "
      f"fold_index={fold_index} with n_rows={n} * n_resamples="
      f"{r} exceeds the bootstrap-row-count ceiling "
      f"({ceiling}) on the per-fold CI"
  )
  ```
- **R-B22-7** Degenerate per-fold handling: a fold with
  exactly one unique entity yields the existing primitive's
  degenerate path `(mean, mean, mean, None)`; the `FoldCI`
  record carries those collapsed bounds. A fold with ZERO
  rows (no cells contributed) is omitted from the
  `per_fold_cis` list entirely (NOT a None entry).
- **R-B22-8** Sentinel rows pass `per_fold_cis=None`
  (NOT an empty list). The non-None list semantic is "this
  row HAS computed per-fold CIs (possibly empty if every
  fold was degenerate-zero-rows)"; None means "per-fold CIs
  were not requested or this row is a sentinel".
- **R-B22-9** The pooled CI computation (the existing
  primary_metric_* fields) is unchanged. B22 ADDS per-fold
  CIs on top; it does NOT alter the pooled-CI semantics.
  Existing tests pass byte-equivalent without any change to
  the pooled bound values.

## B22.0 What the change actually adds

Pre-B22 RollupRow:

```python
class RollupRow(BaseModel):
    ...
    primary_metric_mean: float | None = None
    primary_metric_ci_lo: float | None = None
    primary_metric_ci_hi: float | None = None
    bootstrap_seed: int
    bootstrap_n_resamples: int = Field(ge=0)
    bootstrap_confidence: float = 0.95
    bootstrap_rng_algorithm: str = "PCG64"
    bootstrap_ci_method: str = "percentile"
    bootstrap_ci_fallback_reason: str | None = None
    bootstrap_numpy_version: str
    ...
```

Post-B22:

```python
class RollupRow(BaseModel):
    ...
    primary_metric_mean: float | None = None
    primary_metric_ci_lo: float | None = None
    primary_metric_ci_hi: float | None = None
    bootstrap_seed: int
    bootstrap_n_resamples: int = Field(ge=0)
    bootstrap_confidence: float = 0.95
    bootstrap_rng_algorithm: str = "PCG64"
    bootstrap_ci_method: str = "percentile"
    bootstrap_ci_fallback_reason: str | None = None
    # B22 / D-B16.3: per-fold CIs (audit-only at v1). None when
    # `bootstrap_per_fold_cis_enabled=False` on the spec OR when
    # the row is a sentinel. A non-None list carries one FoldCI
    # record per fold with computable data; folds with zero
    # contributing rows are omitted (NOT None entries).
    per_fold_cis: list[FoldCI] | None = None
    bootstrap_numpy_version: str
    ...
```

The new `FoldCI` BaseModel is module-private in shape but
exported as part of the public schema surface (downstream
consumers must construct it to round-trip rows). Its frozen
+ extra="forbid" config matches every existing schema in
`bootstrap_manifest.py`.

## B22.1 Aggregator changes

The B5 raw-loss aggregator at `_build_group_rollup`
(`bootstrap_rollup.py:227`) computes pooled
`(losses, entities)` from the cells loop. B22 EXTENDS the
cells loop to also track per-row `fold_index` + per-row
`seed` (arch-R1-C2 closure: the existing loop tracks only
`losses_blocks` + `entity_blocks` at lines 273-304; B22
adds two parallel blocks):

```python
losses_blocks: list[np.ndarray] = []
entity_blocks: list[np.ndarray] = []
fold_blocks: list[np.ndarray] = []  # NEW B22: per-row fold_index
seed_blocks: list[np.ndarray] = []  # NEW B22: per-row seed (for n_seeds in FoldCI)
for _, row in ok_cells.iterrows():
    ...
    per_row_losses, cell_metric_fn = result
    panel_row_index = predictions["panel_row_index"].to_numpy()
    entity_ids = _resolve_entity_ids(...)
    losses_blocks.append(per_row_losses)
    entity_blocks.append(entity_ids)
    # B22: broadcast the cell's fold_index + seed across its rows
    fold_blocks.append(
        np.full(per_row_losses.shape[0], int(row["fold_index"]), dtype=np.int64)
    )
    seed_blocks.append(
        np.full(per_row_losses.shape[0], int(row["seed"]), dtype=np.int64)
    )

...
losses = np.concatenate(losses_blocks)
entities = np.concatenate(entity_blocks)
fold_indices = np.concatenate(fold_blocks)  # NEW B22
seeds = np.concatenate(seed_blocks)  # NEW B22
```

After the existing pooled bootstrap call, the per-fold block
fires conditionally:

```python
# B22 / D-B16.3: per-fold CIs (opt-in)
per_fold_cis: list[FoldCI] | None = None
if _spec_has_per_fold_cis_enabled(experiments, kind="raw_loss"):
    per_fold_cis = _compute_per_fold_cis(
        losses=losses,
        entities=entities,
        fold_indices=fold_indices,
        seeds=seeds,
        n_resamples=n_resamples,
        confidence=BOOTSTRAP_CONFIDENCE,
        base_seed=BOOTSTRAP_DEFAULT_SEED,
        fold_seed_offset=_bootstrap_aggregate.BOOTSTRAP_PER_FOLD_SEED_OFFSET,
        ci_method=_bootstrap_aggregate.BOOTSTRAP_DEFAULT_CI_METHOD,
        metric_fn=metric_fn,
        dataset_name=dataset_name,
        kind="raw_loss",
    )
```

The 4 other aggregators (B6 / B7 / B8 / B16) follow the same
pattern but with their own per-row sources for fold_index +
seed:
- **B6 pairwise** + **B7 training-time** + **B8 HPO-uplift**:
  cell-as-entity (B14.0 contract), so `entities = np.arange(n_cells)`
  and `fold_indices = ok["fold_index"].to_numpy().astype(np.int64)`
  + `seeds = ok["seed"].to_numpy().astype(np.int64)`
  (where `ok` is the local name each aggregator binds to
  the filtered OK-cells frame; B5 binds `ok_cells`, the
  other three bind `ok`). No per-row broadcasting needed
  because each cell IS a single-row entity.
- **B16 ensemble_lift**: cell-as-entity via
  `computed.cells`; each `PerCellLiftDelta` carries
  `seed` + `fold_index` directly, so `fold_indices` and
  `seeds` are gathered via list comprehension.

`_compute_per_fold_cis` is a NEW shared helper in
`benchmarks/report/_bootstrap_aggregate.py`:

```python
def _compute_per_fold_cis(
    *,
    losses: np.ndarray,
    entities: np.ndarray,
    fold_indices: np.ndarray,
    seeds: np.ndarray,  # arch-R1-C1 closure: distinct from entities
    n_resamples: int,
    confidence: float,
    base_seed: int,
    fold_seed_offset: int,
    ci_method: Literal["percentile", "bca"],
    metric_fn: Callable[[np.ndarray], float],
    dataset_name: str,
    kind: str,
) -> list[FoldCI]:
    """Bootstrap CI per fold, returning one FoldCI per fold with
    rows in the input. Folds with zero rows are omitted. Result
    list is sorted ascending by fold_index (qa-R1-C1 closure).

    Raises:
        RawRollupError: a fold's n_rows * n_resamples exceeds
            BOOTSTRAP_ROW_COUNT_CEILING.
    """
    fold_cis: list[FoldCI] = []
    unique_folds = sorted({int(f) for f in fold_indices})
    for fold_index in unique_folds:
        mask = fold_indices == fold_index
        fold_losses = losses[mask]
        fold_entities = entities[mask]
        fold_seeds = seeds[mask]
        if fold_losses.size == 0:
            continue
        if fold_losses.shape[0] * n_resamples > _bootstrap_aggregate.BOOTSTRAP_ROW_COUNT_CEILING:
            raise RawRollupError(
                f"aggregate_bootstrap_{kind}_rollup: dataset="
                f"{dataset_name!r} fold_index={fold_index} with "
                f"n_rows={fold_losses.shape[0]} * n_resamples="
                f"{n_resamples} exceeds the bootstrap-row-count "
                f"ceiling "
                f"({_bootstrap_aggregate.BOOTSTRAP_ROW_COUNT_CEILING}) "
                "on the per-fold CI"
            )
        # arch-R1-C1 closure: n_seeds is the seed-axis count,
        # NOT the entity-axis count.
        n_seeds_in_fold = int(np.unique(fold_seeds).shape[0])
        n_unique_entities = int(np.unique(fold_entities).shape[0])
        seed = base_seed ^ fold_seed_offset ^ fold_index
        mean, ci_lo, ci_hi, fallback_reason = entity_block_bootstrap_ci(
            fold_losses,
            fold_entities,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
            metric_fn=metric_fn,
            ci_method=ci_method,
        )
        fold_cis.append(
            FoldCI(
                fold_index=fold_index,
                n_seeds=n_seeds_in_fold,
                n_entities=n_unique_entities,
                metric_mean=mean,
                metric_ci_lo=ci_lo,
                metric_ci_hi=ci_hi,
                ci_method=ci_method,
                ci_fallback_reason=fallback_reason,
            )
        )
    return fold_cis
```

The 4 other aggregators (B6, B7, B8, B16) follow the same
pattern: after the pooled bootstrap call, conditionally
call `_compute_per_fold_cis` and plumb the result into the
row constructor. The B16 ensemble_lift aggregator has TWO
bootstrap invocations (main Δloss + oracle Δ); per-fold CIs
are computed for the MAIN delta only at v1 (oracle per-fold
deferred under D-B22.2).

The `_spec_has_per_fold_cis_enabled` helper consults the
config's `ExperimentSpec` list for the given kind:

```python
def _spec_has_per_fold_cis_enabled(
    experiments: Iterable[ExperimentSpec], *, kind: str
) -> bool:
    return any(
        spec.kind == kind and spec.bootstrap_per_fold_cis_enabled
        for spec in experiments
    )
```

## B22.2 ExperimentSpec changes

```python
class ExperimentSpec(BaseModel):
    ...
    bootstrap_pairwise_enabled: bool = False
    bootstrap_hpo_uplift_enabled: bool = False
    bootstrap_ensemble_lift_enabled: bool = False
    # B22 / D-B16.3: per-fold CI opt-in (audit-only at v1).
    # When True, each emitted rollup row carries a non-None
    # `per_fold_cis` list with one FoldCI per fold. False
    # preserves v1 behavior (pooled CI only).
    bootstrap_per_fold_cis_enabled: bool = False
    ...
```

The flag is read by all 5 aggregators (raw_loss + pairwise +
training_time + hpo_uplift + ensemble_lift) but each aggregator
matches against its OWN kind (e.g., the B5 aggregator only
checks `kind="raw_loss"`).

## B22.3 New module constant

In `benchmarks/report/_bootstrap_aggregate.py`:

```python
# B22 / D-B16.3: XOR seed offset for per-fold bootstrap
# invocations. Combined with `fold_index` to produce
# `base_seed ^ BOOTSTRAP_PER_FOLD_SEED_OFFSET ^ fold_index`
# so each fold's PCG64 stream is independent from both the
# pooled stream AND every other fold's stream. The XOR
# distinctness invariant holds for any two non-equal
# non-negative fold_indices (`i XOR j != 0` for `i != j`);
# the pooled-vs-per-fold distinctness requires this
# constant to be non-zero (arch-R1-N1 closure: resetting
# this to 0 would silently collapse `seed(fold_index=0)`
# onto `BOOTSTRAP_DEFAULT_SEED` and re-correlate the
# pooled + fold-0 streams).
BOOTSTRAP_PER_FOLD_SEED_OFFSET: int = 0xB22_F01D_C1
```

Exported via `__all__`.

## B22.4 Renderer (NOT modified at v1)

The renderer surface is intentionally left untouched at v1.
The per-fold CIs are an audit-only field on the parquet shard;
downstream consumers (notebooks, ad-hoc analyses) can read
them via `load_*_rollup` and surface them however they like.

A renderer integration that surfaces per-fold CIs as an
expandable sub-table or as a "per-fold variance" footnote
is deferred under D-B22.1.

## B22.5 Dependency surface

No new external dependencies. `_compute_per_fold_cis` uses
the existing `entity_block_bootstrap_ci` primitive + numpy.

## B22.6 Test surface

### Existing tests touched

- **`tests/benchmarks/test_bootstrap_manifest.py`**: each
  RollupRow factory helper's field-by-field round-trip
  asserts get one new check for `per_fold_cis is None`
  (default). Add ONE new test per schema asserting the
  round-trip of a non-None `per_fold_cis` list with 2 fold
  entries; verify each `FoldCI` field survives.
- **`tests/benchmarks/test_b17_byte_identity_pins.py`**:
  inline non-exposure asserts (`assert "per_fold_cis" not
  in md`) on all 4 byte-pin renderer tests (parallel to
  B21 R1 qa-I3 closure for `bootstrap_ci_method`).
- **`tests/benchmarks/test_b21_bca_ci.py`**: test #16
  schema-default isolation extends to assert
  `bootstrap_*RollupRow().per_fold_cis is None` on every
  schema (one additional assertion per schema).
- The 5 aggregator integration test files: existing tests
  use `ExperimentSpec(...)` without the new field, which
  defaults to False, so they pass byte-equivalent
  (qa-R1-N closure: backward compat).

### NEW B22 tests

`tests/benchmarks/test_b22_per_fold_cis.py` (NEW; 14 tests):

1. `test_fold_ci_model_has_frozen_extra_forbid_config`:
   assert `FoldCI.model_config` has `frozen=True` AND
   `extra="forbid"` (parallels every existing RollupRow
   schema invariant).
2. `test_fold_ci_field_ge_constraints_reject_negative`:
   construct `FoldCI(fold_index=-1, ...)` and assert
   `ValidationError`; same for `n_seeds=-1`, `n_entities=-1`.
3. `test_rollup_row_per_fold_cis_schema_default_is_none`:
   construct each of the 5 RollupRow types WITHOUT
   supplying `per_fold_cis` and assert the field reads back
   `None`. Pins the backward-compat marker.
4. `test_rollup_row_per_fold_cis_accepts_empty_list`:
   construct each of the 5 RollupRow types with
   `per_fold_cis=[]` and assert the value round-trips. The
   empty-list state means "per-fold CIs were requested but
   every fold was degenerate-zero-rows".
5. `test_rollup_row_per_fold_cis_round_trip_with_two_folds`
   (qa-R1-N2 closure: concrete fixture values): construct
   a B5 RollupRow with two `FoldCI` entries carrying
   concrete distinct values: entry 0 = `FoldCI(
   fold_index=0, n_seeds=3, n_entities=15,
   metric_mean=0.50, metric_ci_lo=0.45, metric_ci_hi=0.55,
   ci_method="bca", ci_fallback_reason=None)` and entry 1
   = `FoldCI(fold_index=1, n_seeds=3, n_entities=15,
   metric_mean=0.60, metric_ci_lo=0.55, metric_ci_hi=0.65,
   ci_method="bca", ci_fallback_reason="p0_at_edge")`.
   Write + load via `write_rollup` / `load_rollup`. Assert
   every field on every entry survives the round-trip
   exactly (the `None` and `"p0_at_edge"` distinction
   catches a silent pd.NA → None coercion regression).
6. `test_experiment_spec_per_fold_cis_disabled_by_default`:
   construct `ExperimentSpec(kind="raw_loss")` and assert
   `bootstrap_per_fold_cis_enabled == False` (backward
   compat default).
7. `test_aggregator_writes_per_fold_cis_none_when_disabled`:
   end-to-end through the B5 raw-loss aggregator with the
   spec default (per-fold disabled). Assert
   `row.per_fold_cis is None` on every emitted row.
8. `test_aggregator_writes_per_fold_cis_list_when_enabled`:
   end-to-end through the B5 aggregator with
   `bootstrap_per_fold_cis_enabled=True`. Assert
   `row.per_fold_cis is not None` AND
   `len(row.per_fold_cis) >= 1` AND every entry is a
   `FoldCI` AND each entry's
   `metric_ci_lo <= metric_mean <= metric_ci_hi` (with the
   audit-fields-typed assertion mirroring B21 R1 qa-I1
   closure: ci_method is "bca", ci_fallback_reason is in
   the documented set).
9. `test_aggregator_per_fold_cis_each_fold_uses_distinct_seed`:
   monkeypatch `entity_block_bootstrap_ci` to capture the
   `seed` kwarg on every call. Run the B5 aggregator with
   per-fold enabled on a fixture with 3 folds. Assert the
   3 captured seeds are all DISTINCT from each other AND
   from the pooled seed (`BOOTSTRAP_DEFAULT_SEED`). Pins
   R-B22-4 seed-derivation contract.
10. `test_aggregator_per_fold_cis_seed_derivation_pin`:
    monkeypatch `entity_block_bootstrap_ci` and capture
    the seed for `fold_index=7`. Assert the captured seed
    equals `BOOTSTRAP_DEFAULT_SEED ^
    BOOTSTRAP_PER_FOLD_SEED_OFFSET ^ 7`. Pins the EXACT
    XOR formula so a regression that swaps the operator
    or omits an operand fails immediately.
11. `test_aggregator_per_fold_cis_omits_zero_row_folds`:
    fixture where fold 2 has zero contributing rows (all
    cells skipped). Assert the emitted `per_fold_cis`
    list contains entries for fold 0 + fold 1 but NOT
    fold 2 (per R-B22-7).
12. `test_aggregator_per_fold_cis_oom_gate_raises`:
    monkeypatch `_bootstrap_aggregate.
    BOOTSTRAP_ROW_COUNT_CEILING` to 1 on a fixture where
    fold 0 has 4 rows + `n_resamples=2`. Assert the
    emitted exception is `RawRollupError` with message
    matching `"fold_index=0"` AND `"per-fold CI"`.
13. `test_aggregator_sentinel_row_carries_per_fold_cis_none_when_enabled`
    (qa-R1-I2 closure: the fixture MUST set
    `bootstrap_per_fold_cis_enabled=True` so the sentinel
    emit path's explicit `per_fold_cis=None` hardcode is
    actually exercised; with the flag off the row would
    carry None regardless of the helper's code):
    fixture with the per-fold flag ON AND the pooled
    aggregator routes through `_emit_sentinel_row` (all
    cells skipped sentinel). Assert the emitted row's
    `per_fold_cis is None` (NOT empty list) per R-B22-8.
14. `test_aggregator_per_fold_cis_single_entity_fold_degenerates_correctly`:
    fixture where one fold has exactly one entity. Assert
    the corresponding `FoldCI` carries
    `metric_mean == metric_ci_lo == metric_ci_hi` AND
    `ci_fallback_reason is None` (per R-B22-7; the
    primitive's degenerate path returns None fallback).
15. `test_aggregator_per_fold_cis_list_is_sorted_by_fold_index_ascending`
    (qa-R1-C1 closure): build a fixture where the loss
    rows are delivered in NON-ascending fold order (e.g.,
    fold 2 rows before fold 0 rows in the cells loop).
    Assert `[fc.fold_index for fc in row.per_fold_cis]`
    is strictly ascending. A mutation that swaps `sorted(
    set(...))` for `set(...)` produces insertion-order
    output that varies with the fixture and would fail
    this assertion.
16. `test_aggregator_per_fold_cis_each_fold_receives_correct_n_resamples`
    (qa-R1-C2 closure): monkeypatch
    `entity_block_bootstrap_ci` to capture `n_resamples`
    on every call. Run the B5 aggregator with per-fold
    enabled AND `bootstrap_n_resamples=137` on the spec.
    Assert every captured per-fold `n_resamples` value
    equals 137 (not silently dropped or hardcoded to a
    profile default). Pins the spec → per-fold call
    propagation contract.
17. `test_non_b5_aggregator_writes_per_fold_cis_list_when_enabled`
    (qa-R1-I1 closure): run one non-B5 aggregator (B6
    pairwise OR B7 training-time) with
    `bootstrap_per_fold_cis_enabled=True`. Assert the
    emitted row's `per_fold_cis is not None` AND every
    entry is a `FoldCI` instance AND each entry's audit
    fields are populated. Catches a conditional omission
    in any non-B5 aggregator (e.g., a missing `if
    _spec_has_per_fold_cis_enabled` guard).
18. `test_non_b5_rollup_row_per_fold_cis_round_trip`
    (qa-R1-I3 closure): write + load a non-B5 RollupRow
    (e.g., `PairwiseRollupRow`) with a 2-entry
    `per_fold_cis` list via `write_pairwise_rollup` /
    `load_pairwise_rollup`. Assert every `FoldCI` field
    survives. Closes the pyarrow nested-struct round-
    trip gap for schemas beyond B5.
19. `test_b16_aggregator_per_fold_cis_does_not_populate_oracle_field`
    (qa-R2-I2 closure: pin the D-B22.2 deferral
    explicitly): run the B16 ensemble_lift aggregator with
    `bootstrap_per_fold_cis_enabled=True` on a fixture
    including oracle loss data. Assert (a) the emitted
    `EnsembleLiftRollupRow` has `per_fold_cis is not None`
    (main delta per-fold IS computed), AND (b) the row's
    schema does NOT carry an attribute named
    `per_fold_oracle_cis` (oracle per-fold is deferred
    under D-B22.2). The second assertion catches a
    silent v1 scope creep where a coder accidentally adds
    the oracle field; the test enforces R-B22-Risk-5.

Expected test delta after the build:
- Existing tests: 903 → 903 (no count change; default-off
  preserves behavior; the byte-pin asserts + the test #16
  per_fold_cis None pin are inline extensions of existing
  tests).
- B22-new: 19 tests.
- Total: 903 + 19 = 922 expected post-refactor.

## B22.7 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B22-Risk-1 | Adding a list field to all 5 RollupRow schemas breaks every existing fixture site. | Low | `per_fold_cis: list[FoldCI] \| None = None` is schema-default; existing fixtures omit it and pass byte-equivalent. R-B22-2 documents this. |
| R-B22-Risk-2 | Pyarrow / parquet handling of nested struct columns is brittle for `list[BaseModel]` round-trips. | Medium | The existing schemas already round-trip nullable primitives via the `pd.NA -> None` coercion. The new field is `list[FoldCI] \| None`; pydantic dumps it as `list[dict]` via `model_dump()`, which pyarrow writes as a struct-array column. The B22.6 test #5 explicitly round-trips a 2-entry list through `write_rollup` / `load_rollup`; if the round-trip is broken, the test catches it before merge. |
| R-B22-Risk-3 | Per-fold CIs at small-N (e.g., 3-5 entities per fold) frequently degenerate to BCa fallback (`p0_at_edge` or `a_overshoot`). | Low (acknowledged) | The audit fields `ci_method` + `ci_fallback_reason` on each `FoldCI` make the degenerate state visible. Test #14 explicitly exercises the single-entity degenerate path; test #8 type-checks the fallback reason. Readers consuming per-fold CIs from notebooks can filter out folds with non-None `ci_fallback_reason` if they want only non-degenerate bounds. |
| R-B22-Risk-4 | The per-fold OOM gate is per-fold; a dataset whose total rows exceed the ceiling but each fold fits could SILENTLY skip the pooled OOM gate (which fires on the total). | Low | The pooled OOM gate runs FIRST (existing B14/B15/B16 contract); the per-fold gate is checked AFTER the pooled gate succeeds. R-B22-6 references the same `BOOTSTRAP_ROW_COUNT_CEILING` constant; a fold that individually exceeds the ceiling raises. |
| R-B22-Risk-5 | The B16 ensemble_lift aggregator has TWO bootstrap invocations (main + oracle); per-fold CIs apply only to the MAIN delta at v1. A reader who expects per-fold oracle CIs will be surprised. | Low | R-B22-1 specifies the v1 scope; oracle per-fold CIs are deferred under D-B22.2. The schema field name (`per_fold_cis`) is generic enough that a future expansion to oracle (e.g., `per_fold_oracle_cis`) would be additive. |

## B22.8 Implementation outline

1. **Constant**: add `BOOTSTRAP_PER_FOLD_SEED_OFFSET:
   int = 0xB22_F01D_C1` to
   `benchmarks/report/_bootstrap_aggregate.py`. Export via
   `__all__`. Add `_compute_per_fold_cis` helper function
   in the same module (consumed by all 5 aggregators).
2. **Schema**: add `FoldCI` BaseModel + `per_fold_cis:
   list[FoldCI] | None = None` field to all 5 RollupRow
   schemas in `benchmarks/bootstrap_manifest.py`.
3. **ExperimentSpec**: add
   `bootstrap_per_fold_cis_enabled: bool = False` to
   `benchmarks/config.py`.
4. **Aggregators**: each of the 5 aggregators consults
   `_spec_has_per_fold_cis_enabled(config.experiments,
   kind=...)`; if True, computes `per_fold_cis` via the
   new helper and plumbs into the row constructor.
5. **Sentinel emit helpers**: each aggregator's
   `_emit_sentinel_row` helper hardcodes `per_fold_cis=None`
   (R-B22-8 contract).
6. **Update existing fixtures**: no fixture site needs
   updates because the schema default `None` covers every
   pre-B22 row construction. Inline asserts added to test
   #16 (B21 schema-default isolation) and the 4 B17
   byte-pin renderer tests (non-exposure).
7. **NEW tests**: add
   `tests/benchmarks/test_b22_per_fold_cis.py` with the
   19 tests (14 design-named + 5 added in R1/R2 closures:
   qa-R1-C1/C2/I1/I3 added tests #15-#18; qa-R2-I2 added
   test #19 for the B16 oracle non-population pin).
8. **Verify**: ruff + pyright clean; 922 tests pass.

## Addressed

R1 design swarm: architecture-reviewer (2C / 4I / 2N
REQUEST_CHANGES), qa-test-coverage (2C / 3I / 2N
REQUEST_CHANGES), style-reviewer (0C / 0I / 0N APPROVE).
Deduplicated total: 4 CRITICAL, 6 IMPROVEMENT, 3 NITPICK
(matches the 13 enumerated closure entries below; the
original R1 dedup tally of 4C/7I/4N counted 1 IMP + 1 NIT
that were merged into other entries without separate
listings. B27 / D-B22.5 closure: header corrected to match
the enumeration).
Closures:

- **arch-R1-C1** (pseudocode computed `n_seeds_in_fold` and
  `n_unique_entities` both from `np.unique(entities[mask])`,
  making the two `FoldCI` fields identical; `n_seeds` must
  track the seed axis, not the entity axis): added a new
  `seeds: np.ndarray` parameter to `_compute_per_fold_cis`
  parallel to `fold_indices`; `n_seeds_in_fold` now computes
  `np.unique(fold_seeds).shape[0]`. The aggregators extend
  their cells loops with a per-row `seed_blocks` parallel
  array (B5 broadcasts the cell's seed across its per-row
  losses; B6/B7/B8 use the cell-as-entity contract where
  `seeds[i] = ok_cells["seed"][i]`).
- **arch-R1-C2** (design referenced `fold_indices` array
  parallel to losses but the existing B5 cells loop at
  `bootstrap_rollup.py:273-304` only tracks
  `losses_blocks` + `entity_blocks`): B22.1 pseudocode
  extended with the explicit `fold_blocks` + `seed_blocks`
  construction. The 4 other aggregators' fold_indices +
  seeds sources are explicitly named.
- **qa-R1-C1** (order-stability of `per_fold_cis` not
  tested; a `sorted -> set` mutation would silently produce
  insertion-order output): added test #15
  `test_aggregator_per_fold_cis_list_is_sorted_by_fold_index_ascending`
  delivering rows in non-ascending fold order and asserting
  the emitted list is sorted ascending.
- **qa-R1-C2** (`bootstrap_n_resamples` propagation not
  tested; a hardcoded n_resamples in `_compute_per_fold_cis`
  would pass all 14 original tests): added test #16
  `test_aggregator_per_fold_cis_each_fold_receives_correct_n_resamples`
  capturing the `n_resamples` kwarg on every per-fold call
  and asserting equality with the spec value.
- **arch-R1-I1** (`FoldCI` not in `__all__`): R-B22-1
  expanded to state the export. B22.8 step 2 implicitly
  mandates the export via the schema additivity contract.
- **qa-R1-I1** (non-B5 aggregator on-path uncovered):
  added test #17
  `test_non_b5_aggregator_writes_per_fold_cis_list_when_enabled`
  exercising B6 (or B7) with the flag on.
- **qa-R1-I2** (test #13 sentinel fixture didn't mandate
  `enabled=True`; vacuous pass with flag off): test #13
  renamed
  `test_aggregator_sentinel_row_carries_per_fold_cis_none_when_enabled`
  with an inline note mandating the flag-on fixture.
- **qa-R1-I3** (parquet round-trip covered B5 only;
  pyarrow nested struct column handling rated Medium
  severity): added test #18
  `test_non_b5_rollup_row_per_fold_cis_round_trip`
  exercising a non-B5 schema.
- **arch-R1-N1** (XOR distinctness invariant not stated
  in the constant's comment; resetting offset to 0 would
  silently re-correlate the pooled + fold-0 streams):
  comment expanded to state the invariant explicitly +
  the regression risk.
- **qa-R1-N2** (test #5 fixture values underspecified):
  test #5 description now enumerates concrete distinct
  values per `FoldCI` entry including a `None` vs
  `"p0_at_edge"` pair on the `ci_fallback_reason` field
  to catch silent pd.NA coercion regressions.
- **arch-R1-I3** (single cross-cutting flag breaks the
  per-kind-flag precedent; the existing 5 spec flags
  (`bootstrap_pairwise_enabled` etc.) are per-aggregator):
  NOT changed. The per-fold-CIs feature is uniform across
  all 5 aggregators (the underlying mechanism is the same
  helper consumed by every aggregator); a per-kind flag
  would be 5 redundant booleans with no per-kind variation
  at v1. The single cross-cutting flag is the cleaner
  choice; D-B22.4 (deferred) names per-kind override if
  a future reader needs to enable per-fold CIs selectively.
- **arch-R1-I4** (B16 ensemble_lift dual-bootstrap scope
  test pin missing; v1 covers main delta only): added an
  explicit pin to test #8's assertion. If `_happy_per_cell`
  produces oracle data, test #8 additionally asserts that
  the emitted EnsembleLiftRollupRow has `per_fold_cis is
  not None` (main per-fold computed) but the schema does
  NOT carry a per-fold ORACLE field at v1; oracle per-fold
  is deferred under D-B22.2.
- **arch-R1-N2** (B22.2 ExperimentSpec sentence about
  "each aggregator matches against its OWN kind" could
  read as ambiguous): NOT changed; the helper signature
  takes `kind` as an explicit string, making the per-kind
  dispatch obvious in the pseudocode.

Test count after R1 closures: 18 new tests (was 14;
arch-R1-C1/C2 didn't add tests but the qa-R1-C1/C2/I1/I3
closures added tests #15-#18); total `903 + 18 = 921`.

### R2 confirming swarm closure

R2 confirming swarm on commit `af1497d`: architecture-
reviewer (0C / 0I / 3N APPROVE), qa-test-coverage (0C /
2I / 1N APPROVE), style-reviewer (0C / 0I / 0N APPROVE).
All three APPROVE. Deduplicated total: 0 CRITICAL, 2
IMPROVEMENT, 4 NITPICK. Closures:

- **arch-R2-N1 + qa-R2-I1** (B22.8 step 7 still said "14
  tests" and step 8 still said "917 tests pass" after R1
  closures widened to 18 / 921): updated step 7 to "19
  tests (14 design-named + 5 R1/R2 closures)" and step 8
  to "922 tests pass". The +1 over the R1 total reflects
  test #19 added by qa-R2-I2.
- **qa-R2-I2** (arch-R1-I4 closure was asserted in the
  Addressed section but no test in the 18-item ledger
  backed it; test #8 is the B5 aggregator, not B16):
  added test #19
  `test_b16_aggregator_per_fold_cis_does_not_populate_oracle_field`
  with explicit assertions on the B16 aggregator's
  emitted row (main per-fold computed, oracle field
  absent from schema).
- **arch-R2-N2** (B22.1 pseudocode named `ok_cells` for
  non-B5 aggregators but `bootstrap_pairwise.py`,
  `bootstrap_training_time.py`, and `bootstrap_hpo_uplift.py`
  all bind to `ok` instead): rewrote the non-B5 paragraph
  to reference `ok` with a parenthetical noting B5 binds
  `ok_cells` specifically.
- **arch-R2-N3** (the R1 closure block's dedup header
  read "4C / 7I / 4N" but the enumerated entries are
  4C / 6I / 3N; arch-R1-I2 unaccounted for): NOT changed.
  The original arch-R1-I2 finding (B22.6 "6 fixture
  sites" miscount) was folded into arch-R1-I1 in the
  closure phrasing; both items refer to the same
  fixture-site enumeration. The header count is a
  bookkeeping inaccuracy that does not affect the
  closure substance; deferred under D-B22.5 for a
  documentation cleanup pass.
- **qa-R2-N1** (test #8 asserts `ci_method is "bca"`
  hardcoded rather than referencing the constant):
  NOT changed. The constant
  `BOOTSTRAP_DEFAULT_CI_METHOD` was set to `"bca"` in
  B21 R-B21-6 and the v1 contract pins it there;
  referencing the constant in the test would weaken the
  pin against a constant flip. The hardcoded `"bca"` is
  intentional.

Test count after R2 closures: 19 new tests (was 18;
qa-R2-I2 added test #19); total `903 + 19 = 922`.

## Deferred

- **D-B22.1**: surface per-fold CIs in the renderer
  markdown as an expandable sub-table or as a per-fold
  variance footnote. v1 keeps the field as a parquet-
  shard audit column only; downstream consumers can read
  it via `load_*_rollup`.
- **D-B22.2**: per-fold CIs for the B16 ensemble_lift
  ORACLE Δ bootstrap (independent from the main Δloss
  per-fold). v1 ships per-fold CIs only for the main
  delta; oracle per-fold would add a sibling field
  `per_fold_oracle_cis` and a third bootstrap invocation
  per fold.
- **D-B22.3**: per-fold CIs across SEEDS (currently per-
  fold means bootstrapping the test-rows OF that fold;
  a complementary view would bootstrap across the seeds
  WITHIN a fold, yielding a fold-mean uncertainty). v1
  ships the test-row bootstrap; the seed-mean bootstrap
  is a separate audit field.
- **D-B22.4** (arch-R1-I3 closure): per-kind override on
  `ExperimentSpec.bootstrap_per_fold_cis_enabled` (e.g.,
  enable per-fold CIs for B5 + B16 but not B6 + B7 + B8).
  v1 ships a single cross-cutting flag because the
  underlying helper is uniform; per-kind override would
  add 5 redundant booleans with no per-kind variation at
  the v1 helper level.
- **D-B22.5** (arch-R2-N3 closure): documentation cleanup
  pass on the R1 closure block to reconcile the dedup
  header counts with the enumerated closure entries.
  Bookkeeping accuracy only; does not affect closure
  substance.
