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
  fields:
  ```python
  class FoldCI(BaseModel):
      model_config = ConfigDict(extra="forbid", frozen=True)
      fold_index: int = Field(ge=0)
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
`(losses, entities)` from the cells loop. After the existing
pooled bootstrap call, the per-fold block fires conditionally:

```python
# B22 / D-B16.3: per-fold CIs (opt-in)
per_fold_cis: list[FoldCI] | None = None
if _spec_has_per_fold_cis_enabled(experiments, kind="raw_loss"):
    per_fold_cis = _compute_per_fold_cis(
        losses=losses,
        entities=entities,
        fold_indices=fold_indices,  # parallel to losses; tracked per row
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

`_compute_per_fold_cis` is a NEW shared helper in
`benchmarks/report/_bootstrap_aggregate.py`:

```python
def _compute_per_fold_cis(
    *,
    losses: np.ndarray,
    entities: np.ndarray,
    fold_indices: np.ndarray,
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
    rows in the input. Folds with zero rows are omitted.

    Raises:
        RawRollupError: a fold's n_rows * n_resamples exceeds
            BOOTSTRAP_ROW_COUNT_CEILING.
    """
    fold_cis: list[FoldCI] = []
    unique_folds = sorted(set(int(f) for f in fold_indices))
    for fold_index in unique_folds:
        mask = fold_indices == fold_index
        fold_losses = losses[mask]
        fold_entities = entities[mask]
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
        n_seeds_in_fold = int(np.unique(entities[mask]).shape[0])  # cell-as-entity
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
# pooled stream AND every other fold's stream.
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
5. `test_rollup_row_per_fold_cis_round_trip_with_two_folds`:
   construct a B5 RollupRow with two `FoldCI` entries,
   write + load via `write_rollup` / `load_rollup`, assert
   every field on every `FoldCI` survives. Use distinct
   non-default values per field to catch silent coercion.
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
13. `test_aggregator_sentinel_row_carries_per_fold_cis_none`:
    fixture where the pooled aggregator routes through
    `_emit_sentinel_row` (all cells skipped sentinel).
    Assert the emitted row's `per_fold_cis is None`
    (NOT empty list) per R-B22-8.
14. `test_aggregator_per_fold_cis_single_entity_fold_degenerates_correctly`:
    fixture where one fold has exactly one entity. Assert
    the corresponding `FoldCI` carries
    `metric_mean == metric_ci_lo == metric_ci_hi` AND
    `ci_fallback_reason is None` (per R-B22-7; the
    primitive's degenerate path returns None fallback).

Expected test delta after the build:
- Existing tests: 903 → 903 (no count change; default-off
  preserves behavior; the byte-pin asserts + the test #16
  per_fold_cis None pin are inline extensions of existing
  tests).
- B22-new: 14 tests.
- Total: 903 + 14 = 917 expected post-refactor.

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
   14 tests.
8. **Verify**: ruff + pyright clean; 917 tests pass.

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
