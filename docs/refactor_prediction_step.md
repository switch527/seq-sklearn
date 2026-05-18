# S4 refactor plan: prediction_step default 1->0 + F1 output row-order

Point-for-point code change spec implementing the S1-S3 consensus'd
design (requirements F1/F3/F6/N1, architecture A4/A5 + Phase 9 ledger).
Grounded in the current code; every step lists file, symbol, the exact
change, and blast radius. Steps are ordered by dependency. The 12
mandatory tests from the implementation_plan Phase 9 ledger (the
original 10 plus S5 R2 additions #11 internal-split calibration-fold
and #12 dedicated generator-removal; #9 extended with two error-path
raises) are mapped to steps. S5 reviews this plan; S6 implements it;
S7/S8 review the code.

## Invariants this refactor must hold

- I-A: `TabularToSequenceConfig.prediction_step` and the
  `TabularConfigParams` adapter default to `0` (contemporaneous).
- I-B: `transform()` is stateless: it returns the caller-order restore
  permutation as a per-call dict key `input_row_order`; it sets NO new
  attribute on the transformer instance.
- I-C: every caller-facing prediction surface (`predict`,
  `predict_proba`, `predict_quantiles`, `predict_with_attention` /
  `predict_with_states`) returns rows in the caller's input `X` row
  order, applying `input_row_order`.
- I-D: the explicit `calibration_set` fold pairs model outputs with
  `y_cal` in a consistent order (reorder by `input_row_order`).
- I-E: one window terminates at each input row; an N-row entity yields
  N predictions; below-`min_periods_predict` rows are NaN-filled (not
  dropped). `len(predict(X)) == len(X)`, `predict(X)` index-aligned to
  `X`.
- I-F: F6 generator is internally coherent (no vestigial label-less
  `prediction_step`).

## Step 1 - config defaults (I-A)

1a. `src/seq_sklearn/config/tabular.py`: `TabularToSequenceConfig`
field `prediction_step: int = Field(default=1, ge=0)` -> `default=0`.
The `ge=0` constraint is unchanged (0 already legal).

1b. `src/seq_sklearn/config/_adapters.py` `TabularConfigParams`:
three sites, all `1` -> `0`: the class-level `prediction_step: int`
annotation (no default value there, leave as-is), the `__init__`
keyword default `prediction_step: int = 1` -> `0`, and confirm the
`__init__` body assignment `self.prediction_step = prediction_step`
needs no change; the `to_pydantic()` passthrough already passes
`self.prediction_step` (no change). Verify no other adapter hardcodes
`1`.

Blast radius: 2 files. Pinned by mandatory test #2 (dual-layer
default-is-0 guard) and #1 (contemporaneous default behavior).

## Step 2 - transform emits the stateless restore permutation (I-B)

`src/seq_sklearn/data/tabular_to_sequence.py` `transform()`:

2a. The current `ordered = X.sort_values([cfg.id_col, cfg.time_col])
.reset_index(drop=True)` DISCARDS the original row mapping. Change so
the original positional index survives the sort WITHOUT mutating `X`
or relying on a possibly-non-unique `X.index`: capture
`orig_pos = np.arange(len(X))`, attach it as a transient positional
key alongside the sort (e.g. sort an index array:
`sort_order = np.lexsort((X[time_col].to_numpy(),
X[id_col].to_numpy()))` is NOT safe for arbitrary dtypes; instead use
`ordered = X.assign(**{_POS: np.arange(len(X))}).sort_values(
[id_col, time_col], kind="stable").reset_index(drop=True)` where
`_POS` is a private, collision-checked column name, then read
`ordered[_POS].to_numpy()` and drop `_POS` before feature extraction).
Stable sort so ties are deterministic (N4). ORDERING CONSTRAINT: the
`.assign(**{_POS: np.arange(len(X))})` MUST execute BEFORE
`sort_values`. Assigning `_POS` to an already-sorted frame records
sorted positions, not original caller positions, and silently defeats
the entire restore mechanism.

2b. The emission loop is NOT a flat `range(len(X))`. It is
`for entity_code, (_, group) in enumerate(ordered.groupby(cfg.id_col,
sort=True))` and, inside, `for window_end in range(n_rows)` where
`window_end` is GROUP-LOCAL (`n_rows = len(group)`) and one window
terminates at each group row. So capture the per-group positional
slice once per group: `group_pos = group[_POS].to_numpy()` (the
`_POS` values for that group, already in ordered (id, time) order),
and for each emitted window append `group_pos[window_end]` to a
running `emitted_pos` list. After both loops, `emitted_pos` is the
caller-X positional index of every emitted window, in emission order.

2c. Restore-permutation construction with an explicit precondition,
not an appeal to I-E. PRECONDITION: `transform()` emits exactly one
window per input row and drops none, so `emitted_pos` is a dense
permutation of `0..len(X)-1`. This holds because: (i) the `min_periods`
fit-time row drop is gated to fit only
(`tabular_to_sequence.py:166`), never reached on the predict path;
(ii) `min_periods_predict` only NaN-fills below-floor rows, it does
not drop them (`tabular_to_sequence.py:352`); (iii) no other code path
in `transform()` skips a row. Enforce the precondition cheaply before
trusting it: `if len(emitted_pos) != len(X): raise DataContractError(
...)`. TESTABILITY SEAM (so the guard is not a vacuous test): factor
the precondition-check + argsort into a small module-private helper
`_restore_permutation(emitted_pos: list[int] | np.ndarray, n: int) ->
np.ndarray` in `tabular_to_sequence.py` that does exactly the
`len != n` raise then `np.argsort(..., kind="stable")`. `transform()`
calls `_restore_permutation(emitted_pos, len(X))`. Mandatory test
#9(d1) unit-tests this helper DIRECTLY with a deliberately short
`emitted_pos` (e.g. `_restore_permutation([0, 1], 3)`) so the raise
is exercised by a real call, not an unconstructable data path, and a
mutation that flips `!=` to `==` or deletes the guard is killed.
Then `input_row_order = np.argsort(np.asarray(emitted_pos),
kind="stable")` (inside the helper). Indexing any emitted-order array
by `input_row_order`
restores caller `X` row order (argsort of a dense permutation is its
inverse; stable for determinism under N4).

2d. Add to the returned dict:
`"input_row_order": torch.as_tensor(input_row_order, dtype=torch.long)`.
Do NOT assign it on `self` (statelessness, I-B; concurrent-predict
race otherwise). Guard the private column name with an explicit
raise, NOT `assert` (asserts are stripped under `python -O`):
`if _POS in X.columns: raise DataContractError(...)`, placed before
the `assign`.

2e. Training-path containment (the new key must NOT leak into the
training batch dict). `transform()` is also called on the FIT path:
`trainer.py:376` `batch = self.transformer.transform(x_panel)`, then
`trainer.py:416` `dataset = _TensorDictDataset(batch)`.
`_TensorDictDataset.__getitem__` (`trainer.py:87`) iterates ALL keys
(`{key: value[index] for key, value in self._batch.items()}`), so the
new `(N,)` `input_row_order` would be sliced per-window and default-
collated into the training batch. The TFT backbone reads keys by name
and would ignore it (no crash), but (i) it pollutes every training
batch with predict-only metadata and (ii) any test asserting the
exact training-batch key set breaks. PRESCRIPTION: in `Trainer.fit`,
immediately after `entity_ids = batch["entity_id"].cpu().numpy()`
(`trainer.py:377`) and before `_TensorDictDataset(batch)`
(`trainer.py:416`), pop the key: `batch.pop("input_row_order",
None)`. `input_row_order` is a predict-time restore artifact with no
training meaning; the fit path never restores caller order. Use
`pop(..., None)` so fit stays correct even if a future transform
variant omits the key.

Blast radius: 2 files (`tabular_to_sequence.py`, `trainer.py`; the
`trainer.py` touch is the single `batch.pop` in `fit`). Pinned by
mandatory test #9 (statelessness: `input_row_order` in the dict, no
instance attribute, no cross-call leak; plus the two new error-path
raises) and #3 (shuffled string-id row-order).

## Step 3 - base predict path restores caller order (I-C, I-E)

`src/seq_sklearn/models/_base.py`:

3a. `_predict_raw(X)` (`_base.py:587`) currently returns
`tuple[Tensor, np.ndarray]` (logits in transform/sorted order,
`below` mask, also sorted-space). Change its signature to the
3-tuple `tuple[Tensor, np.ndarray, np.ndarray]`:
`(raw, below, input_row_order)` where `input_row_order` is read from
the same `batch` produced by the internal `transform()` call (the
batch already drives the forward; no second transform). Single
committed signature, NOT "extend the tuple OR return the batch" -
one shape so call sites are mechanical. There are exactly TWO literal
`_predict_raw` callers to update (verified): `_classifier.
predict_proba` (`_classifier.py:124`, `raw, below =
self._predict_raw(X)` -> `raw, below, iro = self._predict_raw(X)`)
and `_regressor._calibrated_matrix` (`_regressor.py:78`, the
`_predict_raw` call inside it; `_calibrated_matrix` def is at
`_regressor.py:78`). The classifier `predict` (`_classifier.py:147`)
and regressor `predict` / `predict_quantiles` (`_regressor.py:106` /
`:115`) inherit the reorder transitively through `predict_proba` /
`_calibrated_matrix` - they are NOT separate `_predict_raw` callers.
DTYPE: `input_row_order` in the batch dict is `torch.long`
(Step 2d). A torch tensor cannot index a numpy array; convert once at
the consumer: `iro = batch["input_row_order"].cpu().numpy()` and use
the numpy form for every `out[iro]`. The threshold-tuning /
`_post_fit` path does NOT call `_predict_raw` on caller X (it
operates in sorted calibration space, see Step 5) and is unaffected -
state this so a future reader does not thread `iro` through it.

3b. Apply the permutation as the LAST step before returning the numpy
array, once, at the array boundary (not inside `_proba_from_raw`):
`out = out[iro]`. ORDER-OF-OPERATIONS (correctness trap): `below` is
in sorted (transform) space, same as `raw`. The below-floor NaN-fill
`out[below] = np.nan` MUST be written FIRST, while `out` is still in
sorted order; only THEN `out = out[iro]`. Do NOT permute `below`
itself, and do NOT apply `out[iro]` before the NaN-fill - either
ordering NaN-fills the wrong rows (silently correct on an already-
sorted CPU panel, wrong whenever a predict call shuffles an entity
block). Concretely in `predict_proba`: `raw, below, iro =
self._predict_raw(X); proba = self._proba_from_raw(raw);
proba[below] = np.nan; proba = proba[iro]; return proba`. Callers
that reorder: classifier `predict` / `predict_proba`, regressor
`predict` / `predict_quantiles` (via `_calibrated_matrix`).
EXCLUSION: `_regressor._calibrate_raw` (def at `_regressor.py:91`)
must NOT reorder - it is shared with `predict_with_attention`, which
does its own per-field reorder in Step 4; reordering inside
`_calibrate_raw` would double-permute the attention path.
`_calibrate_raw` stays in sorted space; only its array-boundary
caller `_calibrated_matrix` (`_regressor.py:78`) applies `iro`
(after the same below-first NaN-fill ordering).

Blast radius: `_base.py`, `_classifier.py`, `_regressor.py`. Pinned by
mandatory tests #3 (array surfaces, shuffled string-id panel) and #6
(below-min_periods_predict NaN-fill preserves count, index-aligned).

## Step 4 - predict_with_attention restores every per-row field (I-C)

`src/seq_sklearn/models/transformer/_base.py`
`predict_with_attention` (classifier + regressor variants): apply
`input_row_order` to EVERY per-row field before constructing the
dataclass. Enumerated (verified against
`src/seq_sklearn/inference/attention.py`):
- `AttentionOutput` (8, all per-row): `predictions`, `probabilities`,
  `logits`, `var_selection_weights`, `static_var_selection_weights`,
  `attention_weights`, `padding_mask`, `entity_id`.
- `RegressionAttentionOutput` (6 per-row): `predictions`,
  `var_selection_weights`, `static_var_selection_weights`,
  `attention_weights`, `padding_mask`, `entity_id`.
- `RegressionAttentionOutput.quantiles_used`: fit-time
  `tuple[float,...] | None`, NOT per-row -> leave untouched
  (shuffle-invariant).

Note on `padding_mask`: its source in `predict_with_attention` is the
post-forward `output.padding_mask` (the `BackboneOutput` field), NOT
`batch["padding_mask"]`; both are in transform/sorted row order, so
reorder `output.padding_mask` by `input_row_order` exactly like the
other per-row tensors. `input_row_order` itself comes from the same
`batch` the forward consumed (Step 3a seam) - thread it into
`predict_with_attention`, do not recompute.

Note on prediction-bearing fields (no double-reorder). `predictions`
(regressor) and `predictions` / `probabilities` / `logits`
(classifier) inside `predict_with_attention` are computed from a
LOCAL forward / local `_calibrate_raw` / head call WITHIN
`predict_with_attention`, in sorted space - they do NOT flow through
the Step 3b `_predict_raw` array-boundary reorder (Step 3b's
EXCLUSION of `_calibrate_raw` is precisely what keeps this path
single-reorder). Therefore `predict_with_attention` itself applies
`input_row_order` ONCE to every per-row field (the prediction fields
and the introspection fields alike), and there is no
already-reordered input feeding it. Single reorder, at this one
site, for the whole dataclass.

Blast radius: 1 file. Pinned by mandatory test #8 (full enumeration +
`quantiles_used` invariance).

## Step 5 - calibration-fold alignment (I-D)

`src/seq_sklearn/models/_base.py` `_calibration_fold` (def
`_base.py:336`). It has TWO consumers, both of which must inherit the
fix, so the reorder is pinned INSIDE `_calibration_fold` (not at a
caller): (1) the calibrator-fit path `_base.py:406`
(`raw, targets = self._calibration_fold(...)`), and (2) the
threshold-tuner path `_classifier._post_fit` (`_classifier.py:151`,
calling `_calibration_fold` at `_classifier.py:164` then feeding
`ThresholdTuner`). This is the exact path mandatory test #7 pins. The
"Blast radius: 1 file" framing must NOT be read as "one call site" -
fixing inside `_calibration_fold` is what makes both consumers
correct at once; fixing at only one caller silently reintroduces the
Gemini G-C2 F2 mispairing for the other.

Two branches with explicitly different row-set behavior. State both
in the code so neither is "fixed" later:

- Explicit `calibration_set` branch (`_base.py:355-358`, currently
  `batch = transformer.transform(x_cal); return
  self._raw_outputs(batch), torch.as_tensor(
  self._encode_targets(y_cal))`): raw outputs are in
  `transform(x_cal)` SORTED order and span ALL x_cal rows (transform
  drops none on this path; below-floor rows are NaN-filled, not
  dropped - same dense-permutation precondition as Step 2c). `y_cal`
  is in caller order. SINGLE RULE: reorder the raw outputs by
  `batch["input_row_order"]` so they land in caller order, then pair
  with `y_cal`. Do NOT offer the "or index y_cal by the inverse"
  alternative - one direction only, so the calibrator/threshold tuner
  always sees (caller-order outputs, caller-order y_cal). No
  redundant length check is needed here: `batch["input_row_order"]`
  already satisfies the Step 2c dense-permutation precondition (it is
  produced by the same `transform()` call and `transform()` is the
  sole producer; the `len(emitted_pos) != len(X)` raise lives inside
  `transform()`, so this consumer inherits the guarantee). The
  NaN-filled below-floor rows must be handled by the existing
  calibration-fit masking exactly as today (this refactor does not
  change which rows the calibrator drops, only their order).
- Internal-split branch (no explicit `calibration_set`,
  `_base.py:359-379`): derives `cal_idx` from the SAME sorted `batch`
  and pairs `_raw_outputs(batch)[cal_idx]` with `batch["target"]
  [cal_idx]` - both already in sorted space, indexed by the same
  `cal_idx`, so self-consistent. NO change. Add a code comment
  stating this branch is intentionally sorted-space-consistent and
  must NOT have `input_row_order` applied (doing so would MIS-order a
  currently-correct pairing).

Blast radius: 1 file (`_base.py`, the single reorder inside
`_calibration_fold`; both consumers `_base.py:406` and
`_classifier.py:164` inherit it). Pinned by mandatory test #7
(explicit-`calibration_set` branch, mispairing-sensitive, unsorted
string-id `X_cal`) AND mandatory test #11 (internal-split branch
stays sorted-space self-consistent, `input_row_order` NOT applied).

## Step 6 - F6 generator coherence (I-F) [S5 DECISION: REMOVE]

`src/seq_sklearn/data/synthetic/generator.py` declares
`prediction_step: int = 1` and uses `target_idx = window_end +
prediction_step; if target_idx >= n_periods: continue` purely as a
tail-window skip-guard; it never shifts the label (steps 7-9 compute
`z` from `phi(window)`).

DECISION (S5 confirmed - recommended option adopted): **remove the
generator `prediction_step` parameter entirely** and the skip-guard.
Rationale: F6 is contemporaneous by spec; the parameter never produced
forecast-aligned data, so it is dead surface; v1 N1 acceptance is
entirely contemporaneous (no forecast-mode synthetic data needed);
removing it is the lowest-risk way to reach I-F and makes
mandatory-test #5's "parity with generator forecast mode" clause moot
(test #5 then pins only `TabularToSequence.prediction_step>0`'s
horizon-edge clamp, which is the real, kept forecasting path).
Alternative considered and deferred: make `generator.prediction_step>0`
emit genuinely forecast-aligned targets so forecast-mode synthetic
data exists; deferred (one-line entry in the implementation_plan
Deferred section) because it is net-new functionality (more code +
risk) outside v1 scope and not required by any N1 threshold.

6a. Removal completeness guard. After deleting the parameter, the
skip-guard, and the `target_idx`/`n_periods` machinery, run
`grep -rn "prediction_step" src/seq_sklearn/data/synthetic/` and
confirm ZERO hits remain (no vestigial default, type annotation, or
`__init__` keyword). The generator must not expose `prediction_step`
on any surface (constructor signature, dataclass field, config
passthrough). State this as an explicit done-criterion, not an
implied one. STALE-PROSE COROLLARY: the module docstring
"Target-window emission rule" paragraph (`generator.py:22-29`)
narrates the `prediction_step` skip-guard semantics
("...AND a target row exists ``prediction_step`` periods after the
window end...provided the prediction-step target exists..."). The
grep catches the `prediction_step` token there, but the FULL
paragraph must be rewritten to the contemporaneous rule (one labelled
row emitted per valid window end; no forward target row; entities
shorter than `lookback` still emit one left-padded window), not just
the keyword stripped. Stale narrative with the keyword removed would
pass the grep yet still misdescribe behavior.

6b. Tail-trim count delta (NOT a uniform per-entity offset). The old
skip-guard discarded the final window of each entity whenever
`window_end + 1 >= n_periods`, so removing it changes the generator's
emitted window count per entity from `n_periods - 1` to `n_periods`
(one extra terminal window per entity). This is the CORRECT
contemporaneous count (one window per row, consistent with I-E). Two
consequences for downstream assertions, both routed into Step 8 as
named obligations (not "might also need" asides): (i) every exact
sample/row/tensor-shape count derived from `SyntheticPanelGenerator`
output must be recomputed against `n_periods` (not `n_periods - 1`)
per entity; (ii) the delta is NOT a uniform `+num_entities`: a
single-period entity (`n_periods == 1`) previously emitted ZERO
windows (`n_periods - 1 == 0`) and now emits ONE, so it goes from
ABSENT to PRESENT in the panel. Any test asserting the SET of entity
ids, the number of distinct entities, or per-entity grouping (not
just the total row count) over a panel that can contain
single-period entities must be re-derived, because the entity-id set
itself changes, not only counts.

Blast radius: 1 file (+ test #5 wording + the Step 8 count-delta
sweep). Callers passing `prediction_step=` to
`SyntheticPanelGenerator` in tests must be swept (Step 8).

## Step 7 - the 12 mandatory tests

Implement exactly the implementation_plan Phase 9 ledger items
#1-#12 (file::function targets are pinned there). Net-new test files:
`tests/integration/test_predict_row_order.py` (#3 shuffled-string-id
array order, #6 below-floor NaN-fill count, #7 calibration-fold
explicit-`calibration_set` mispairing-sensitive, #8
predict_with_attention full-field order, #11 internal-split
calibration-fold sorted-space self-consistency), 
`tests/integration/test_contemporaneous_signal_reachable.py` (#4
fast signal floor), and additions to
`tests/unit/data/test_tabular_to_sequence.py` (#1 contemporaneous
default, #5 horizon-edge clamp, #9 statelessness value-oracle PLUS
the two new `transform()` error-path raises: `len(emitted_pos) !=
len(X)` and `_POS`-column collision, both `pytest.raises(
DataContractError)`),
`tests/unit/data/test_synthetic_generator.py` (#12 dedicated
generator-removal test: no `prediction_step` on any surface +
`n_periods` per-entity emission + single-period-entity set change,
kept SEPARATE from the `TabularToSequence` clamp test #5 so neither
can be silently omitted while the other is written),
`tests/unit/config/test_tabular.py` + `test_adapters.py` (#2).

## Step 8 - Phase 1-8 re-validation sweep (#10)

`grep -rn "prediction_step" tests/ src/` and
`grep -rn "SyntheticPanelGenerator(\|TabularToSequenceConfig(\|
TabularConfigParams(" tests/`. For every construction WITHOUT an
explicit `prediction_step`, determine whether the test asserted
behavior that depended on the old `=1` windowing (target alignment,
below-floor counts, sentinel positions, synth->tensor shapes,
window_time_index). Two distinct deltas to chase: (i) the
`TabularToSequence`/config default flip `1 -> 0` (contemporaneous
target alignment, below-floor counts, sentinel positions); (ii) the
Step 6b generator tail-trim removal, which raises emitted windows per
entity from `n_periods - 1` to `n_periods` - EVERY exact
sample/row/tensor-shape count derived from `SyntheticPanelGenerator`
output must be recomputed against the `n_periods` count. Known
suspects from the S2/S3 audit:
`tests/unit/config/test_tabular.py` + `test_adapters.py` (default
assertions, updated in place per #2), `tests/unit/data/
test_tabular_to_sequence.py` (below-floor / sentinel /
target-alignment), `tests/integration/test_synth_to_tensors.py`,
`tests/integration/test_tabular_to_backbone.py`,
`tests/unit/data/test_synthetic_generator.py` (generator
prediction_step usage + tail-trim count, per Step 6),
`tests/e2e/` (`test_acceptance_thresholds.py`,
`test_calibration_coverage_per_strategy.py`, `test_imbalance_smoke.py`,
`test_quickstart.py` - all consume generator output with exact
counts and/or the old default and are the highest-count-delta-risk
suite). Update each to the contemporaneous default and the
`n_periods` count (including entity-id SET re-derivation per Step 6b
consequence (ii) for any panel that can contain single-period
entities), and record the change. The specific known pydantic-default
guard to flip: `tests/unit/config/test_tabular.py:16`
(`assert cfg.prediction_step == 1` -> `== 0`), updated in place per
#2. The Phase 9 quickstart/acceptance `prediction_step`-caused
`xfail`s are removed once #4 (fast signal reachability) is green and
the slow acceptance test passes; the F1.1/sklearn-tags `xfail` is
unrelated and stays.

COMPLETENESS DONE-CRITERION (not implementer judgment). "Known
suspects" is a starting list, not an exhaustive one. The Step 8
obligation is closed only when the commit body records: (a) the
verbatim hit count of `grep -rn "prediction_step" tests/ src/` and
`grep -rn "SyntheticPanelGenerator(\|TabularToSequenceConfig(\|
TabularConfigParams(" tests/`; and (b) a per-hit classification line
for EVERY hit: either "affected, updated" (with the new expected
value) or "not affected" (with a one-clause reason, e.g. passes an
explicit `prediction_step=` kwarg that pins intent). A hit left
unclassified fails the gate. This forecloses a silent implicit-`=1`
reliance in a file not on the suspects list.

## Step 9 - gate

Single background job, gpu deselected: ruff format + ruff check +
pyright (0 errors), then `pytest -m "not gpu"` full Phases 1-9 + 3
randomized seed passes. Coverage bar, stated explicitly (not "the
project bar"): every changed `src/seq_sklearn/` file must hold
`--cov` line coverage at or above the threshold recorded in the
implementation_plan Phase 9 gate, AND the suite-wide coverage delta
must be non-negative (CLAUDE.md Rule 3: "coverage delta must not
decrease"). If the implementation_plan Phase 9 gate names no numeric
floor, the binding constraint is the non-negative-delta rule plus
no-new-uncovered-lines on touched files; record the measured
before/after numbers in the commit body. The commit body must ALSO
carry the Step 8 completeness record (grep hit counts + per-hit
classification); the gate is not green until that record is present.
pgrep orphan sweep after;
specific `git add` of changed paths only (never `-A`; exclude
unrelated docs), then push. The full pre-existing suite must stay
green (Step 8 guarantees no silent dependence on the old default).

## Sequencing & risk

1->2->3->4->5 are the code change (config first so 2-5 build on the
new default; 2 before 3-5 since they consume `input_row_order`). 6 is
independent (generator); the S5 decision is made (remove the vestigial
parameter, Step 6 confirmed). 7-8 follow the code. 9 gates.
Lowest-risk ordering; the only non-mechanical risk is the permutation
construction in Step 2 (mitigated by the test #9 value-oracle plus
tests #3/#6) and the Step 6b tail-trim count delta (mitigated by the
named Step 8 recount obligation).

## S5 review ledger (Addressed / Deferred)

Round 1 (dual-model swarm): style 0/0/0 APPROVE x2; arch-opus,
code-opus, qa-opus, qa-sonnet REQUEST_CHANGES. Addressed: P-C1
(Step 2b group-local `group_pos[window_end]` replacing the
unimplementable `global_row_index_of`); P-C2 (Step 2c explicit
dense-0..N-1 permutation precondition + `len` raise, not an appeal
to I-E); P-C3 (test #9 strengthened to a hand-computed permutation
VALUE oracle); P-C4 (Step 5 two-branch row-set semantics made
explicit); plus the Step 3a single-3-tuple commit, the
`_calibrate_raw` double-reorder EXCLUSION, the Step 6 generator-
removal completeness + tail-trim delta, the Step 8 e2e suspects, the
Step 9 explicit coverage bar, and the N2 explicit-raise.

Round 2 (dual-model swarm): style 0/0/0 APPROVE x2; code-opus
APPROVE (0C/4I/2N); qa-sonnet APPROVE (0C/3I/1N); code-sonnet,
arch-opus, qa-opus REQUEST_CHANGES. Addressed:
- code-sonnet C1: Trainer blast-radius gap - Step 2e added
  (`batch.pop("input_row_order", None)` in `Trainer.fit` before
  `_TensorDictDataset`); Step 2 blast radius corrected to 2 files.
- code-sonnet C2: wrong citations fixed - `_calibrate_raw` def is
  `_regressor.py:91` (was :103), `_calibrated_matrix` is
  `_regressor.py:78` (was :88).
- code-sonnet C3: Step 3b order-of-operations trap - explicit
  "`below` is sorted-space; NaN-fill BEFORE `out[iro]`; never
  permute `below`" with the concrete `predict_proba` sequence.
- arch-opus C1: Step 5 reorder pinned INSIDE `_calibration_fold`;
  both consumers named (`_base.py:406` calibrator path,
  `_classifier._post_fit` `:151`/`:164` threshold path); test #11
  added for the internal-split branch.
- qa-opus C1: mandatory test #11 added (internal-split calibration-
  fold sorted-space self-consistency, the default `cal_fraction>0`
  path test #7 does not reach).
- qa-opus C2: test #9 extended with two `pytest.raises(
  DataContractError)` error-path cases (len-mismatch precondition,
  `_POS` collision).
- code-opus I1-I4: Step 4 single-reorder note for locally-derived
  prediction fields; Step 3a "two literal `_predict_raw` callers";
  Step 5 inherited-guard note; Step 6b non-uniform delta (single-
  period entity-id SET change).
- code-sonnet I1-I3: Step 6a stale-docstring corollary; Step 3a
  `.cpu().numpy()` dtype-conversion note; Step 2a assign-before-sort
  ordering constraint.
- qa I2: Step 8 + Step 9 completeness done-criterion (record grep
  hit count + per-hit classification in the commit body).
- qa-sonnet I3: test #12 added (dedicated generator-removal test,
  decoupled from #5).
- NITPICKs N1/N2 folded: Step 1b three adapter sites enumerated;
  Step 8 cites `test_tabular.py:16`.

Round 3 (dual-model swarm, confirming): ALL 7 reviewers APPROVE,
ZERO CRITICAL. style 0/0/0 x2; code-opus 0C/3I/2N; code-sonnet
0C/0I/1N; arch-opus 0C/1I/2N; qa-opus 0C/2I/1N; qa-sonnet 0C/2I/2N.
All five R2 CRITICAL fixes independently re-verified correct against
the code (Trainer `batch.pop` placement, `_regressor.py:91`/`:78`
citations, Step 3b NaN-before-permute order, Step 5 reorder pinned
inside `_calibration_fold` with both consumers, tests #11/#12/#9(d)).
Addressed in R3:
- qa-opus I1 + qa-sonnet I1 (converged): test #9(d1) raise was
  unreachable via data so the test risked being vacuous - Step 2c
  now factors a module-private `_restore_permutation(emitted_pos,
  n)` helper and ledger #9(d1) unit-tests it directly with a short
  `emitted_pos`, killing the flip/delete mutation by a real call.
- qa-opus I2 + qa-sonnet I2 (converged): ledger #11 now mandates a
  non-degenerate binary+threshold-tuning fixture with a sensitivity
  clause (`decision_threshold_` neither 0.0 nor 1.0; identity within
  1e-6) so a degenerate panel cannot pass vacuously.
- qa-sonnet N1 + code-opus (prose): ledger #11 `cal_idx`/`keep`
  imprecision fixed to `keep = cal_idx[~_below_floor_mask...]`
  (`_base.py:369`); generator docstring cite corrected `:21-29` ->
  `:22-29`.

Deferred (one-line reasons, non-blocking, recorded for S6/Gemini):
- arch-opus R3-I1: Step 5's explicit-`calibration_set` branch
  inherits only `len(input_row_order) == len(x_cal)` from the
  Step 2c guard, not `len(x_cal) == len(y_cal)`. DEFERRED: this is
  pre-existing late-failure behavior the refactor neither introduces
  nor changes (the `_encode_targets(y_cal)` pairing at `_base.py:358`
  already fails late today on a length mismatch); adding a new
  early-length check is out of scope for a row-order refactor and
  belongs to a separate input-validation pass.
- code-opus R3-I3 (doc-anchor precision in the Step 2c proof and the
  Step 8 in-place-edit anchor): DEFERRED as NITPICK-grade polish; no
  logic change, the implementer has the file:line anchors already.
- qa-sonnet R3-N2: Step 2e `batch.pop` is not pinned by any of
  #1-#12. DEFERRED: the TFT backbone reads batch keys by name so an
  un-popped key produces no observable regression; the pop is a
  cleanliness/blast-radius containment, not a correctness contract,
  and pinning a "key absent" assertion would couple a test to an
  internal batch-dict shape the design intentionally leaves loose.
- qa-opus R2-N1 (1-row/empty `X` edge): DEFERRED NITPICK; the
  `len(emitted_pos) != len(X)` raise plus the dense-permutation
  precondition already make a 1-row `X` a trivial identity.

S5 consensus REACHED after 3 rounds (zero CRITICAL on the confirming
round; every IMPROVEMENT resolved or deferred with a reason; only
NITPICKs remain). Cleared for S6 (implement the code change), then
S7 Claude /review consensus, then S8 Gemini code consensus.
