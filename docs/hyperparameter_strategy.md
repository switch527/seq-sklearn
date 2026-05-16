# Hyperparameter exposure strategy

> **Status: design rationale and promotion procedure.** This document is
> not an authoritative spec. The authoritative sources are:
>
> - `docs/requirements.md`: the config surface, per-hyperparameter
>   stability tiers, the F5 validity matrix, the F7 `suggest_params`
>   contract, and the N1 mandatory tests.
> - `docs/architecture.md`: A4 config schemas and adapter pattern, A16
>   Optuna integration, A12 doc rendering.
> - `docs/implementation_plan.md`: per-phase module and test rosters.
> - the code itself (`src/seq_sklearn/config/`): the verbatim pydantic
>   schemas and the `extract_deprecated_extras` helper.
>
> This document owns only the *why* (the rationale for a four-tier
> hierarchy) and the *promotion procedure* (the living ALPHA → BETA →
> STABLE workflow a maintainer follows when a benchmark identifies a
> new knob worth exposing). It deliberately carries no schemas, code
> blocks, or test tables: those duplicate the sources above and drift
> against them.

## Requirements

The hyperparameter-exposure architecture is graded against:

1. **Benchmark headroom**: deep-sequence libraries are made or broken by
   their ability to compete on standard benchmarks. Benchmark performance
   is dominated by hyperparameter tuning. The library must let users tune
   broadly without us shipping every knob on day one.
2. **First-use ergonomics**: a new user reading the README quickstart
   should see a small, well-documented set of hyperparameters. Field
   counts on `TFTConfig` directly affect the `mkdocstrings` rendered
   field-table size (per architecture A12 / griffe-pydantic); a kitchen-
   sink page is a documented usability cost.
3. **Forward additivity**: new hyperparameters discovered through
   internal benchmark testing must land additively (MINOR bumps under
   the project's semver, per `docs/requirements.md` versioning policy)
   without breaking existing callers, search-space code, save / load
   artifacts, or documentation.
4. **No box-in-a-corner**: nothing in v1 may force a MAJOR bump to
   expose a hyperparameter that future benchmark testing shows is
   important.
5. **sklearn-compatible mutation**: `set_params(optimizer__learning_rate=0.001)`
   must work through nested sub-configs without further plumbing.
6. **Optuna-compatible sampling**: `suggest_params` must continue to
   produce configs closed under the F5 validity matrix; expanding the
   search space to additional hyperparameters is an opt-in operation.
7. **Promotion discipline**: when an internal benchmark identifies a
   hyperparameter that moves the needle, the process to expose it must
   be documented, low-cost, and produce a CHANGELOG entry.
8. **No silent breakage on promotion**: when an `extra` dict key is
   promoted to a typed field, the dict path keeps working (as a
   documented deprecation alias) so existing callers see no behavior
   change.
9. **Save / load type fidelity**: anything passed via the `extra` dict
   escape hatch must survive `model_dump()` then JSON then
   `model_validate()` round-trip with identical types. Save / load
   failures or silent type coercion are prohibited.

## Context

seq-sklearn ships TFT in v1 plus six more deep-sequence models in v2 / v3
(PatchTST, TimesNet, TST, LSTM, GRU, LSTM-FCN). Each is performance-
sensitive in different ways (TFT VSN gating and dropout taxonomy,
PatchTST patch length and stride, TimesNet FFT period detection, and so
on), plus loss-side, optimizer-side, scheduler-side, and sampler-side
knobs that move benchmarks across every model family.

Forty-plus tunable hyperparameters across the library is a conservative
estimate. A flat-with-inheritance config shape reaches roughly sixty
fields by v3 if we just keep extending one config class. That field
explosion (and the resulting kitchen-sink doc page and box-in-a-corner
risk) is what the four-tier hierarchy is designed against.

## The four-tier hierarchy (conceptual)

The configuration surface is partitioned into four tiers. The exact
schemas live in `src/seq_sklearn/config/` and architecture A4; only the
*purpose* of each tier matters here.

- **Tier 1, family sub-configs** (`OptimizerConfig`, `SchedulerConfig`,
  `LossConfig`, `SamplerConfig`): each family of options owns its own
  frozen sub-config with its name, tunable defaults, and an `extra`
  escape hatch. This stops optimizer / scheduler / loss / sampler knobs
  from scattering across the main config.
- **Tier 2, main configs** (`BaseTrainingConfig`, `BaseModelConfig`,
  `<Model>Config`): the small, well-documented surface a quickstart
  user sees. It nests the Tier 1 sub-configs rather than carrying their
  fields flat.
- **Tier 3, advanced model sub-config** (`<Model>AdvancedConfig`): the
  BETA landing zone for model-specific experimental knobs that have
  benchmark evidence but are not yet stability-guaranteed. Empty in v1.
- **Tier 4, the `extra` escape hatch**: an ALPHA-tier landing zone on
  every sub-config, restricted to a JSON-safe value union. It lets a
  maintainer expose a knob for benchmark experimentation with no schema
  change and no release.

The escape hatch plus a documented promotion ramp is the actual
contribution: it satisfies "no box-in-a-corner" (anything can be reached
via `extra` today) and "forward additivity" (anything proven via
benchmark graduates to a typed field additively).

## Promotion path: ALPHA → BETA → STABLE

A new hyperparameter discovered via internal benchmark testing goes
through three stages with a documented promotion gate at each step.

### Stage 1: ALPHA (escape hatch)

A maintainer identifies a knob worth tunable exposure and exposes it
first via the appropriate sub-config's `extra` tuple (for example
`OptimizerConfig(name="adamw", extra=(("amsgrad", True),))`, which is
passed through to `torch.optim.AdamW`).

CHANGELOG entry: "Added support for `amsgrad` AdamW flag via
`OptimizerConfig.extra`. ALPHA: documentation in CHANGELOG only."

### Stage 2: BETA (typed field on a family or advanced sub-config)

Benchmark testing on the synthetic DGP plus at least one external
dataset shows the hyperparameter moves the headline metric by at least
the threshold below. The knob is promoted to a typed field on the
appropriate family sub-config (e.g. `OptimizerConfig.amsgrad: bool =
False`) or on the model's `<Model>AdvancedConfig`. The default must be
behavior-neutral so existing callers are unaffected.

CHANGELOG entry: "Promoted `amsgrad` from `OptimizerConfig.extra` to a
typed `OptimizerConfig.amsgrad: bool` field (BETA). Benchmark: ECE
improvement of 1.4% on the F6 multiclass DGP. Defaults to `False`
(neutral); existing callers unaffected. BETA: defaults may change
without a MAJOR bump."

**Deprecation alias contract**: when a key is promoted, the `extra`-dict
path keeps working indefinitely as an alias. The promotion itself is a
one-line registry edit, adding the `<extra-key>: <typed-field-name>`
mapping to `_PROMOTED_KEYS_BY_FAMILY` in
`src/seq_sklearn/config/_extras.py`. The `extract_deprecated_extras`
helper (in the same module; consumed by every family factory) then
detects the promoted key, emits a `DeprecationWarning`, routes the
supplied value onto the typed field, and raises `ConfigError` if the
caller set both the typed field and the `extra` key. Maintainers do not
write per-factory fallback code; routing through the single helper is
the contract.

The dict path never becomes a silent no-op: the supplied value reaches
the model via the typed field, and the behavior contract is preserved
indefinitely with the warning as the only change. If maintainers later
decide the alias is no longer worth carrying, removal requires a MAJOR
bump (and at least one MINOR cycle of `DeprecationWarning`) per the
project's semver policy.

**Promoted-field default constraint**: every key registered in
`_PROMOTED_KEYS_BY_FAMILY` must map to a typed field with an explicit
default. A field with no default (e.g. `LossConfig.strategy`) has
`FieldInfo.default == PydanticUndefined`, which makes the helper's
"both paths set" detection ambiguous. The
`test_extract_deprecated_extras_meta_promoted_keys_exist` meta-test
asserts both that the typed field exists and that its default is not
`PydanticUndefined` for every promoted key.

### Stage 3: STABLE (no further promotion needed)

After two MINOR releases at BETA without breaking changes, the field
becomes STABLE by default. The `docs/requirements.md` stability table
adds an entry. No further code changes; the promotion is documentary.

CHANGELOG entry: "`OptimizerConfig.amsgrad` graduates from BETA to
STABLE. Default value (`False`) is now stability-guaranteed; removal
requires a MAJOR bump."

### Promotion gate criteria

Each promotion requires:

1. **Benchmark evidence**: a documented improvement in the metric for
   the affected `target_kind` (see "Benchmark gate metrics" below) on
   the F6 synthetic DGP at the canonical seed triple plus at least one
   external dataset.
2. **Default-neutral landing**: the default value must preserve existing
   behavior so existing callers are unaffected.
3. **CHANGELOG rationale**: the entry names the benchmark, the
   improvement, and the field's tier.
4. **Optuna search-space update**: `suggest_params` adds the field to
   its sampling space. Per requirements F7, this is gated by the opt-in
   `search_advanced` / `search_extras` flags so the default search space
   stays small; the signature change is MINOR-bump-compatible.
5. **Test addition**: at least one unit test covers the promoted typed
   field plus one test exercises the deprecation alias.

### Benchmark gate metrics

Headline metrics per `target_kind` for promotion-gate purposes. These
are v1.0 starting thresholds, revisited once `docs/benchmarks.md`
populates with external-dataset numbers:

| `target_kind` | Headline metric | Improvement threshold |
|---|---|---|
| `binary` | accuracy on F6 DGP | >= 0.5 percentage points |
| `multiclass` | macro-F1 on F6 DGP | >= 0.5 percentage points |
| `regression_point` | R^2 on F6 DGP | >= 0.01 absolute |
| `regression_quantile` | empirical coverage gap on 80% interval | >= 0.01 absolute |
| (calibration) | ECE on calibration fold | >= 0.5 percentage points |

The metric choices match the N1 acceptance test families in
`docs/requirements.md`; the absolute pass thresholds in N1 are
release-gate thresholds, while the deltas here are promotion-gate
thresholds. Both are calibrated against the F6 DGP three-seed median.

"External dataset" means any dataset outside the synthetic DGP for which
the maintainers commit reproducible benchmark numbers to
`docs/benchmarks.md` (a v1.x deliverable; v1.0 ships the file as a
skeleton). Until that file populates, the F6 DGP three-seed-median
improvement is the sole gate.

## Stability-tier mapping

Hyperparameter tiers map to the project-wide stability tiers in
`docs/requirements.md` "Per-module stability tiers":

| Hyperparameter tier | Stability mapping | Removal cost |
|---|---|---|
| STABLE | STABLE in main config / family sub-config | MAJOR bump required |
| BETA | BETA in `<Model>AdvancedConfig` or family sub-config | MINOR bump + DeprecationWarning cycle |
| ALPHA | INTERNAL: `extra` tuple on any sub-config | No version bump; CHANGELOG-only |

A field's tier is surfaced in three places: the pydantic field
docstring carries the tier marker; the `docs/requirements.md`
per-module stability table lists BETA hyperparameters per model; and
the `docs/api/` mkdocstrings render groups BETA fields under a "BETA"
header (architecture A12).

## Extending the procedure to v2 / v3 models

The same partitioning applies to every future model: model-architecture
knobs that are documented in the model's paper and reference
implementation land STABLE on the main `<Model>Config` (users pick them
like `hidden_size`); experimental model-specific knobs land on
`<Model>AdvancedConfig`; family-of-options knobs inherit from the Tier 1
sub-configs.

Worked intent per model:

- **PatchTST**: `patch_length`, `patch_stride`, `channel_independent`
  STABLE on the main config; RevIN normalization and gated mixing
  weights reserved for `PatchTSTAdvancedConfig`.
- **TimesNet**: `n_periods_top_k`, `mask_aware_fft` STABLE; advanced
  reserved for FFT-detection threshold tuning.
- **TST**: `positional_encoding` (sinusoidal / learned) and
  `normalization` (layer / batch) STABLE; advanced reserved for
  ConvTran-style positional refinement.
- **LSTM / GRU**: `bidirectional`, `recurrent_dropout_kind` (per the A6.1
  recurrent skeleton), `bptt_window` STABLE; advanced reserved for
  AWD-LSTM-style weight-tying experiments.
- **LSTM-FCN**: `branch_width`, `fcn_kernel_sizes` STABLE; advanced
  reserved for branch-fusion variants.

## Alternatives considered

### Alternative A: keep the flat config, rely only on `extra` dicts

Cost: low (one `extra` field on `BaseModelConfig`). Risk: hyperparameters
that warrant a typed field stay in the dict forever, because there is no
documented promotion path. Discoverability craters; benchmark wins go
unshipped. Rejected: solves only Tier 4. The promotion ramp is the
actual contribution.

### Alternative B: keep the flat config, add `<Model>AdvancedConfig` only

Cost: medium (one new sub-config per model). Risk: family-of-options
scattering persists; adding AdamW `betas` and SGD `momentum` to the flat
config still produces a 60-field surface by v3. Rejected: half-fix.
Family-of-options nesting is the bigger source of scaling pain.

### Alternative C: full hierarchical config with no escape hatch

Cost: high (all of the tier-1/2/3 work, plus discipline to type every
new field). Risk: ALPHA-tier experimentation requires shipping a
release; benchmark testing slows down. Rejected: ergonomic regression
for the maintainer. The `extra` tuple adds a few lines per sub-config
and unlocks rapid experimentation.

### Alternative D: defer everything to v2

Cost: zero now; high later (full refactor of estimator constructors,
search spaces, save / load schema once v1 ships). Risk: v1 callers build
code against the flat config; v2 either breaks them or carries the flat
config forever. Rejected: kicks the can. Phase 1 is the lowest-cost
moment to do this because no callers exist yet.

## Open question

**Curated ALPHA-key enumeration**: `src/seq_sklearn/tuning/_alpha_keys.py`
is the maintainer-controlled list of `extra` keys eligible for
`search_extras=True` sampling. v1 ships it empty; the population
workflow (a maintainer adds keys here when they land an ALPHA
passthrough they want Optuna-tunable) is pinned at the Phase 8
implementation, not here.
