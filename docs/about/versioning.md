# Versioning and deprecation policy

seq-sklearn follows [Semantic Versioning 2.0.0](https://semver.org/)
and [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).
The contract below tells you what is covered by SemVer and how
breaking changes are introduced.

## What "public API" means

The public surface is exactly:

- Symbols re-exported by `seq_sklearn` (the `__all__` list in
  `src/seq_sklearn/__init__.py`, defined by architecture A3).
- Methods documented in the [API reference](../reference/api).
- Module attributes reached without a leading underscore in the
  import path (e.g. `seq_sklearn.tuning.suggest_params` is public;
  `seq_sklearn.models._base.BaseSequenceEstimator` is not).

Anything reachable only through an underscore-prefixed module or
attribute is **INTERNAL** and not covered by the stability
guarantee. We reserve the right to change INTERNAL surfaces in
MINOR releases.

## Stability tiers

| Tier | Examples | Stability |
|---|---|---|
| **STABLE** | `TFTClassifier.fit/predict/predict_proba`, `TabularToSequence`, `EntityTimeSeriesSplit`, `HardwareTier`, `detect` | Breaking change requires MAJOR |
| **BETA** | `export_onnx`, `predict_with_attention`, `AttentionOutput`/`RegressionAttentionOutput` | Fields may be added in MINOR (never removed); use attribute access |
| **ALPHA** | `suggest_params` default search space | Defaults may change without MINOR bump; pass an explicit search space for stable behavior |
| **INTERNAL** | `seq_sklearn._*` modules | Not part of the public API; change at any time |

The full per-symbol tier table is in `docs/requirements.md`
"Per-module stability tiers (v1)".

## SemVer cadence

- **MAJOR** bump on any breaking change to a STABLE public symbol.
- **MINOR** bump on a backwards-compatible addition (new public
  symbol, new BETA field, new default that changes behavior — the
  CHANGELOG entry calls it out explicitly).
- **PATCH** bump on a bug fix that does not change behavior beyond
  fixing the bug.

## Deprecation policy

When we change a STABLE symbol, we follow this process:

1. **Introduce the replacement in a MINOR release.** The old symbol
   continues to work and emits a `DeprecationWarning` pointing at
   the replacement. CHANGELOG entry under `Deprecated`.
2. **Wait at least one MINOR release.** Users who track minor
   releases have time to update.
3. **Remove the old symbol in the next MAJOR release.** CHANGELOG
   entry under `Removed`.

A deprecation `DeprecationWarning` includes:

- The old symbol's name.
- The version it was deprecated in.
- The version it will be removed in (if known).
- A short pointer to the replacement.

Example shape:

```python
import warnings
warnings.warn(
    "TFTClassifier.fit_predict() is deprecated since 1.1.0 and will be "
    "removed in 2.0.0; call fit() then predict() instead.",
    DeprecationWarning,
    stacklevel=2,
)
```

## How to track versions

- **GitHub releases** carry the tagged version and a one-screen
  summary of the changes.
- **The CHANGELOG** is the authoritative per-release diff. The
  `[Unreleased]` section at the top accumulates pending changes.
- **`importlib.metadata.version("seq-sklearn")`** is the
  single-source-of-truth for the installed version. The library's
  `__version__` reads from there; don't hardcode strings.

## Migration

When we introduce a breaking change, the
[migration template](migration_template) is filled in for that
release and shipped under `docs/about/migration_1.x_to_2.x.md`. The
template groups changes by surface area (config, estimator API,
serialization, etc.) and gives before/after for every renamed
symbol.

In v1.0.0 there is nothing to migrate yet (it is the first stable
release). The template ships now so the form is ready when the
first breaking change lands.

```{testcode}
import seq_sklearn

assert isinstance(seq_sklearn.__version__, str)
assert seq_sklearn.__version__ != ""
```
