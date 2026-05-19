# Migration template

This is the empty template that gets filled in for each MAJOR
release with a real breaking change. v1.0.0 is the first stable
release, so there is nothing to migrate yet.

The template structure below ships now so the form is ready when
the first breaking change lands (likely v2.0.0 with PatchTST /
TimesNet / TST family completion).

## Filename convention

The filled-in migration guide for a release goes at
`docs/about/migration_{old}_to_{new}.md`, e.g.
`migration_1.x_to_2.x.md`.

## Template

```{code-block} markdown
# Migration: {old version} → {new version}

## Summary

One-paragraph description of the major release's theme and the
shape of the changes (additive vs replacing vs removing a major
surface).

## Quick fix-or-defer table

| Change | Surface | Action | Codemod-supported |
|---|---|---|---|
| `OldClass` → `NewClass` | estimator | rename | yes |
| `param_name` → `new_param_name` | config | rename | yes |
| `removed_method()` | estimator | replace with `new_method()` | no |
| ... | ... | ... | ... |

## Changes by surface area

### Estimator API

- **`OldClass` → `NewClass`.**
  - Before: ```python
    clf = OldClass(...)
    clf.fit(...)
    ```
  - After: ```python
    clf = NewClass(...)
    clf.fit(...)
    ```
  - Reason: ...
  - Deprecation window: introduced in {minor}, removed in {major}.

### Configuration

- **`TFTConfig.param_x` → `TFTConfig.new_param_x`.**
  - Before / after, reason, window.

### Serialization

- **Saved-model format change (if any).**
  - Old format readable on `{new version}` for one MAJOR cycle.
  - `seq_sklearn migrate /path/to/old/model /path/to/new/model`
    converts saved models in place.

### Public surface (`__all__`)

- **Added:** {list}
- **Removed:** {list, each with the replacement}
- **Renamed:** {list, each with the codemod}

## Codemod (if shipped)

```bash
pipx run bump-seq-sklearn /path/to/your/code
```

What it rewrites:

- Class renames.
- Keyword-argument renames.
- Import-path moves.

What it does NOT rewrite:

- Behavioral changes (you have to read the corresponding section
  and decide whether the new behavior is what you want).
- Comments and docstrings that mention the old names.

## Deprecation warnings you should see

Every removed symbol emitted a `DeprecationWarning` in the
preceding MINOR release. If you ran with
`-W "error::DeprecationWarning"`, your tests already failed; that
is the intended early-warning path.

## Old version still available

For one MAJOR cycle, the old surface is importable under
`seq_sklearn.v{N-1}`:

```python
from seq_sklearn.v1 import TFTClassifier as OldTFTClassifier
```

Use this for gradual migration; the new public surface is what we
ship support for.
```
