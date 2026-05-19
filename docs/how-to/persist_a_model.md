# Persist and reload a fitted model

seq-sklearn ships `save`/`load` on every estimator. The on-disk
format is **safetensors for weights + JSON for config and fitted
state**, NOT pickle.

## The recommended path: `save` and `load`

```python
from seq_sklearn import TFTClassifier

clf = TFTClassifier(...).fit(panel_train, y_train)
clf.save("./my_model")    # creates a directory

# Later, in the same or a different process:
loaded = TFTClassifier.load("./my_model")
preds = loaded.predict(panel_test)
```

The saved directory contains:

- `weights.safetensors` — the model parameters, in safetensors
  format (zero-copy, language-agnostic, no pickle).
- `config.json` — the pydantic config dump, plus the fitted-state
  attributes (classes seen, schema, fitted scalers/encoders state).

`load` rebuilds the estimator from the JSON config, then loads the
weights into the rebuilt model.

## Why not pickle

```{warning}
**Never load untrusted pickles.** A pickle file is arbitrary Python
code; loading one runs that code. Bad actors regularly use pickle
files to ship malware. The safetensors + JSON format the library
ships is data-only — there is no executable payload.
```

`joblib.dump`/`joblib.load` (sklearn's traditional persistence) is
pickle under the hood and carries the same security and
version-coupling problems. Avoid it for production artifacts.

## Version pinning

When you load a model trained on version X with version Y, you
inherit Y's behavior. The library's public API is SemVer-stable, but
the binary representation of weights is tied to the architecture
modules.

The safe production pattern:

- **Pin `seq-sklearn`, `torch`, and `numpy` in your serving
  environment** to the exact versions you trained with.
- **Containerize the serving environment.** A Docker image with the
  pinned versions and the saved-model directory is the most reliable
  reload guarantee.
- A future minor release may emit a `MigrationsWarning` when loading
  a model from an older library version; for now, treat saved models
  as bound to their training version.

## Decision tree

- **Same-version, same-Python serving:** `clf.save` / `load`.
- **Cross-language / cross-runtime serving:** ONNX, see
  [export to ONNX](export_onnx.md).
- **Long-term archive of fitted state:** save the directory *and*
  the `(seq-sklearn, torch, numpy)` version triple in a
  metadata file. Containerize for guaranteed reload.

## What about `cloudpickle` / `joblib`?

They work for in-session round-trips but inherit pickle's security
and version-coupling problems. Prefer `save`/`load` for any model
that crosses a session boundary, even within the same project.

```{testcode}
from seq_sklearn import TFTClassifier

assert hasattr(TFTClassifier, "save")
assert hasattr(TFTClassifier, "load")
```
