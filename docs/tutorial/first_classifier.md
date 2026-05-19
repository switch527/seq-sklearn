# Train your first TFT classifier

This is the linear, finishable spine: a working binary classifier on
a bundled synthetic panel, evaluated with sklearn's accuracy score,
ending with the variable-selection weights the model used. Every line
below is the literal content of `examples/quickstart.py`, which the
N1 quickstart-in-CI test imports and the README mirrors — there is
ONE executable source, never two.

## Setup

You need `seq-sklearn` installed and `scikit-learn`. The example uses
the bundled `SyntheticPanelGenerator` to produce a small balanced
panel and the public `TFTClassifier`. Hardware is CPU; on a GPU box
the same code uses CUDA automatically.

```bash
pip install seq-sklearn
```

## The full example

```{literalinclude} ../../examples/quickstart.py
:language: python
:linenos:
```

## What the pieces are

- **`SyntheticPanelGenerator`**: a deterministic data-generating
  process that emits a tidy panel (one row per entity per period)
  with static and time-varying covariates and a binary label. Real
  data slots in here, the rest of the example is unchanged.
- **`TabularToSequenceConfig`**: the single declarative spec that
  tells the library which columns are the `(entity, time)` key, which
  are static vs time-varying, and which are real vs categorical. The
  same config object is reused for predict.
- **`TFTClassifier(...)`**: a regular sklearn estimator with
  `fit`/`predict`/`predict_proba`/`get_params`. Hyperparameters are
  ordinary keyword arguments (`hidden_size`, `attention_heads`,
  `max_epochs`, `batch_size`); the architecture pieces (variable
  selection, attention, gated residuals) come from Lim et al. 2021.
- **`run_quickstart` returning accuracy**: the function is what the
  e2e test imports and asserts on, which is also why the README and
  the docs site can mirror its body — the executable form is the
  single source.

## What to do next

- The [how-to guides](../how-to/index) answer specific operational
  questions ("which fields matter for tuning", "how do I extract
  attention", "how do I export to ONNX", "how do I split a
  multi-entity panel without leaking").
- The [reference](../reference/index) is the auto-generated API and
  config surface plus the glossary of contract terms.
- The [explanation pages](../explanation/index) cover the design
  decisions: why a TFT for classification, how the interpretability
  surfaces work, the determinism model.
