# Observability

seq-sklearn emits a stream of structured log events via the
`seq_sklearn` Python logger (F11). The event names + payload
schemas below are the v1 contract; consumers (MLflow, W&B, custom
sinks) bind to these names.

## Event schema

Every event has:

- a `name` (dotted path, stable across releases),
- a `payload` (a flat dict of typed fields),
- a timestamp (added by the logger).

Events are emitted at `INFO` level by default. Override with the
standard `logging` machinery:

```python
import logging
logging.getLogger("seq_sklearn").setLevel(logging.DEBUG)
```

## v1 event catalog

### Fit lifecycle

| Event name | Payload fields | Emitted at |
|---|---|---|
| `fit.start` | `seed: int`, `task_type: str`, `panel_rows: int`, `n_entities: int`, `n_classes: int \| None` | `fit()` entry |
| `fit.split` | `train_rows: int`, `val_rows: int`, `cal_rows: int` | After `compute_three_way_split` |
| `fit.epoch.start` | `epoch: int`, `lr: float` | Each epoch start |
| `fit.epoch.end` | `epoch: int`, `train_loss: float`, `val_loss: float`, `val_metric: float`, `wall_seconds: float` | Each epoch end |
| `fit.early_stop` | `epoch: int`, `reason: str` | When patience exhausts |
| `fit.calibrate` | `strategy: str`, `param: float` | When `cal_fraction > 0` |
| `fit.end` | `epochs: int`, `best_val_metric: float`, `wall_seconds: float` | `fit()` return |

### Predict / inference

| Event name | Payload fields | Emitted at |
|---|---|---|
| `predict.start` | `panel_rows: int`, `below_floor_rows: int` | `predict()` entry |
| `predict.end` | `wall_seconds: float` | `predict()` return |

### Optuna integration

| Event name | Payload fields | Emitted at |
|---|---|---|
| `optuna.trial.start` | `trial_number: int`, `params: dict` | Each trial start |
| `optuna.trial.prune` | `trial_number: int`, `epoch: int`, `intermediate_value: float` | Pruning hook fired |
| `optuna.trial.end` | `trial_number: int`, `value: float`, `state: str` | Each trial end |

### ONNX export

| Event name | Payload fields | Emitted at |
|---|---|---|
| `export.onnx.start` | `path: str`, `batch: int` | `export_onnx()` entry |
| `export.onnx.end` | `path: str`, `bytes: int`, `wall_seconds: float` | `export_onnx()` return |

## Wiring to MLflow / W&B

The two reference examples in the gallery
([examples gallery](../examples_gallery/index)) show how to bind
`fit.epoch.end` events to an experiment tracker via a logging
handler. The library does NOT auto-wire to any tracker; you opt in
by registering a handler on the `seq_sklearn` logger.

## Stability

Event names are STABLE (breaking change requires a MAJOR bump).
Payload fields are additive: MINOR releases may add fields but
never remove or rename. Consumers should treat unknown payload
fields as opaque, not crash.

```{testcode}
import logging

logger = logging.getLogger("seq_sklearn")
assert logger.name == "seq_sklearn"
```
