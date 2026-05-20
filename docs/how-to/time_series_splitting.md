# Time-series splitting (and avoiding temporal leakage)

The single most common mistake in time-series ML is splitting by row
or by entity at random. Both leak future information. seq-sklearn
ships `EntityTimeSeriesSplit` (architecture A9.1) as the
sklearn-compatible splitter that does it correctly.

## Why random splits leak

Consider a panel with 10 entities and 12 monthly periods each (120
rows). A random 80/20 split puts 96 rows in train, 24 in test, chosen
uniformly. For nearly every entity, that means some periods of that
entity's history are in train and some in test. At inference you are
asking the model to predict month 9 for entity A, having already
shown it months 1-8 AND 10-12 of A during training. That's leakage:
the model has seen the future of the entity it is now scoring.

Entity-disjoint splits (some entities in train, others in test) avoid
the within-entity leak but introduce another problem: every test
entity is a *cold start*, which is rarely what production looks like.

## What `EntityTimeSeriesSplit` does

For each fold `i`, the splitter constructs an expanding window
**per entity**: for every entity, the test segment is the most-recent
chunk and the train segment is everything before it (plus the
preceding `lookback - 1` rows of history as context). Every entity
appears in every fold. There is no entity-level cold start, and no
test-target row is also a train-target row (the only overlap is
history-context, which is required because the model needs a window
of length `lookback` to make a prediction).

## Wrong vs Right

**Wrong**, the silent-leak version:

```{code-block} python
# Don't do this on a multi-entity time-series panel.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    panel, y, test_size=0.2, random_state=42
)
```

`train_test_split` shuffles rows; train and test will contain
overlapping periods of the same entities. The accuracy you report is
inflated.

**Wrong**, the cold-start version:

```{code-block} python
# Also not what production looks like for an established entity base.
import numpy as np
rng = np.random.default_rng(0)
entities = panel["entity_id"].unique()
rng.shuffle(entities)
train_entities = entities[: int(0.8 * len(entities))]
mask_train = panel["entity_id"].isin(train_entities)
```

Every test entity is a cold start. Fine if production is a steady
stream of new entities; misleading if production scores existing
entities forward in time.

**Right**, the entity-time-expanding-window splitter:

```{code-block} python
from seq_sklearn import EntityTimeSeriesSplit, TFTClassifier

splitter = EntityTimeSeriesSplit(
    n_splits=5,
    lookback=12,             # match the model's lookback
    id_col="entity_id",
    time_col="period",
)

for fold, (train_idx, test_idx) in enumerate(splitter.split(panel)):
    clf = TFTClassifier(lookback=12, hidden_size=64, max_epochs=10)
    clf.fit(panel.iloc[train_idx], y[train_idx])
    score = clf.score(panel.iloc[test_idx], y[test_idx])
    print(f"fold {fold}: {score:.3f}")
```

The splitter returns row indices (sklearn convention), so it composes
with `cross_val_score`, `GridSearchCV`, and Optuna's
`OptunaSearchCV` unchanged.

## A note on calibration folds

Inside `fit()`, the library uses a separate `compute_three_way_split`
(architecture A5) for train/val/cal *within* a fold. That is the
estimator's internal concern; it does not affect the external CV
loop. `EntityTimeSeriesSplit` is what you hand to sklearn; the
calibration partition is automatic and time-ordered (so the same
F2 multi-entity-random-split leak cannot return through the val
split).

## A common mistake

Re-binding the lookback on the splitter and the estimator
independently. The splitter's `lookback` MUST equal the estimator's
`tabular_config__lookback`; mismatch silently changes what counts
as "history-only context" in the fold. Resolve `lookback` once per
run and pass the same value to both.
