# Quickstart

```{warning}
This is the target v1 API. The import does not work until the first
release ships. It is documented now so the design is legible before
code-complete.
```

## The data shape

seq-sklearn consumes a tabular panel: one row per entity per period,
with a key of `(entity_id, period)`. The same shape works for customers
by month, patients by visit, devices by day, or sensors by hour. The
model never learns what an entity is; it sees one sequence per entity.

## Train a classifier

```python
from seq_sklearn import TFTClassifier
from sklearn.metrics import roc_auc_score

clf = TFTClassifier(lookback=12, hidden_size=128)
clf.fit(X_train, y_train)                    # X: tidy entity-by-period DataFrame
proba = clf.predict_proba(X_test)
print(f"AUC: {roc_auc_score(y_test, proba[:, 1]):.3f}")
```

## Compose into the sklearn ecosystem

Every estimator implements the sklearn contract, so it drops into
`Pipeline`, `GridSearchCV`, `cross_val_score`, and Optuna search with no
adapter:

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([("clf", TFTClassifier(lookback=12))])
pipe.fit(X_train, y_train)
```

## Read the interpretability surfaces

Variable selection and temporal attention are returned outputs, not a
separate analysis step:

```python
out = clf.predict_with_attention(X_test)     # frozen dataclass
out.variable_selection_weights               # which features mattered, per step
out.temporal_attention                       # which timesteps mattered
```

A worked end-to-end churn example, with the rendered attention figure,
lands with the v1 release.
