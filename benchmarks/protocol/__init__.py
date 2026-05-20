"""Fair-comparison protocol (B4).

Phase B3 lands four leaves consumed by every adapter family:

- :func:`benchmarks.protocol.lookback.resolve_lookback`: the single
  `L_resolved` source bound to the splitter, the deep model's
  `tabular_config.lookback`, the GBM/passthrough featurizer, and
  (Phase B3-followup) the raw-MTS reshape for TSC adapters.
- :func:`benchmarks.protocol.split.make_splitter`: per-entity
  expanding-window CV that respects entity grouping (`A9.1`).
- :func:`benchmarks.protocol.featurize.lag_featurize`: panel ->
  flat tabular X for GBM / sklearn-passthrough adapters.
- :func:`benchmarks.protocol.fingerprint.fingerprint_folds`:
  SHA-256 of the canonical fold layout (B8.1).

The raw-MTS reshape (B4.4) for the classical-TSC family is deferred
to a Phase B3-followup branch alongside the aeon-blocked TSC adapter.
"""

from benchmarks.protocol.featurize import lag_featurize
from benchmarks.protocol.fingerprint import fingerprint_folds
from benchmarks.protocol.lookback import resolve_lookback
from benchmarks.protocol.split import iter_folds, make_splitter

__all__ = [
    "fingerprint_folds",
    "iter_folds",
    "lag_featurize",
    "make_splitter",
    "resolve_lookback",
]
