"""Lookback resolution (B4.1b).

A single `L_resolved` value is bound to four consumers per the
B4.1b contract: the split (deep-model window count), the deep
model's `tabular_config.lookback`, the featurizer (GBM lag depth),
and the raw-MTS reshape (TSC window length). The protocol fans out
to all four consumers from the same source, so a `resolve_lookback`
helper here is the canonical point of contact.

The default is `spec.lookback` from the dataset's `DatasetSpec`;
the `override` parameter is the experiment-driver hook (Phase B5+
will plumb a per-experiment override for sensitivity sweeps).
"""

from benchmarks.config import DatasetSpec


def resolve_lookback(spec: DatasetSpec, override: int | None = None) -> int:
    """Return the canonical `L_resolved` for a (dataset, experiment).

    Raises:
        ValueError: ``override`` is not positive.
    """
    if override is not None:
        if override < 1:
            raise ValueError(f"resolve_lookback: override must be >= 1; got {override}")
        return override
    return spec.lookback
