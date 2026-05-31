"""Synthetic panel datasets backed by SyntheticPanelGenerator.

Three deterministic variants for controlled-difficulty
benchmark runs that need no network, no auth, and no
on-disk archive:

- ``synthetic_binary_small``: binary classification, ~64 entities,
  monthly grain, mixed real + categorical features, moderate
  signal_strength.
- ``synthetic_multiclass_small``: 4-class problem, otherwise
  the same shape as the binary variant.
- ``synthetic_regression_small``: ``regression_point`` target
  with the same panel shape; lets the regression metric path
  exercise without C-MAPSS being on disk.

The integrity SHA-256 is derived from the DGP parameters so
the registry's required-field shape is satisfied. Loaders
ignore ``cache_root`` (there is no archive to read); the panel
is regenerated from the pinned seed on every load. The seed
+ dgp_version pin reproducibility; bumping either bumps the
SHA, which surfaces as a registry-invariants test break.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from benchmarks.config import DatasetSpec
from benchmarks.datasets._base import PanelDataset
from benchmarks.registry.datasets import register_dataset
from seq_sklearn.data.synthetic.generator import (
    SyntheticPanelGenerator,
)

_DGP_VERSION = 1
_DGP_SEED = 0x5717E71C5EEDB13  # "synthetic-seed-B13" mnemonic, deterministic

# Each variant pins (kind, n_entities, periods, signal_strength,
# noise_level, num_classes, class_balance) so the SHA is a hash of
# the generator's effective configuration. Changing any pinned knob
# bumps the SHA → registry-invariants test surfaces the drift.
# Mixed-feature variants: 3 real + 2 categorical channels each side
# (static + time-varying). The lag featurizer int-encodes
# categoricals via pandas category codes (sorted-unique order) so
# the resulting lag columns are int64 — accepted by LightGBM,
# XGBoost, and CatBoost without further casting. periods_per_entity
# floor is set above lookback + fold-count so the expanding-window
# CV's first fold always keeps at least one above-floor entity per
# dataset.
_VARIANTS: dict[str, dict[str, object]] = {
    "synthetic_binary_small": {
        "target_kind": "binary",
        "num_entities": 96,
        "periods_per_entity": (24, 48),
        "num_static_categorical": 2,
        "num_static_real": 3,
        "num_time_varying_real": 3,
        "num_time_varying_categorical": 2,
        "class_balance": 0.5,
        "signal_strength": 0.75,
        "noise_level": 0.1,
        "lookback": 12,
    },
    "synthetic_multiclass_small": {
        "target_kind": "multiclass",
        "num_entities": 96,
        "periods_per_entity": (24, 48),
        "num_static_categorical": 2,
        "num_static_real": 3,
        "num_time_varying_real": 3,
        "num_time_varying_categorical": 2,
        "num_classes": 4,
        "signal_strength": 0.75,
        "noise_level": 0.1,
        "lookback": 12,
    },
    "synthetic_regression_small": {
        "target_kind": "regression_point",
        "num_entities": 96,
        "periods_per_entity": (24, 48),
        "num_static_categorical": 2,
        "num_static_real": 3,
        "num_time_varying_real": 3,
        "num_time_varying_categorical": 2,
        "signal_strength": 0.75,
        "noise_level": 0.1,
        "target_noise": 0.1,
        "lookback": 12,
    },
}


def _params_sha(name: str, params: dict[str, object]) -> str:
    """SHA-256 of the variant's effective DGP parameters.

    Hashes the pinned dict plus ``_DGP_VERSION`` and ``_DGP_SEED``
    so the registry's required ``integrity_sha256`` field carries
    real meaning for synthetic datasets: any change to the DGP
    knobs flips the hash.
    """
    payload = {
        "name": name,
        "dgp_version": _DGP_VERSION,
        "seed": _DGP_SEED,
        **{k: v for k, v in params.items()},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _build_spec(name: str, params: dict[str, object]) -> DatasetSpec:
    kind = params["target_kind"]
    # Build a temporary generator to read the spec's effective
    # column names; cheap because no panel is materialized.
    gen = SyntheticPanelGenerator(**params)  # type: ignore[arg-type]
    feature_real_cols = tuple(gen.static_real_cols + gen.time_varying_real_cols)
    feature_categorical_cols = tuple(
        gen.static_categorical_cols + gen.time_varying_categorical_cols
    )
    sha = _params_sha(name, params)
    positive_label = 1 if kind == "binary" else None
    return DatasetSpec(
        name=name,
        task_type=kind,  # type: ignore[arg-type]
        access_tier="OPEN",
        size_tier="small",
        balance="balanced",
        modality="mixed",
        # `source_uri` is required; for synthetic the "source" is
        # the DGP module + seed. Uses a stable internal URI so the
        # registry-invariants test sees a well-formed string.
        source_uri=(
            f"synthetic://seq_sklearn.data.synthetic.SyntheticPanelGenerator"
            f"?dgp_version={_DGP_VERSION}&seed={_DGP_SEED}&variant={name}"
        ),
        integrity_sha256=sha,
        archive_basename=f"{name}.synthetic",
        entity_col=gen.id_col,
        time_col=gen.time_col,
        target_col="y",
        feature_real_cols=feature_real_cols,
        feature_categorical_cols=feature_categorical_cols,
        lookback=int(params.get("lookback", 12)),  # type: ignore[arg-type]
        observation_cutoff_rule=(
            "Synthetic DGP emits one labelled row per window end per "
            "entity (F6 contemporaneous). No external cutoff."
        ),
        densification_policy=None,
        positive_label=positive_label,
        excluded=False,
        citation=(
            "seq_sklearn SyntheticPanelGenerator (F6 DGP). "
            f"dgp_version={_DGP_VERSION}, seed={_DGP_SEED}."
        ),
    )


def _make_loader(name: str, params: dict[str, object]) -> Callable[[Path], PanelDataset]:
    def load(cache_root: Path) -> PanelDataset:  # noqa: ARG001
        """Generate the panel deterministically from the pinned seed.

        ``cache_root`` is part of the loader protocol but unused
        here: synthetic panels are not archived on disk; every
        call regenerates byte-identically from the seed.
        """
        gen = SyntheticPanelGenerator(**params)  # type: ignore[arg-type]
        panel, y = gen.generate(seed=_DGP_SEED)
        return PanelDataset(spec=_SPECS[name], panel=panel, y=y)

    return load


_SPECS: dict[str, DatasetSpec] = {
    name: _build_spec(name, params) for name, params in _VARIANTS.items()
}

for _name, _params in _VARIANTS.items():
    register_dataset(_SPECS[_name], _make_loader(_name, _params))
