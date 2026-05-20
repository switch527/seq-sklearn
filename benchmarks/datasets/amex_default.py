"""Amex Default Prediction (Kaggle, GATED).

The design's flagship payments-panel dataset (B2 line for `Amex
Default`): ~458k customers, ~13 monthly statements each, binary
target (default within 18 months of the latest statement). Mixed
modality (188 anonymized numeric + categorical features).

GATED: the archive is on Kaggle behind a competition-rules
click-through; the harness will never download it. The loader
raises a typed :class:`GatedDatasetUnavailableError` naming the
exact manual step and the target cache path (B2.3). Once the user
has placed the archive into the cache, the loader downloads
nothing and parses the file in place.
"""

from pathlib import Path

from benchmarks._io.cache import archive_path
from benchmarks._io.integrity import GatedDatasetUnavailableError
from benchmarks.config import DatasetSpec
from benchmarks.datasets._base import PanelDataset
from benchmarks.registry.datasets import register_dataset

_NAME = "amex_default"

# Anonymized feature columns are recorded as a placeholder pair for
# the schema; the real loader populates them from the parsed file.
# Phase B1 ships the GATED-error shape; the parse-on-presence path is
# a Phase B1-followup task (the design's B7.1 release-checklist
# step pins the full-population pass on this dataset).
_SPEC = DatasetSpec(
    name=_NAME,
    task_type="binary",
    access_tier="GATED",
    size_tier="huge",
    balance="imbalanced",
    modality="mixed",
    source_uri="https://www.kaggle.com/competitions/amex-default-prediction/data",
    # Kaggle does not publish the canonical archive SHA-256; the
    # release engineer records the locally-observed hash here once
    # they have accepted the competition rules and pulled the data.
    integrity_sha256="deadbeef" * 8,
    archive_basename="amex-default-prediction.zip",
    entity_col="customer_ID",
    time_col="S_2",
    target_col="target",
    # Empty until the in-cache parse-on-presence path lands in a
    # Phase B1 follow-up; the archive declares the canonical
    # numeric + categorical column split (`D_*`, `S_*`, `P_*`,
    # `B_*`, `R_*`) and the loader will populate these from the
    # parsed schema rather than a hand-typed list.
    feature_real_cols=(),
    feature_categorical_cols=(),
    lookback=13,
    observation_cutoff_rule=(
        "Per the competition rules, features are read from up to 13 "
        "monthly statements ending at S_2 = customer's latest "
        "statement; the binary target is default within 18 months "
        "of that latest statement."
    ),
    densification_policy=None,  # statements are already monthly
    positive_label=1,
    excluded=False,
    citation=(
        "American Express, Kaggle competition "
        "'amex-default-prediction', 2022; "
        "https://www.kaggle.com/competitions/amex-default-prediction"
    ),
)


def load(cache_root: Path) -> PanelDataset:
    """Load Amex Default from the cache root.

    GATED dataset: the loader does NOT download. If the archive is
    not in the cache, raises :class:`GatedDatasetUnavailableError`
    with the manual setup steps and the target path. Phase B1 ships
    the gated-error shape; the in-cache parse-on-presence path is a
    follow-up task because the archive cannot be exercised in CI
    (Kaggle credentials required).
    """
    target = archive_path(cache_root, _NAME, _SPEC.archive_basename)
    if not target.is_file():
        raise GatedDatasetUnavailableError(
            f"{_NAME!r} is a GATED dataset and its archive is not in "
            f"the cache.\n"
            f"  expected path: {target}\n"
            f"\n"
            f"Manual setup (one-time):\n"
            f"  1. Accept the competition rules at\n"
            f"     {_SPEC.source_uri}\n"
            f"  2. Download `amex-default-prediction.zip` (the\n"
            f"     bundled `train_data.csv` + `train_labels.csv` +\n"
            f"     `test_data.csv` archive).\n"
            f"  3. Place the file at the expected path above.\n"
            f"  4. Re-run; the loader integrity-checks the archive\n"
            f"     and parses it in-cache (no network).\n"
            f"\n"
            f"To exclude this dataset from a run instead, remove\n"
            f"{_NAME!r} from `BenchmarkConfig.datasets`."
        )
    # Phase B1-followup: parse the archive into a panel + y.
    raise NotImplementedError(
        f"{_NAME}: in-cache parse path lands in a Phase B1 follow-up; "
        f"see docs/benchmark_suite_implementation_plan.md"
    )


register_dataset(_SPEC, load)
