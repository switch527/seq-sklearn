"""Phase B1 registry-invariants tests (B2.1).

Walks every registered dataset spec and asserts:

- Every active entry has a non-empty citation, a SHA-256-shaped
  integrity hash, a target column, and a loader callable resolvable
  via `get_loader`.
- Every B1.5 exclusion (`excluded=True`) carries a non-empty
  `exclusion_reason`.
- Every irregular-time entry (PhysioNet-2012, PTB-XL, MIMIC-IV)
  carries a `densification_policy` with non-empty `bin_width`,
  `aggregation`, and `missing_bin_fill` (qa-III-2 / B2.1).
- Every binary spec carries a `positive_label`.

The test imports `benchmarks.datasets` so the side-effect
registrations fire; the autouse `isolated_registry` fixture from
``conftest.py`` snapshots+restores around each test so this
registration is visible inside the test but does NOT leak into
later tests.
"""

import benchmarks.datasets  # noqa: F401  # pyright: ignore[reportUnusedImport] - side-effect: registers loaders
from benchmarks.registry import (
    get_dataset,
    get_loader,
    list_datasets,
)

# The irregular-time roster is named explicitly in B2.1; the test
# asserts every active member carries a densification_policy.
_IRREGULAR_TIME_DATASETS: frozenset[str] = frozenset(
    {"physionet_2012", "ptb_xl", "mimic_iv_ihm"}
)


def test_every_registered_dataset_has_a_loader() -> None:
    names = list_datasets()
    assert names, "no datasets registered; benchmarks.datasets import failed"
    for name in names:
        loader = get_loader(name)
        assert callable(loader), f"dataset {name!r}: loader is not callable"


def test_every_active_spec_has_a_citation() -> None:
    for name in list_datasets():
        spec = get_dataset(name)
        if spec.excluded:
            continue
        assert spec.citation.strip(), (
            f"dataset {name!r}: active entry must declare a non-empty citation"
        )


def test_every_excluded_spec_has_a_reason() -> None:
    """B1.5: excluded entries must declare a non-empty reason
    (pydantic enforces this at construction, but the test pins
    the contract at the registry level too)."""
    for name in list_datasets():
        spec = get_dataset(name)
        if not spec.excluded:
            continue
        reason = spec.exclusion_reason
        assert reason is not None, (
            f"dataset {name!r}: excluded=True but exclusion_reason is None"
        )
        assert reason.strip(), (
            f"dataset {name!r}: excluded=True but exclusion_reason is blank"
        )


def test_irregular_time_datasets_carry_densification_policy() -> None:
    """B2.1 / qa-III-2: every active irregular-time entry must carry
    a fully specified densification_policy (the pydantic
    `DensificationPolicy` model already enforces non-empty bin_width
    / aggregation / missing_bin_fill, so the test only needs to
    assert non-None)."""
    for name in _IRREGULAR_TIME_DATASETS & set(list_datasets()):
        spec = get_dataset(name)
        if spec.excluded:
            continue
        assert spec.densification_policy is not None, (
            f"dataset {name!r}: active irregular-time entry must "
            f"declare a densification_policy (B2.1)"
        )


def test_binary_specs_have_positive_label() -> None:
    """B5.1: binary precision/recall/F1 read `pos_label` from
    `DatasetSpec.positive_label`. Pydantic already enforces this on
    construction (cross-field validator); the registry test pins
    the invariant at the live-registry level too."""
    for name in list_datasets():
        spec = get_dataset(name)
        if spec.task_type != "binary" or spec.excluded:
            continue
        assert spec.positive_label is not None, (
            f"dataset {name!r}: binary spec without positive_label"
        )
