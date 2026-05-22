"""Per-family HPO-space registry (Phase B8).

The HPO uplift driver pairs every (dataset, model) cell with two
arms: the `default` arm (reusing the B5 manifest at variant=
"default") and the `tuned` arm. The tuned arm sources its
hyperparameter samples from a per-family search-space module
registered here. The protocol is intentionally narrow:

- The `HPOSpace.sample(trial, model_spec, dataset_spec)` callable
  takes one Optuna trial plus the registry's model / dataset specs
  and returns the kwargs dict the adapter's
  `hyperparameters=` channel consumes.
- The `search_space_size` integer reports the number of searchable
  hyperparameters (B6.4.0 parity disclosure: reported on every
  tuned row and rendered in the report next to the Δ).

GBM and TSC search spaces register from their own modules when those
adapter families ship (per `benchmarks/adapters/__init__.py`'s B2-
followup note); the registry is the single seam.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import optuna
from pydantic import BaseModel, ConfigDict, Field

from benchmarks.config import DatasetSpec, ModelFamily, ModelSpec

__all__ = [
    "HPO_REGISTRY",
    "HPOFamilyNotRegisteredError",
    "HPORegistration",
    "HPOSpace",
    "HPOSpaceSampler",
    "get_hpo_space",
    "register_hpo_space",
]


HPOSpaceSampler = Callable[
    [optuna.trial.BaseTrial, ModelSpec, DatasetSpec],
    dict[str, Any],
]


class HPOFamilyNotRegisteredError(KeyError):
    """Raised when a `ModelFamily` has no `HPOSpace` registered.

    The HPO uplift driver SKIPS cells whose model family has no
    search space registered (writes a typed `skipped_reason` so the
    report can footnote them); the typed error is for callers who
    expect the lookup to succeed.
    """


class HPOSpace(BaseModel):
    """Per-family search-space metadata record.

    Pydantic-frozen so the metadata value is immutable. The
    `sampler` callable lives next to it in `HPORegistration`
    (pydantic disallows callable fields without a custom validator);
    the two move together through `register_hpo_space` and
    `get_hpo_space`, which is the public surface the harness uses.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: ModelFamily
    search_space_size: int = Field(ge=0)
    description: str  # human-readable summary rendered in the report


@dataclass(frozen=True)
class HPORegistration:
    """One family's registered (space, sampler) pair (B9 arch-I5).

    Collapses the prior dual-dict registry (`HPO_REGISTRY` +
    `_HPO_SAMPLERS`) into a single record so a future caller cannot
    drift the two halves apart.
    """

    space: HPOSpace
    sampler: HPOSpaceSampler


# Single source of truth: family -> (space, sampler).
HPO_REGISTRY: dict[ModelFamily, HPORegistration] = {}


def register_hpo_space(space: HPOSpace, sampler: HPOSpaceSampler) -> None:
    """Register a per-family search space.

    Idempotent for re-registration of an identical record + sampler
    (re-import of an HPO module under a test fixture is harmless);
    a conflicting re-registration raises `ValueError` so a typo in
    a downstream module is loud.

    Raises:
        ValueError: a different `HPOSpace` or `sampler` is already
            registered for `space.family`.
    """
    existing = HPO_REGISTRY.get(space.family)
    if existing is not None:
        if existing.space != space or existing.sampler is not sampler:
            raise ValueError(
                f"register_hpo_space: family {space.family!r} already "
                f"registered with a different HPOSpace or sampler"
            )
        return
    HPO_REGISTRY[space.family] = HPORegistration(space=space, sampler=sampler)


def get_hpo_space(family: ModelFamily) -> tuple[HPOSpace, HPOSpaceSampler]:
    """Look up the registered `(HPOSpace, sampler)` for a family.

    Raises:
        HPOFamilyNotRegisteredError: no HPO module has registered
            this family yet.
    """
    registration = HPO_REGISTRY.get(family)
    if registration is None:
        raise HPOFamilyNotRegisteredError(
            f"get_hpo_space: family {family!r} has no HPO search space "
            f"registered; registered families: {sorted(HPO_REGISTRY)}"
        )
    return registration.space, registration.sampler
