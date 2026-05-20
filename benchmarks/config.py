"""Pydantic v2 config schema for the benchmark harness.

`BenchmarkConfig` is the single config object the CLI loads from a
TOML file. It composes ``DatasetSpec`` (per-dataset registration
record, B2.2 of the design), ``ModelSpec`` (per-model registration
record, B3.2), and ``ExperimentSpec`` (the four B6 experiment kinds).

The field set is the scaffold form: enough structure to validate a
minimal config and register stubs, with hooks for the per-phase
extensions (B1 datasets, B2 models, B3-B8 experiments) to grow into.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Single source of truth for the supported task types; mirrors the
# library's own `Literal` so a registry entry cannot declare a task
# type the library doesn't accept.
TaskType = Literal[
    "binary",
    "multiclass",
    "regression_point",
    "regression_quantile",
]

AccessTier = Literal["OPEN", "GATED"]
SizeTier = Literal["small", "medium", "large", "huge"]
Balance = Literal["balanced", "imbalanced"]
Modality = Literal["numeric", "mixed"]

ModelFamily = Literal["seq_sklearn", "gbm", "tsc", "sklearn_passthrough"]

ExperimentKind = Literal[
    "raw_loss",
    "ensemble",
    "training_time",
    "hpo_uplift",
]


class DensificationPolicy(BaseModel):
    """Per-dataset irregular-time densification rule (B2.2).

    For datasets whose observations are irregular in wall-clock time
    (PhysioNet-2012, PTB-XL, MIMIC-IV), the loader, not the library,
    owns resampling to a fixed period grid. Every field is required
    for an active irregular-time entry; B2.1's registry-invariants
    test asserts this.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    bin_width: str = Field(min_length=1)  # e.g. "1h", "10s"
    aggregation: Literal["mean", "last", "max", "min", "sum", "first"]
    missing_bin_fill: Literal["forward", "zero", "nan", "drop"]


class DatasetSpec(BaseModel):
    """Per-dataset registry record (B2.2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    task_type: TaskType
    access_tier: AccessTier
    size_tier: SizeTier
    balance: Balance
    modality: Modality
    source_uri: str = Field(min_length=1)
    integrity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_basename: str = Field(min_length=1)
    entity_col: str = Field(min_length=1)
    time_col: str = Field(min_length=1)
    target_col: str = Field(min_length=1)
    feature_real_cols: tuple[str, ...] = ()
    feature_categorical_cols: tuple[str, ...] = ()
    lookback: int = Field(ge=1)
    observation_cutoff_rule: str | None = None
    densification_policy: DensificationPolicy | None = None
    positive_label: int | str | None = None
    excluded: bool = False
    exclusion_reason: str | None = None
    citation: str = Field(min_length=1)

    @model_validator(mode="after")
    def _positive_label_only_on_binary(self) -> "DatasetSpec":
        """B2.2: `positive_label` is required on binary specs and
        absent on multiclass and regression specs."""
        if self.task_type == "binary":
            if self.positive_label is None:
                raise ValueError(
                    f"DatasetSpec({self.name!r}): task_type='binary' "
                    f"requires `positive_label` to be set (the class "
                    f"the imbalanced framing treats as positive); "
                    f"see B2.2 / B5.1 of the benchmark design"
                )
        elif self.positive_label is not None:
            raise ValueError(
                f"DatasetSpec({self.name!r}): task_type={self.task_type!r} "
                f"must not declare `positive_label`; it is binary-only"
            )
        return self

    @model_validator(mode="after")
    def _excluded_carries_reason(self) -> "DatasetSpec":
        """B1.5: B1.5 negatives must declare a non-empty
        `exclusion_reason`; active specs must NOT declare one."""
        if self.excluded:
            if not self.exclusion_reason:
                raise ValueError(
                    f"DatasetSpec({self.name!r}): excluded=True requires "
                    f"a non-empty `exclusion_reason` (B1.5)"
                )
        elif self.exclusion_reason is not None:
            raise ValueError(
                f"DatasetSpec({self.name!r}): exclusion_reason is set "
                f"but excluded=False; either flip `excluded` or clear "
                f"the reason"
            )
        return self

    @model_validator(mode="after")
    def _feature_cols_disjoint(self) -> "DatasetSpec":
        """B2.2: every feature column appears in exactly one of
        `feature_real_cols` / `feature_categorical_cols` (a column
        cannot be declared as both numeric and categorical)."""
        overlap = set(self.feature_real_cols) & set(self.feature_categorical_cols)
        if overlap:
            raise ValueError(
                f"DatasetSpec({self.name!r}): columns {sorted(overlap)} "
                f"appear in both feature_real_cols and feature_categorical_cols"
            )
        return self


class ModelSpec(BaseModel):
    """Per-model registry record (B3.2).

    The field set is the scaffold for Phase B2 (adapters); Phase B2
    will add the adapter callable, default-config hooks, and the
    per-family search-space module pointer (used by Phase B8 HPO).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    family: ModelFamily
    task_types: tuple[TaskType, ...] = Field(min_length=1)
    supports_proba: bool
    reason: str = Field(min_length=1)


class ExperimentSpec(BaseModel):
    """Per-experiment configuration (B6)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ExperimentKind
    seeds: tuple[int, ...] = Field(default=(0, 1, 2), min_length=1)


_ALL_SENTINEL = "all"


class BenchmarkConfig(BaseModel):
    """Top-level config the CLI loads (the single entry-point object).

    `datasets` and `models` are tuples of registry names; the special
    value ``"all"`` (sentinel) means "every registered name" and must
    appear alone (no mixed sentinel-and-name and no duplicates).
    Cross-validation against the live registry happens in
    `benchmarks.run.main` after registries are populated by Phase B1
    / B2.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    datasets: tuple[str, ...] = Field(min_length=1)
    models: tuple[str, ...] = Field(min_length=1)
    experiments: tuple[ExperimentSpec, ...] = Field(min_length=1)
    output_dir: Path
    cache_dir: Path | None = None

    @model_validator(mode="after")
    def _all_sentinel_alone(self) -> "BenchmarkConfig":
        """The ``"all"`` sentinel must appear alone, not mixed with
        concrete names, and the tuple must not carry duplicates.
        Catches typos at config-load time."""
        for field_name in ("datasets", "models"):
            names = getattr(self, field_name)
            if _ALL_SENTINEL in names and len(names) != 1:
                raise ValueError(
                    f"BenchmarkConfig.{field_name}: the {_ALL_SENTINEL!r} "
                    f"sentinel must appear alone; got {list(names)}"
                )
            if len(names) != len(set(names)):
                raise ValueError(
                    f"BenchmarkConfig.{field_name}: duplicate entries: "
                    f"{[n for n in names if names.count(n) > 1]}"
                )
        return self
