"""Phase B0 scaffold gates.

What the suite proves:

1. The package imports.
2. The pydantic config schema validates a minimal payload (including
   the binary `positive_label` cross-field validator from B2.2).
3. The registry stubs are wired (decorators import, mappings exist,
   typed lookup errors fire).
4. `python -m benchmarks.run --dry-run` exits 0 against the minimal
   TOML config.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from benchmarks.config import (
    BenchmarkConfig,
    DatasetSpec,
    ExperimentSpec,
    ModelSpec,
)
from benchmarks.datasets._base import PanelDataset
from benchmarks.registry import (
    DatasetNotRegisteredError,
    ModelNotRegisteredError,
    get_dataset,
    get_loader,
    get_model,
    list_datasets,
    list_models,
    register_dataset,
    register_model,
)

# Tests legitimately exercise `_ConfigLoadError` and `_load_config`;
# they are module-private helpers but the test gate is one of their two
# consumers (the other being `main()` itself). Pyright flags the
# private-from-outside use; suppress per name.
from benchmarks.run import (
    _ConfigLoadError,  # pyright: ignore[reportPrivateUsage]
    _load_config,  # pyright: ignore[reportPrivateUsage]
    main,
)
from pydantic import ValidationError

import benchmarks


def _stub_loader(_cache_root: Path) -> PanelDataset:
    """Loader stub for tests that exercise registration only; never
    invoked because the registry tests do not call `get_loader`."""
    raise NotImplementedError("stub loader; tests should not invoke it")


def test_package_imports() -> None:
    # Pyright doesn't surface `__all__` as a module attribute (known
    # quirk: it's treated as an export marker, not as runtime state);
    # `getattr` is the explicit-runtime form.
    assert getattr(benchmarks, "__all__", None) == []


def test_minimal_benchmark_config_validates() -> None:
    cfg = BenchmarkConfig(
        datasets=("a",),
        models=("m",),
        experiments=(ExperimentSpec(kind="raw_loss"),),
        output_dir=Path("/tmp/out"),
    )
    assert cfg.experiments[0].seeds == (0, 1, 2)


def test_extra_forbid_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="extra"):
        BenchmarkConfig(
            datasets=("a",),
            models=("m",),
            experiments=(ExperimentSpec(kind="raw_loss"),),
            output_dir=Path("/tmp/out"),
            wat=42,  # type: ignore[call-arg]
        )


def _valid_dataset_kwargs(
    *,
    name: str = "ds",
    task_type: str = "regression_point",
    positive_label: int | str | None = None,
) -> dict[str, Any]:
    """Helper: a syntactically valid `DatasetSpec` kwargs dict.

    Returns ``dict[str, Any]`` because the pydantic constructor does
    its own per-field validation; a stricter return type would
    require a `TypedDict` mirror of `DatasetSpec`, which is duplicate
    bookkeeping the test does not need.
    """
    return {
        "name": name,
        "task_type": task_type,
        "access_tier": "OPEN",
        "size_tier": "small",
        "balance": "balanced",
        "modality": "numeric",
        "source_uri": "https://example.com/data.csv",
        "integrity_sha256": "0" * 64,
        "archive_basename": "data.csv",
        "entity_col": "id",
        "time_col": "t",
        "target_col": "y",
        "feature_real_cols": ("x",),
        "feature_categorical_cols": (),
        "lookback": 8,
        "observation_cutoff_rule": None,
        "densification_policy": None,
        "positive_label": positive_label,
        "excluded": False,
        "exclusion_reason": None,
        "citation": "Acme et al. 2020",
    }


def test_dataset_spec_binary_requires_positive_label() -> None:
    with pytest.raises(ValueError, match="positive_label"):
        DatasetSpec(**_valid_dataset_kwargs(task_type="binary"))


def test_dataset_spec_binary_accepts_positive_label() -> None:
    spec = DatasetSpec(**_valid_dataset_kwargs(task_type="binary", positive_label=1))
    assert spec.positive_label == 1


def test_dataset_spec_non_binary_rejects_positive_label() -> None:
    with pytest.raises(ValueError, match="binary-only"):
        DatasetSpec(**_valid_dataset_kwargs(task_type="multiclass", positive_label="event"))


def test_dataset_spec_integrity_sha_pattern_enforced() -> None:
    bad = _valid_dataset_kwargs()
    bad["integrity_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="pattern"):
        DatasetSpec(**bad)


def test_dataset_spec_excluded_without_reason_raises() -> None:
    """B1.5: `excluded=True` requires a non-empty `exclusion_reason`."""
    bad = _valid_dataset_kwargs()
    bad["excluded"] = True
    bad["exclusion_reason"] = None
    with pytest.raises(ValueError, match="non-empty `exclusion_reason`"):
        DatasetSpec(**bad)


def test_dataset_spec_excluded_with_blank_reason_raises() -> None:
    """`excluded=True, exclusion_reason=''` is rejected (the blank
    string is the degenerate version of None and fails the same
    invariant)."""
    bad = _valid_dataset_kwargs()
    bad["excluded"] = True
    bad["exclusion_reason"] = ""
    with pytest.raises(ValueError, match="non-empty `exclusion_reason`"):
        DatasetSpec(**bad)


def test_dataset_spec_active_with_exclusion_reason_raises() -> None:
    """B1.5: an active spec (`excluded=False`) must NOT carry an
    `exclusion_reason`; pydantic rejects the stale-reason shape."""
    bad = _valid_dataset_kwargs()
    bad["excluded"] = False
    bad["exclusion_reason"] = "stale reason from a previous edit"
    with pytest.raises(ValueError, match="exclusion_reason is set"):
        DatasetSpec(**bad)


def test_dataset_spec_overlapping_feature_cols_raises() -> None:
    """B2.2: a column cannot be both numeric and categorical."""
    bad = _valid_dataset_kwargs()
    bad["feature_real_cols"] = ("x", "y")
    bad["feature_categorical_cols"] = ("y", "z")
    with pytest.raises(ValueError, match="appear in both"):
        DatasetSpec(**bad)


def test_dataset_spec_excluded_binary_without_positive_label_ok() -> None:
    """code-I2: excluded specs are exempt from the
    `_positive_label_only_on_binary` validator; an excluded binary
    spec with `positive_label=None` validates cleanly because it
    will never be used in any experiment."""
    kw = _valid_dataset_kwargs(task_type="binary")
    kw["excluded"] = True
    kw["exclusion_reason"] = "one-row-per-entity; fails B1.3"
    spec = DatasetSpec(**kw)
    assert spec.excluded is True
    assert spec.positive_label is None


def test_dataset_registry_register_lookup_list() -> None:
    spec = DatasetSpec(**_valid_dataset_kwargs(name="unique_for_scaffold_test"))
    register_dataset(spec, _stub_loader)
    assert get_dataset("unique_for_scaffold_test") is spec
    assert get_loader("unique_for_scaffold_test") is _stub_loader
    assert "unique_for_scaffold_test" in list_datasets()


def test_dataset_registry_get_missing_raises_typed() -> None:
    with pytest.raises(DatasetNotRegisteredError):
        get_dataset("__never_registered__")


def test_dataset_registry_get_loader_missing_raises_typed() -> None:
    with pytest.raises(DatasetNotRegisteredError):
        get_loader("__never_registered__")


def test_dataset_registry_duplicate_name_different_spec_raises() -> None:
    register_dataset(DatasetSpec(**_valid_dataset_kwargs(name="dup_test")), _stub_loader)
    different = DatasetSpec(**_valid_dataset_kwargs(name="dup_test")).model_copy(
        update={"lookback": 9999}
    )
    with pytest.raises(ValueError, match="already registered"):
        register_dataset(different, _stub_loader)


def test_dataset_registry_duplicate_name_different_loader_raises() -> None:
    spec = DatasetSpec(**_valid_dataset_kwargs(name="dup_loader_test"))
    register_dataset(spec, _stub_loader)

    def _other_loader(_cache_root: Path) -> PanelDataset:
        raise NotImplementedError

    with pytest.raises(ValueError, match=r"different\s+loader"):
        register_dataset(spec, _other_loader)


def test_model_spec_requires_at_least_one_task_type() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ModelSpec(
            name="m",
            family="gbm",
            task_types=(),
            supports_proba=True,
            reason="because",
        )


def test_model_registry_register_lookup_list() -> None:
    spec = ModelSpec(
        name="m_for_scaffold_test",
        family="seq_sklearn",
        task_types=("binary", "multiclass"),
        supports_proba=True,
        reason="library's TFTClassifier; covers the panel-classification arm",
    )
    register_model(spec)
    assert get_model("m_for_scaffold_test") is spec
    assert "m_for_scaffold_test" in list_models()


def test_model_registry_get_missing_raises_typed() -> None:
    with pytest.raises(ModelNotRegisteredError):
        get_model("__never_registered__")


def test_cli_dry_run_exits_zero(minimal_config_toml: Path) -> None:
    """End-to-end gate: the published entry point validates a
    minimal config and exits 0. The subprocess path is the one CI
    actually runs, so it's the one we test."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.run",
            "--config",
            str(minimal_config_toml),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"benchmarks.run --dry-run exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


# ----------------------------------------------------------------- #
# Registry: idempotent same-spec + duplicate-different-spec on model
# registry + typed-error-subclass contract (qa R1 C1/I1/I3)
# ----------------------------------------------------------------- #


def test_dataset_registry_idempotent_same_spec() -> None:
    spec = DatasetSpec(**_valid_dataset_kwargs(name="idempotent_ds"))
    register_dataset(spec, _stub_loader)
    register_dataset(spec, _stub_loader)  # should not raise
    assert get_dataset("idempotent_ds") is spec


def test_model_registry_idempotent_same_spec() -> None:
    spec = ModelSpec(
        name="idempotent_m",
        family="gbm",
        task_types=("binary",),
        supports_proba=True,
        reason="idempotent registration smoke",
    )
    register_model(spec)
    register_model(spec)
    assert get_model("idempotent_m") is spec


def test_model_registry_duplicate_name_different_spec_raises() -> None:
    spec = ModelSpec(
        name="dup_m",
        family="gbm",
        task_types=("binary",),
        supports_proba=True,
        reason="initial",
    )
    register_model(spec)
    different = spec.model_copy(update={"supports_proba": False})
    with pytest.raises(ValueError, match="already registered"):
        register_model(different)


def test_dataset_not_registered_error_is_subclass_of_key_error() -> None:
    """The typed-catch contract: `except KeyError` must catch our
    typed error, and `except DatasetNotRegisteredError` must NOT
    catch a plain `KeyError`."""
    assert issubclass(DatasetNotRegisteredError, KeyError)
    with pytest.raises(KeyError):
        get_dataset("__never_registered__")
    with pytest.raises(DatasetNotRegisteredError):
        get_dataset("__never_registered__")


def test_model_not_registered_error_is_subclass_of_key_error() -> None:
    assert issubclass(ModelNotRegisteredError, KeyError)
    with pytest.raises(KeyError):
        get_model("__never_registered__")
    with pytest.raises(ModelNotRegisteredError):
        get_model("__never_registered__")


# ----------------------------------------------------------------- #
# Registry isolation across tests (arch R1 C1)
# ----------------------------------------------------------------- #


def test_registry_isolation_canary_a() -> None:
    """Companion of `_b` below. Each test registers the same canary
    name; if the conftest autouse fixture isolates registry state,
    both pass regardless of the order `pytest-randomly` picks. If the
    fixture is missing, the second test runs into the
    duplicate-name guard and fails."""
    register_dataset(
        DatasetSpec(**_valid_dataset_kwargs(name="isolation_canary")),
        _stub_loader,
    )
    assert "isolation_canary" in list_datasets()


def test_registry_isolation_canary_b() -> None:
    register_dataset(
        DatasetSpec(**_valid_dataset_kwargs(name="isolation_canary")),
        _stub_loader,
    )
    assert "isolation_canary" in list_datasets()


# ----------------------------------------------------------------- #
# Cross-field validators (qa R1 N1) + BenchmarkConfig sentinel (arch I2)
# ----------------------------------------------------------------- #


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "regression_point", "regression_quantile"],
)
def test_dataset_spec_non_binary_rejects_positive_label_parametrized(task_type: str) -> None:
    with pytest.raises(ValueError, match="binary-only"):
        DatasetSpec(**_valid_dataset_kwargs(task_type=task_type, positive_label="x"))


def test_benchmark_config_all_sentinel_alone_rejects_mix() -> None:
    with pytest.raises(ValueError, match="sentinel must appear alone"):
        BenchmarkConfig(
            datasets=("all", "amex"),
            models=("m",),
            experiments=(ExperimentSpec(kind="raw_loss"),),
            output_dir=Path("/tmp/out"),
        )


def test_benchmark_config_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicate entries"):
        BenchmarkConfig(
            datasets=("a", "a"),
            models=("m",),
            experiments=(ExperimentSpec(kind="raw_loss"),),
            output_dir=Path("/tmp/out"),
        )


def test_benchmark_config_all_sentinel_alone_rejects_mix_on_models() -> None:
    """Companion to the datasets-side test, exercising the `models`
    iteration of `_all_sentinel_alone` so a future edit that strips
    the loop's models tuple element is caught."""
    with pytest.raises(ValueError, match="sentinel must appear alone"):
        BenchmarkConfig(
            datasets=("a",),
            models=("all", "gbm"),
            experiments=(ExperimentSpec(kind="raw_loss"),),
            output_dir=Path("/tmp/out"),
        )


def test_benchmark_config_rejects_duplicate_names_on_models() -> None:
    with pytest.raises(ValueError, match="duplicate entries"):
        BenchmarkConfig(
            datasets=("a",),
            models=("m", "m"),
            experiments=(ExperimentSpec(kind="raw_loss"),),
            output_dir=Path("/tmp/out"),
        )


def test_benchmark_config_all_sentinel_alone_accepts_solo() -> None:
    cfg = BenchmarkConfig(
        datasets=("all",),
        models=("all",),
        experiments=(ExperimentSpec(kind="raw_loss"),),
        output_dir=Path("/tmp/out"),
    )
    assert cfg.datasets == ("all",)


def test_experiment_spec_rejects_empty_seeds() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ExperimentSpec(kind="raw_loss", seeds=())


# ----------------------------------------------------------------- #
# _load_config error paths (qa R1 C3)
# ----------------------------------------------------------------- #


def test_load_config_missing_file_raises_typed(tmp_path: Path) -> None:
    with pytest.raises(_ConfigLoadError, match="not found"):
        _load_config(tmp_path / "does_not_exist.toml")


def test_load_config_malformed_toml_raises_typed(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[[[ not valid", encoding="utf-8")
    with pytest.raises(_ConfigLoadError, match="not valid TOML"):
        _load_config(path)


def test_load_config_schema_failure_raises_typed(tmp_path: Path) -> None:
    """Schema-valid TOML but the payload fails pydantic validation
    (empty `experiments` violates `min_length=1`). The chained
    exception is the original `ValidationError` so a caller can
    inspect it if needed."""
    path = tmp_path / "schema_bad.toml"
    path.write_text(
        f'datasets = ["a"]\nmodels = ["m"]\noutput_dir = "{tmp_path / "out"}"\nexperiments = []\n',
        encoding="utf-8",
    )
    with pytest.raises(_ConfigLoadError, match="schema validation") as excinfo:
        _load_config(path)
    assert isinstance(excinfo.value.__cause__, ValidationError)


# ----------------------------------------------------------------- #
# CLI in-process gates (qa R1 C2; gives run.py real coverage)
# ----------------------------------------------------------------- #


def test_main_returns_zero_on_dry_run(minimal_config_toml: Path) -> None:
    rc = main(["--config", str(minimal_config_toml), "--dry-run"])
    assert rc == 0


def test_benchmark_config_rejects_duplicate_experiment_kinds(tmp_path: Path) -> None:
    """B8 arch-I2 (absorbed in B9): the config validator must
    reject more than one `ExperimentSpec` per kind so the
    asymmetric first-spec-wins budget + union seeds combination
    cannot silently pick one over the other."""
    del tmp_path
    with pytest.raises(ValueError, match="at most one"):
        BenchmarkConfig(
            datasets=("a",),
            models=("m",),
            experiments=(
                ExperimentSpec(kind="raw_loss", seeds=(0,)),
                ExperimentSpec(kind="raw_loss", seeds=(1,)),
            ),
            output_dir=Path("/tmp/out"),
        )


@pytest.mark.parametrize(
    "kind",
    ["raw_loss", "ensemble", "training_time", "hpo_uplift"],
)
def test_benchmark_config_rejects_duplicates_for_every_kind(kind: str) -> None:
    """qa-N2: parametrize over every `ExperimentKind` so the
    validator's regression catch covers all four kinds, not just
    raw_loss. The B8 arch-I2 motivating case was an
    `hpo_uplift` duplicate with budget overrides; pin that
    explicitly."""
    with pytest.raises(ValueError, match="at most one"):
        BenchmarkConfig(
            datasets=("a",),
            models=("m",),
            experiments=(
                ExperimentSpec(kind=kind, seeds=(0,)),  # type: ignore[arg-type]
                ExperimentSpec(kind=kind, seeds=(1,)),  # type: ignore[arg-type]
            ),
            output_dir=Path("/tmp/out"),
        )


def test_build_run_environment_detects_cuda_hardware_tier() -> None:
    """B9 arch-C2 close: when CUDA is available the default
    hardware_tier is gpu_single, else cpu. Pin both paths with
    `torch.cuda.is_available` mocked."""
    from unittest.mock import patch

    from benchmarks.experiments import build_run_environment

    with patch("benchmarks.experiments.raw_loss._detect_hardware_tier", return_value="gpu_single"):
        env = build_run_environment(profile="smoke")
        assert env.hardware_tier == "gpu_single"
    with patch("benchmarks.experiments.raw_loss._detect_hardware_tier", return_value="cpu"):
        env = build_run_environment(profile="smoke")
        assert env.hardware_tier == "cpu"


def test_dispatch_kinds_unknown_kind_returns_two(tmp_path: Path) -> None:
    """qa-I4: the canary branch in `_dispatch_kinds` (an unknown
    `ExperimentKind` slips into the loop) must return 2 so the
    CLI exit code remains the documented value for unimplemented
    drivers."""
    from benchmarks.experiments import build_run_environment
    from benchmarks.run import _dispatch_kinds  # pyright: ignore[reportPrivateUsage]

    config = BenchmarkConfig(
        datasets=("a",),
        models=("m",),
        experiments=(ExperimentSpec(kind="raw_loss", seeds=(0,)),),
        output_dir=tmp_path / "out",
    )
    env = build_run_environment(profile="smoke", hardware_tier="cpu")
    rc = _dispatch_kinds(
        ["__unknown_kind__"],
        config=config,
        output_root=tmp_path / "out",
        env=env,
    )
    assert rc == 2


def test_main_returns_one_when_driver_raises_value_error(tmp_path: Path) -> None:
    """qa-I4 + B7/B8 deferral close: a `ValueError` out of any
    driver maps to exit 1 (matches the config-load exit code) so
    the user sees a clean message, not a Python traceback."""
    from tests.benchmarks._fakes import register_all_fakes_and_get_panels

    register_all_fakes_and_get_panels()

    output_dir = tmp_path / "out"
    config_path = tmp_path / "benchmark.toml"
    # An hpo_uplift run with no raw_loss manifest written first
    # would raise; but the driver also raises ValueError when the
    # config declares `cache_dir = None`. Use the missing-cache_dir
    # path as the controlled trigger; the config below omits the
    # `cache_dir` line entirely so it defaults to None.
    config_path.write_text(
        'datasets = ["fake_binary"]\n'
        'models = ["fake_constant_binary"]\n'
        f'output_dir = "{output_dir}"\n'
        "\n"
        "[[experiments]]\n"
        'kind = "raw_loss"\n'
        "seeds = [0]\n",
        encoding="utf-8",
    )
    # `cache_dir` defaults to None when omitted from the TOML.
    rc = main(["--config", str(config_path), "--experiment", "raw_loss"])
    assert rc == 1


def test_main_resume_after_crash_preserves_run_id(tmp_path: Path) -> None:
    """B9 arch-C1 close: an existing `run_manifest.json` (left
    behind by a crashed previous run) must NOT be overwritten with
    a fresh `run_id` on resume. The second invocation reuses the
    prior manifest's `run_id` + `started_at_utc` so the
    `ResultRow.run_id == RunManifest.run_id` join continues to
    hold for cells the first invocation already wrote."""
    from benchmarks.run_manifest import load_run_manifest

    from tests.benchmarks._fakes import register_all_fakes_and_get_panels

    register_all_fakes_and_get_panels()
    output_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(
        'datasets = ["fake_binary"]\n'
        'models = ["fake_constant_binary"]\n'
        f'output_dir = "{output_dir}"\n'
        f'cache_dir = "{cache_dir}"\n'
        "\n"
        "[[experiments]]\n"
        'kind = "raw_loss"\n'
        "seeds = [0]\n",
        encoding="utf-8",
    )
    rc1 = main(["--config", str(config_path), "--experiment", "raw_loss"])
    assert rc1 == 0
    manifest_1 = load_run_manifest(output_dir)
    # Second invocation against the same output_dir: must reuse
    # the prior run_id + started_at_utc.
    rc2 = main(["--config", str(config_path), "--experiment", "raw_loss"])
    assert rc2 == 0
    manifest_2 = load_run_manifest(output_dir)
    assert manifest_1.run_id == manifest_2.run_id
    assert manifest_1.started_at_utc == manifest_2.started_at_utc
    # The completion stamp is rewritten each time.
    assert manifest_2.completed_at_utc is not None


def test_main_dispatch_covers_every_experiment_kind() -> None:
    """B8 wires every `ExperimentKind` literal through the CLI
    dispatcher; the exit-2 branch in `main()` is a canary for
    future kinds added to the literal without an accompanying
    dispatcher arm. Pin that every current kind is in the
    dispatch order so a Literal extension surfaces here."""
    import typing

    from benchmarks.config import ExperimentKind
    from benchmarks.run import _DISPATCH_ORDER  # pyright: ignore[reportPrivateUsage]

    declared = set(typing.get_args(ExperimentKind))
    dispatched = set(_DISPATCH_ORDER)
    assert declared <= dispatched, (
        f"every ExperimentKind must appear in benchmarks/run.py's "
        f"_DISPATCH_ORDER; missing: {declared - dispatched}"
    )


def test_main_returns_one_on_missing_config(tmp_path: Path) -> None:
    rc = main(["--config", str(tmp_path / "does_not_exist.toml"), "--dry-run"])
    assert rc == 1


def test_main_returns_one_on_experiment_not_in_config(
    minimal_config_toml: Path,
) -> None:
    """The minimal config declares only `raw_loss`; --experiment=ensemble
    must error rather than silently no-op."""
    rc = main(
        [
            "--config",
            str(minimal_config_toml),
            "--experiment",
            "ensemble",
            "--dry-run",
        ]
    )
    assert rc == 1


def test_main_accepts_experiment_kind_in_config(minimal_config_toml: Path) -> None:
    rc = main(
        [
            "--config",
            str(minimal_config_toml),
            "--experiment",
            "raw_loss",
            "--dry-run",
        ]
    )
    assert rc == 0


def test_main_experiment_all_runs_raw_loss_before_dependents(
    tmp_path: Path,
) -> None:
    """B7 R1 ordering fix: `--experiment=all` must run `raw_loss`
    BEFORE the report-only kinds (`ensemble`, `training_time`) so
    the dependents see a populated manifest. A previous version of
    the CLI sorted the kinds alphabetically, which put `ensemble`
    first and crashed on an empty manifest. Pin the order so any
    future regression to alphabetical sort is caught."""
    from tests.benchmarks._fakes import register_all_fakes_and_get_panels

    register_all_fakes_and_get_panels()

    output_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    config_path = tmp_path / "benchmark.toml"
    # Declare the kinds out-of-order on purpose; the dispatcher's
    # topological tuple, not declaration order or alphabetical sort,
    # controls execution.
    config_path.write_text(
        'datasets = ["fake_binary"]\n'
        'models = ["fake_constant_binary"]\n'
        f'output_dir = "{output_dir}"\n'
        f'cache_dir = "{cache_dir}"\n'
        "\n"
        "[[experiments]]\n"
        'kind = "training_time"\n'
        "seeds = [0]\n"
        "\n"
        "[[experiments]]\n"
        'kind = "ensemble"\n'
        "seeds = [0]\n"
        "\n"
        "[[experiments]]\n"
        'kind = "raw_loss"\n'
        "seeds = [0]\n",
        encoding="utf-8",
    )
    rc = main(["--config", str(config_path), "--experiment", "all"])
    # All three drivers must complete; if `ensemble` or
    # `training_time` ran before `raw_loss`, they would raise on an
    # empty manifest and the CLI would propagate the exception
    # rather than return 0.
    assert rc == 0
    # The three Markdown reports land side-by-side under output_dir.
    assert (output_dir / "leaderboard.md").exists()
    assert (output_dir / "pairwise.md").exists()
    assert (output_dir / "training_time.md").exists()


def test_main_training_time_dispatch_writes_report_and_returns_zero(
    tmp_path: Path,
) -> None:
    """B7 contract: `--experiment=training_time` aggregates the B5
    manifest, writes `output_root/training_time.md` from the CLI
    (not the driver), and exits 0. This pins the dispatch arm so a
    future refactor cannot drop the report-write without the test
    catching it."""
    # The fakes registry exposes `fake_binary` + `fake_constant_binary`;
    # registering it here makes the CLI dispatch see a single OK
    # (dataset, model) group after raw_loss runs.
    from tests.benchmarks._fakes import register_all_fakes_and_get_panels

    register_all_fakes_and_get_panels()

    output_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(
        'datasets = ["fake_binary"]\n'
        'models = ["fake_constant_binary"]\n'
        f'output_dir = "{output_dir}"\n'
        f'cache_dir = "{cache_dir}"\n'
        "\n"
        "[[experiments]]\n"
        'kind = "raw_loss"\n'
        "seeds = [0]\n"
        "\n"
        "[[experiments]]\n"
        'kind = "training_time"\n'
        "seeds = [0]\n",
        encoding="utf-8",
    )
    # raw_loss first (B7 reads its manifest).
    rc_rl = main(["--config", str(config_path), "--experiment", "raw_loss"])
    assert rc_rl == 0
    rc_tt = main(["--config", str(config_path), "--experiment", "training_time"])
    assert rc_tt == 0
    report_path = output_dir / "training_time.md"
    assert report_path.exists()
    md = report_path.read_text(encoding="utf-8")
    assert "# Training-time report" in md
    assert "fake_binary" in md
    assert "fake_constant_binary" in md


# ----------------------------------------------------------------- #
# No-deep-imports gate (arch R1 I1): benchmarks must consume the
# library via its public facade only.
# ----------------------------------------------------------------- #


def test_benchmarks_package_does_not_import_seq_sklearn_internals() -> None:
    """Walk every `benchmarks/**/*.py` AST and reject any import
    that reaches into a private `seq_sklearn` component at any
    depth.

    Original R1 version checked only the second dot-component
    (catching `seq_sklearn._foo` but NOT `seq_sklearn.models._classifier`).
    B8 R1 (arch-C1) caught the gap; the tighter form below walks
    every component after `seq_sklearn` so any underscore-prefixed
    submodule, sub-package, or member raises.
    """
    import ast

    root = Path(benchmarks.__file__).resolve().parent
    leaked: list[str] = []
    for py in root.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            # An `import a, b` statement has multiple aliases; walking
            # `node.names[0]` only would miss a `, seq_sklearn._foo` in
            # the second position. Iterate every alias.
            mods: list[str] = []
            if isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    mods.append(node.module)
            elif isinstance(node, ast.Import):
                mods.extend(alias.name for alias in node.names)
            else:
                continue
            for mod in mods:
                if not mod.startswith("seq_sklearn."):
                    continue
                # Walk EVERY component after `seq_sklearn`. A leak is
                # any component (any depth) starting with `_`.
                parts = mod.split(".")[1:]
                if any(p.startswith("_") for p in parts):
                    leaked.append(f"{py.relative_to(root.parent)}: {mod}")
    assert not leaked, (
        f"benchmarks/ must consume the library via the public facade "
        f"only; deep imports into private seq_sklearn components found: "
        f"{leaked}"
    )
