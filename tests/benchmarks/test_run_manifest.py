"""Phase B9 RunManifest tests (B8.1).

Pins the manifest schema, the atomic write contract, the
round-trip invariant, and the env fingerprint's best-effort
behavior on a CPU-only mock host.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.run_manifest import (
    EnvironmentFingerprint,
    RunManifest,
    build_environment_fingerprint,
    build_run_manifest,
    load_run_manifest,
    run_manifest_path,
    write_run_manifest,
)

from tests.benchmarks._fakes import register_all_fakes_and_get_panels


def _config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        datasets=("ds_a", "ds_b"),
        models=("m_x",),
        experiments=(ExperimentSpec(kind="raw_loss", seeds=(0, 1)),),
        output_dir=tmp_path / "out",
    )


# --- environment fingerprint --------------------------------------------------


def test_build_environment_fingerprint_populates_required_fields() -> None:
    env = build_environment_fingerprint()
    # Platform + python are always present (no best-effort branch).
    assert env.platform
    assert env.python_version
    # Pinned packages always have keys; values may be None when
    # the package is absent from the active environment.
    assert "seq-sklearn" in env.package_versions
    assert "torch" in env.package_versions
    assert "pydantic" in env.package_versions


def test_build_environment_fingerprint_handles_missing_nvidia_smi() -> None:
    """Pin that a missing `nvidia-smi` collapses to `cuda_driver=None`
    rather than raising; the manifest must be writable on a
    CPU-only host."""
    with patch("benchmarks.run_manifest.shutil.which", return_value=None):
        env = build_environment_fingerprint()
        assert env.cuda_driver is None


def test_resolve_cuda_driver_version_subprocess_oserror_returns_none() -> None:
    """qa-I1 + code-I2: an OSError out of `subprocess.run` (e.g.,
    nvidia-smi binary on PATH but exec fails) must collapse to None
    rather than escape."""
    with (
        patch("benchmarks.run_manifest.shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch("benchmarks.run_manifest.subprocess.run", side_effect=OSError("fake")),
    ):
        env = build_environment_fingerprint()
        assert env.cuda_driver is None


def test_resolve_cuda_driver_version_nonzero_returncode_returns_none() -> None:
    """qa-I1 + code-I2: a non-zero exit from nvidia-smi must
    collapse to None."""
    from subprocess import CompletedProcess

    with (
        patch("benchmarks.run_manifest.shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch(
            "benchmarks.run_manifest.subprocess.run",
            return_value=CompletedProcess(args=[], returncode=1, stdout="", stderr=""),
        ),
    ):
        env = build_environment_fingerprint()
        assert env.cuda_driver is None


def test_resolve_cuda_driver_version_empty_stdout_returns_none() -> None:
    """qa-I1 + code-I2: a zero-exit-but-empty-stdout result also
    collapses to None (defensive against a future
    --format=csv,noheader change)."""
    from subprocess import CompletedProcess

    with (
        patch("benchmarks.run_manifest.shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch(
            "benchmarks.run_manifest.subprocess.run",
            return_value=CompletedProcess(args=[], returncode=0, stdout="   \n", stderr=""),
        ),
    ):
        env = build_environment_fingerprint()
        assert env.cuda_driver is None


def test_resolve_cpu_model_empty_processor_returns_none() -> None:
    """qa-N1: `platform.processor()` returns "" on some Linux
    kernel + arch combos; the helper must report None so the
    manifest does not masquerade an empty string as a valid
    identifier."""
    with patch("benchmarks.run_manifest.platform.processor", return_value=""):
        env = build_environment_fingerprint()
        assert env.cpu_model is None


# --- build / read round-trip --------------------------------------------------


def test_build_run_manifest_sorts_dataset_and_model_tuples(tmp_path: Path) -> None:
    """The roster is sorted so two manifests over the same config
    are byte-identical."""
    config = BenchmarkConfig(
        datasets=("ds_z", "ds_a"),  # unsorted on input
        models=("m_b", "m_a"),
        experiments=(ExperimentSpec(kind="raw_loss"),),
        output_dir=tmp_path / "out",
    )
    manifest = build_run_manifest(
        config,
        run_id="r1",
        library_git_sha="sha",
        profile="smoke",
        hardware_tier="cpu",
        output_root=tmp_path / "out",
    )
    assert manifest.datasets == ("ds_a", "ds_z")
    assert manifest.models == ("m_a", "m_b")


def test_write_run_manifest_round_trip(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifest = build_run_manifest(
        config,
        run_id="r-id",
        library_git_sha="deadbeef",
        profile="smoke",
        hardware_tier="cpu",
        output_root=config.output_dir,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    written_path = write_run_manifest(config.output_dir, manifest)
    assert written_path == run_manifest_path(config.output_dir)
    loaded = load_run_manifest(config.output_dir)
    assert loaded == manifest


def test_write_run_manifest_produces_deterministic_json_bytes(tmp_path: Path) -> None:
    """Two manifests built from the same config produce
    byte-identical JSON (the dict's `sort_keys=True` is the
    invariant)."""
    config = _config(tmp_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    env = EnvironmentFingerprint(
        platform="Linux-test",
        python_version="3.14.0",
        cpu_model="test-cpu",
        gpu_model=None,
        cuda_runtime=None,
        cuda_driver=None,
        package_versions={"seq-sklearn": "0.0.0"},
    )
    m1 = build_run_manifest(
        config,
        run_id="r1",
        library_git_sha="sha",
        profile="smoke",
        hardware_tier="cpu",
        output_root=config.output_dir,
        started_at_utc="2026-05-22T00:00:00+00:00",
        environment=env,
    )
    m2 = build_run_manifest(
        config,
        run_id="r1",
        library_git_sha="sha",
        profile="smoke",
        hardware_tier="cpu",
        output_root=config.output_dir,
        started_at_utc="2026-05-22T00:00:00+00:00",
        environment=env,
    )
    write_run_manifest(config.output_dir, m1)
    bytes1 = run_manifest_path(config.output_dir).read_bytes()
    write_run_manifest(config.output_dir, m2)
    bytes2 = run_manifest_path(config.output_dir).read_bytes()
    assert bytes1 == bytes2


def test_write_run_manifest_atomic_replace_on_overwrite(tmp_path: Path) -> None:
    """A second write replaces the first atomically (no partial
    file). Pinning that the file ends with the new contents
    confirms the write-to-temp-then-rename path."""
    config = _config(tmp_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    m1 = build_run_manifest(
        config,
        run_id="run-1",
        library_git_sha="sha",
        profile="smoke",
        hardware_tier="cpu",
        output_root=config.output_dir,
    )
    write_run_manifest(config.output_dir, m1)
    m2 = build_run_manifest(
        config,
        run_id="run-2",  # different run_id
        library_git_sha="sha",
        profile="smoke",
        hardware_tier="cpu",
        output_root=config.output_dir,
    )
    write_run_manifest(config.output_dir, m2)
    loaded = load_run_manifest(config.output_dir)
    assert loaded.run_id == "run-2"


def test_load_run_manifest_missing_raises() -> None:
    with pytest.raises(FileNotFoundError, match="no run manifest"):
        load_run_manifest(Path("/tmp/does-not-exist-b9-manifest-test"))


def test_load_run_manifest_invalid_json_raises_value_error(tmp_path: Path) -> None:
    """qa-I2: a corrupt JSON file (partial write, disk corruption)
    must surface as a typed ValueError so the CLI catch maps it to
    a clean exit 1, not a json.JSONDecodeError traceback."""
    out = tmp_path / "out"
    out.mkdir(parents=True)
    (out / "run_manifest.json").write_text("not valid json }{{")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_run_manifest(out)


def test_load_run_manifest_schema_mismatch_raises(tmp_path: Path) -> None:
    """A manifest JSON written by an older library version with
    missing required fields must surface as a pydantic
    `ValidationError`, not a silent partial load."""
    from pydantic import ValidationError

    out = tmp_path / "out"
    out.mkdir(parents=True)
    # Write a JSON missing the required `environment` field.
    payload = {"run_id": "r1", "started_at_utc": "2026-05-22T00:00:00+00:00"}
    (out / "run_manifest.json").write_text(json.dumps(payload))
    with pytest.raises(ValidationError):
        load_run_manifest(out)


# --- completed_at_utc lifecycle ----------------------------------------------


def test_completed_at_utc_starts_none_and_is_set_on_completion(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    initial = build_run_manifest(
        config,
        run_id="r1",
        library_git_sha="sha",
        profile="smoke",
        hardware_tier="cpu",
        output_root=config.output_dir,
    )
    assert initial.completed_at_utc is None
    completed = build_run_manifest(
        config,
        run_id="r1",
        library_git_sha="sha",
        profile="smoke",
        hardware_tier="cpu",
        output_root=config.output_dir,
        started_at_utc=initial.started_at_utc,
        completed_at_utc="2026-05-22T12:00:00+00:00",
        environment=initial.environment,
    )
    assert completed.completed_at_utc == "2026-05-22T12:00:00+00:00"


# --- structural --------------------------------------------------------------


def test_build_run_manifest_records_dataset_sha256_when_registered(
    tmp_path: Path,
) -> None:
    """B9 arch-I4: each dataset's `integrity_sha256` round-trips
    onto the manifest's `dataset_sha256` map. Registered datasets
    surface their hash; unregistered names map to None."""
    from benchmarks.config import DatasetSpec
    from benchmarks.registry.datasets import register_dataset

    sha = "f" * 64
    spec = DatasetSpec(
        name="ds_b9_test",
        task_type="binary",
        access_tier="OPEN",
        size_tier="small",
        balance="balanced",
        modality="numeric",
        source_uri="https://example.com/d.csv",
        integrity_sha256=sha,
        archive_basename="d.csv",
        entity_col="id",
        time_col="t",
        target_col="y",
        feature_real_cols=("x",),
        feature_categorical_cols=(),
        lookback=8,
        observation_cutoff_rule=None,
        densification_policy=None,
        positive_label=1,
        excluded=False,
        exclusion_reason=None,
        citation="Test 2026",
    )

    def _stub_loader(_cache_root: Path) -> object:
        raise NotImplementedError

    register_dataset(spec, _stub_loader)

    config = BenchmarkConfig(
        datasets=("ds_b9_test", "ds_unregistered"),
        models=("m",),
        experiments=(ExperimentSpec(kind="raw_loss"),),
        output_dir=tmp_path / "out",
    )
    manifest = build_run_manifest(
        config,
        run_id="r1",
        library_git_sha="sha",
        profile="smoke",
        hardware_tier="cpu",
        output_root=tmp_path / "out",
    )
    assert manifest.dataset_sha256["ds_b9_test"] == sha
    assert manifest.dataset_sha256["ds_unregistered"] is None


def test_run_manifest_is_frozen_and_extra_forbid(tmp_path: Path) -> None:
    from pydantic import ValidationError

    config = _config(tmp_path)
    manifest = build_run_manifest(
        config,
        run_id="r1",
        library_git_sha="sha",
        profile="smoke",
        hardware_tier="cpu",
        output_root=tmp_path / "out",
    )
    with pytest.raises(ValidationError):
        manifest.run_id = "r2"  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ValidationError):
        RunManifest(
            run_id="r1",
            started_at_utc="2026-05-22T00:00:00+00:00",
            profile="smoke",
            hardware_tier="cpu",
            library_git_sha="sha",
            benchmarks_git_sha="sha",
            environment=manifest.environment,
            experiments=manifest.experiments,
            output_root=str(tmp_path / "out"),
            stray_field=1,  # pyright: ignore[reportCallIssue]
        )


# --- B13 RunManifest.fingerprint() invariants --------------------------------


@pytest.fixture
def _registered_fakes() -> None:
    register_all_fakes_and_get_panels()


def _build_simple_manifest(
    *,
    library_git_sha: str = "0" * 40,
    completed_at_utc: str | None = None,
    tmp_path: Path,
) -> Any:
    from benchmarks.config import BenchmarkConfig, ExperimentSpec
    from benchmarks.run_manifest import build_run_manifest

    config = BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(ExperimentSpec(kind="raw_loss", seeds=(0,)),),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    return build_run_manifest(
        config=config,
        run_id="run-1",
        library_git_sha=library_git_sha,
        profile="smoke",
        hardware_tier="cpu",
        output_root=tmp_path / "out",
        started_at_utc="2026-05-28T12:00:00+00:00",
        completed_at_utc=completed_at_utc,
    )


def test_run_manifest_fingerprint_same_content_is_stable(
    _registered_fakes: None, tmp_path: Path
) -> None:
    """qa-I1: two manifests built with identical inputs produce
    identical SHA-256 hex digests."""
    m1 = _build_simple_manifest(tmp_path=tmp_path)
    m2 = _build_simple_manifest(tmp_path=tmp_path)
    assert m1.fingerprint() == m2.fingerprint()
    assert len(m1.fingerprint()) == 64  # SHA-256 hex digest length


def test_run_manifest_fingerprint_differs_on_library_git_sha_change(
    _registered_fakes: None, tmp_path: Path
) -> None:
    """qa-I1: changing `library_git_sha` changes the digest. The
    Gemini-C3 stale-rollup detection depends on this."""
    m1 = _build_simple_manifest(library_git_sha="a" * 40, tmp_path=tmp_path)
    m2 = _build_simple_manifest(library_git_sha="b" * 40, tmp_path=tmp_path)
    assert m1.fingerprint() != m2.fingerprint()


def test_run_manifest_fingerprint_invariant_under_completed_at_utc_change(
    _registered_fakes: None, tmp_path: Path
) -> None:
    """qa-I1: the volatile `completed_at_utc` field is excluded
    from the digest by design (a run with `completed_at_utc=None`
    written at start-of-run and the same run with
    `completed_at_utc=<timestamp>` written at run-end must produce
    the SAME fingerprint, otherwise stale-rollup detection breaks
    on every clean run)."""
    m1 = _build_simple_manifest(completed_at_utc=None, tmp_path=tmp_path)
    m2 = _build_simple_manifest(completed_at_utc="2026-05-28T13:00:00+00:00", tmp_path=tmp_path)
    assert m1.fingerprint() == m2.fingerprint()
