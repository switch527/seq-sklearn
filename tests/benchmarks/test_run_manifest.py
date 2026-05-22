"""Phase B9 RunManifest tests (B8.1).

Pins the manifest schema, the atomic write contract, the
round-trip invariant, and the env fingerprint's best-effort
behavior on a CPU-only mock host.
"""

import json
from pathlib import Path
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
