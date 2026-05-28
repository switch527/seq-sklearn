"""Per-run reproducibility manifest (Phase B9 / B8.1).

Design B8.1 names the manifest as the single artifact sufficient
to reproduce a benchmark run: library + benchmarks git SHAs, the
dependency-version fingerprint, the environment fingerprint
(CPU / GPU model, CUDA version, driver), the dataset + model
roster the run touched, the resolved experiment specs, and a
pointer to the per-cell `ResultRow` shards under `results/`.

The run manifest is written EARLY in `benchmarks/run.py` (before
any cell starts executing) so a crashed run still leaves a
manifest of intent on disk. After the run finishes successfully,
the CLI rewrites it with a non-None `completed_at_utc` field;
B7.2 atomic-write semantics guarantee no half-written manifest.

This is the durable companion to `benchmarks/manifest.py`'s
`ResultRow`, which describes ONE cell. The two split cleanly: a
ResultRow records what came out of a cell; a RunManifest records
what went in to the whole run.
"""

import json
import logging
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.manifest import atomic_write_bytes
from benchmarks.registry import DatasetNotRegisteredError, get_dataset

logger = logging.getLogger(__name__)

_RUN_MANIFEST_FILENAME = "run_manifest.json"

# Dependency packages the manifest pins. The list is intentionally
# the harness's STABLE supply chain; a future TSC adapter branch
# extends this list when its optional deps land. Packages not
# installed return `None` (best-effort). The Phase B10 GBM
# adapter family (LightGBM / XGBoost / CatBoost) adds the three
# library entries below; the manifest's reproducibility contract
# pins each library's version on every run that touches the
# corresponding adapter.
_PINNED_PACKAGES: tuple[str, ...] = (
    "seq-sklearn",
    "torch",
    "lightning",
    "pandas",
    "numpy",
    "scikit-learn",
    "scipy",
    "optuna",
    "pydantic",
    "lightgbm",
    "xgboost",
    "catboost",
    # B12: aeon for classical-TSC family. Optional dep; absent in
    # the default CI install. `_safe_pkg_version` records None
    # when the package is missing.
    "aeon",
)


__all__ = [
    "EnvironmentFingerprint",
    "RunManifest",
    "build_run_manifest",
    "load_run_manifest",
    "run_manifest_path",
    "write_run_manifest",
]


class EnvironmentFingerprint(BaseModel):
    """OS + hardware + dependency fingerprint for a run.

    Every field is best-effort: a missing package, an unavailable
    `nvidia-smi`, or a CPU model that the platform module cannot
    expose collapses to a structured "unknown" sentinel rather
    than raising, so the manifest can still be written on the
    crash path.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    platform: str  # `platform.platform()` -> OS + kernel + arch
    python_version: str  # `sys.version.split()[0]`
    cpu_model: str | None  # best-effort; `platform.processor()` or `unknown`
    gpu_model: str | None  # torch.cuda.get_device_name(0) or None
    cuda_runtime: str | None  # torch.version.cuda or None
    cuda_driver: str | None  # nvidia-smi --query-gpu=driver_version or None
    package_versions: dict[str, str | None]  # name -> resolved version or None


class RunManifest(BaseModel):
    """Per-run reproducibility record (design B8.1).

    Written to `output_root/run_manifest.json` at the start of a
    benchmark run; the same path is rewritten with
    `completed_at_utc` populated on a clean exit.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Identity + lifecycle.
    run_id: str
    started_at_utc: str  # ISO-8601 UTC
    completed_at_utc: str | None = None  # populated on clean exit
    profile: str  # smoke / standard / full
    hardware_tier: str  # cpu / gpu_single

    # Reproducibility anchors.
    library_git_sha: str
    benchmarks_git_sha: str  # same as library in v1 (mono-repo)
    environment: EnvironmentFingerprint

    # Roster of what the run touched. The names are sorted ascending
    # so two manifests over the same config produce byte-identical
    # JSON output (round-trip invariant).
    datasets: tuple[str, ...] = Field(min_length=0)
    models: tuple[str, ...] = Field(min_length=0)
    experiments: tuple[ExperimentSpec, ...]

    # Per-dataset content fingerprints (B8.1 reproducibility
    # contract). Maps dataset_name -> the SHA-256 recorded on the
    # registered `DatasetSpec.integrity_sha256` field, so the
    # manifest pins which version of each dataset the run touched.
    # A name absent from the registry at manifest-build time maps
    # to `None`; the build is best-effort to keep the manifest
    # writable even on a partial registry.
    dataset_sha256: dict[str, str | None] = Field(default_factory=dict)

    # Output layout.
    output_root: str  # absolute or relative path; round-trips as str
    results_relpath: str = "results"  # always under `results/`


def _safe_pkg_version(name: str) -> str | None:
    """Resolve a package's installed version; return None if missing."""
    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        return None


def _resolve_cuda_driver_version() -> str | None:
    """Call `nvidia-smi --query-gpu=driver_version --format=csv,noheader`
    and return the first line, or None on any failure.

    Best-effort: nvidia-smi may be absent (CPU-only host), it may
    take too long (timeout fires), or the binary may not be on
    PATH. Every failure path returns None.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    out = completed.stdout.strip().splitlines()
    if not out:
        return None
    return out[0].strip() or None


def _resolve_gpu_model() -> str | None:
    """Return the first CUDA device's name, or None if CUDA absent."""
    try:
        import torch  # import-on-demand to avoid CPU-only import overhead
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def _resolve_cuda_runtime() -> str | None:
    """Return torch's compiled CUDA runtime version, or None."""
    try:
        import torch
    except ImportError:
        return None
    return getattr(torch.version, "cuda", None)


def _resolve_cpu_model() -> str | None:
    """Return the CPU model string; None if the platform cannot
    expose it.

    `platform.processor()` returns an empty string on Linux for
    some kernel + arch combos; in that case the field is reported
    as None rather than masquerading as a valid identifier.
    """
    cpu = platform.processor()
    if not cpu:
        return None
    return cpu


def build_environment_fingerprint() -> EnvironmentFingerprint:
    """Capture the OS + hardware + dependency fingerprint for the
    current process.

    Every helper is best-effort: a packaged-version lookup that
    fails or a missing nvidia-smi collapses to None. The manifest
    is meant to be writable on every supported platform without
    raising.
    """
    return EnvironmentFingerprint(
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        cpu_model=_resolve_cpu_model(),
        gpu_model=_resolve_gpu_model(),
        cuda_runtime=_resolve_cuda_runtime(),
        cuda_driver=_resolve_cuda_driver_version(),
        package_versions={name: _safe_pkg_version(name) for name in _PINNED_PACKAGES},
    )


def _resolve_dataset_sha256(dataset_names: tuple[str, ...]) -> dict[str, str | None]:
    """Look up each dataset's registered `integrity_sha256`.

    Best-effort: a name absent from the registry at manifest-build
    time maps to `None` so the manifest can still be written even
    when an experiment driver is about to refuse the missing-name
    cell. The registry's typed error stays out of the manifest
    layer.
    """
    out: dict[str, str | None] = {}
    for name in dataset_names:
        try:
            spec = get_dataset(name)
        except DatasetNotRegisteredError:
            out[name] = None
            continue
        out[name] = spec.integrity_sha256
    return out


def build_run_manifest(
    config: BenchmarkConfig,
    *,
    run_id: str,
    library_git_sha: str,
    profile: str,
    hardware_tier: str,
    output_root: Path,
    started_at_utc: str | None = None,
    completed_at_utc: str | None = None,
    environment: EnvironmentFingerprint | None = None,
) -> RunManifest:
    """Assemble a `RunManifest` from a config + run-environment.

    Defaults `started_at_utc` to "now" and `environment` to
    `build_environment_fingerprint()` for caller ergonomics. The
    `completed_at_utc` defaults to None (manifest-of-intent form);
    the CLI rewrites the manifest at clean exit with a populated
    timestamp. The library SHA is the same value B5's
    RunEnvironment carries; B8.1 names this as the second SHA's
    mono-repo collapse (a future split would diverge).
    """
    if started_at_utc is None:
        started_at_utc = datetime.now(UTC).isoformat()
    if environment is None:
        environment = build_environment_fingerprint()
    sorted_datasets = tuple(sorted(config.datasets))
    return RunManifest(
        run_id=run_id,
        started_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
        profile=profile,
        hardware_tier=hardware_tier,
        library_git_sha=library_git_sha,
        benchmarks_git_sha=library_git_sha,
        environment=environment,
        datasets=sorted_datasets,
        models=tuple(sorted(config.models)),
        experiments=config.experiments,
        dataset_sha256=_resolve_dataset_sha256(sorted_datasets),
        output_root=str(output_root),
    )


def run_manifest_path(output_root: Path) -> Path:
    """Path of the run manifest under `output_root`."""
    return output_root / _RUN_MANIFEST_FILENAME


def write_run_manifest(output_root: Path, manifest: RunManifest) -> Path:
    """Persist `manifest` atomically to `output_root/run_manifest.json`.

    Uses the same write-to-temp-then-rename pattern the per-cell
    manifest does so a crashed write leaves no half-file on disk.
    Returns the absolute path the JSON landed at.
    """
    dest = run_manifest_path(output_root)
    payload = json.dumps(
        manifest.model_dump(),
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    atomic_write_bytes(payload, dest)
    logger.info("run manifest written to %s", dest)
    return dest


def load_run_manifest(output_root: Path) -> RunManifest:
    """Load and validate the run manifest under `output_root`.

    Raises:
        FileNotFoundError: no manifest at `output_root/
            run_manifest.json`.
        ValueError: the manifest file is not valid JSON (e.g., a
            partially-written file from a hard crash mid-write,
            though `atomic_write_bytes` makes that vanishingly
            unlikely).
        pydantic.ValidationError: the manifest on disk does not
            match the current `RunManifest` schema (e.g., the
            manifest was written by an older library version with
            different fields).
    """
    path = run_manifest_path(output_root)
    if not path.exists():
        raise FileNotFoundError(
            f"load_run_manifest: no run manifest at {path}; run "
            f"`python -m benchmarks.run --config <config.toml>` "
            f"first"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"load_run_manifest: manifest at {path} is not valid JSON: {exc}") from exc
    return RunManifest.model_validate(payload)
