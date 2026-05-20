"""Per-loader offline smoke tests for the OPEN datasets.

Each test constructs a synthetic mini-archive in ``tmp_path``,
recomputes its SHA-256, monkeypatches that hash into the spec, and
runs the loader. The output is asserted in F2 contract shape: one
row per ``(entity_id, time)`` with the spec's feature columns and
a typed target.

The synthetic payloads are intentionally tiny (a handful of
entities, a handful of cycles) so the suite stays in the cheap CPU
budget. The real archives are never exercised in CI.
"""

import hashlib
import io
import zipfile
from pathlib import Path

import benchmarks.datasets  # noqa: F401  # pyright: ignore[reportUnusedImport] - side-effect: registers loaders
import pytest
from benchmarks._io.cache import archive_path
from benchmarks.datasets import c_mapss_fd001 as c_mapss_module
from benchmarks.datasets._base import PanelDataset


def _make_cmapss_fd001_mini_archive() -> bytes:
    """Build a 5-engine x 4-cycle synthetic FD001 archive with the
    24-column whitespace-separated shape the loader expects.

    Returns the raw zip bytes; the test writes them to the cache
    path the loader looks up and patches the spec's integrity hash
    to match.
    """
    rows: list[str] = []
    for engine_id in range(1, 6):  # five engines
        for cycle in range(1, 5):  # four cycles each
            cols = [str(engine_id), str(cycle)]
            # Three op-setting columns; mock a degradation pattern.
            cols += [f"{0.1 * cycle:.4f}"] * 3
            # Twenty-one sensor columns; mock monotonic drift.
            cols += [f"{cycle * (s + 1) * 0.01:.4f}" for s in range(21)]
            rows.append(" ".join(cols))
    payload = ("\n".join(rows) + "\n").encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("train_FD001.txt", payload)
    return buf.getvalue()


def test_c_mapss_fd001_loads_from_synthetic_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end loader smoke: place the synthetic archive at the
    expected cache path, point the spec's integrity hash at its
    SHA-256, and assert the loader emits an F2-shape panel."""
    spec = c_mapss_module._SPEC  # pyright: ignore[reportPrivateUsage]
    archive_bytes = _make_cmapss_fd001_mini_archive()
    sha = hashlib.sha256(archive_bytes).hexdigest()

    target = archive_path(tmp_path, spec.name, spec.archive_basename)
    target.write_bytes(archive_bytes)

    patched_spec = spec.model_copy(update={"integrity_sha256": sha})
    monkeypatch.setattr(c_mapss_module, "_SPEC", patched_spec)

    out = c_mapss_module.load(tmp_path)
    assert isinstance(out, PanelDataset)
    # Five engines x four cycles = twenty rows.
    assert len(out.panel) == 20
    assert len(out.y) == 20
    # F2 panel columns: entity, time, then features.
    assert list(out.panel.columns[:2]) == ["entity_id", "cycle"]
    assert "op_setting_1" in out.panel.columns
    assert "sensor_21" in out.panel.columns
    # RUL is correct for engine 1: cycles 1-4, max=4, RUL=[3,2,1,0].
    e1_rows = out.panel[out.panel["entity_id"] == 1]
    e1 = e1_rows.sort_values("cycle")  # pyright: ignore[reportCallIssue]
    assert list(out.y[e1.index]) == [3.0, 2.0, 1.0, 0.0]


def test_c_mapss_fd001_integrity_mismatch_raises(
    tmp_path: Path,
) -> None:
    """The cache layer's integrity check fires when the archive
    bytes don't match the spec's hash. The byte-flipped archive
    test from B2.2."""
    from benchmarks._io.integrity import DatasetIntegrityError

    spec = c_mapss_module._SPEC  # pyright: ignore[reportPrivateUsage]
    target = archive_path(tmp_path, spec.name, spec.archive_basename)
    target.write_bytes(b"this is not a valid c-mapss archive")
    # The spec's integrity_sha256 is the deadbeef placeholder; the
    # archive's real SHA differs. Loader must raise.
    with pytest.raises(DatasetIntegrityError):
        c_mapss_module.load(tmp_path)


def test_c_mapss_fd001_missing_archive_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """No archive in the cache and no auto-download: the loader
    surfaces ``FileNotFoundError`` from the cache layer rather
    than silently no-op'ing."""
    with pytest.raises(FileNotFoundError):
        c_mapss_module.load(tmp_path)
