"""B2.3: GATED dataset loaders raise `GatedDatasetUnavailableError`
when the archive is absent.

The test points the loader at a fresh `tmp_path` cache root with
nothing under `archives/<dataset>/` and asserts the typed raise.
Verifies the error message includes the source URI (the manual
setup link) and the expected cache path (so the user knows where
to drop the file).

Phase B1 covers Amex; the remaining GATED loaders land in
follow-up phases and will plug into the same parametrize.
"""

from pathlib import Path

import benchmarks.datasets  # noqa: F401  # pyright: ignore[reportUnusedImport] - side-effect: registers loaders
import pytest
from benchmarks._io.integrity import GatedDatasetUnavailableError
from benchmarks.registry import get_dataset, get_loader

_GATED_DATASETS_IN_THIS_PHASE: tuple[str, ...] = ("amex_default",)


@pytest.mark.parametrize("name", _GATED_DATASETS_IN_THIS_PHASE)
def test_gated_loader_raises_when_archive_absent(
    name: str, tmp_path: Path
) -> None:
    spec = get_dataset(name)
    assert spec.access_tier == "GATED", (
        f"dataset {name!r} is parametrized as GATED but its spec "
        f"declares access_tier={spec.access_tier!r}"
    )
    loader = get_loader(name)
    with pytest.raises(GatedDatasetUnavailableError) as excinfo:
        loader(tmp_path)
    message = str(excinfo.value)
    # The error names the source URI so the user can find the
    # acceptance click-through.
    assert spec.source_uri in message, (
        f"GATED loader for {name!r} must name the source URI in its "
        f"error message; got:\n{message}"
    )
    # The error names the expected cache path so the user knows
    # where to drop the file.
    assert str(tmp_path) in message, (
        f"GATED loader for {name!r} must name the expected cache path; "
        f"got:\n{message}"
    )
    assert spec.archive_basename in message, (
        f"GATED loader for {name!r} must name the archive basename; "
        f"got:\n{message}"
    )
