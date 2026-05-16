"""Safetensors + JSON save / load primitives (per architecture A17).

A persisted model is a directory holding ``weights.safetensors`` (a
tensor-only safetensors archive) and ``state.json`` (the human-readable
config / fit-state / metadata block). The library never calls
``torch.load``; the load path is ``weights_only`` by construction since
safetensors carries no executable payload.

Schema migrations are registered in :data:`MIGRATIONS` keyed by
``(from_version, to_version)``. The registry is empty in v1, so
:func:`_migrate` short-circuits on a v1 checkpoint. The first entry
lands when v1.1 needs to migrate v1 saves.
"""

import json
import logging
import warnings
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from seq_sklearn import __version__
from seq_sklearn.errors import PredictionError

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "MIGRATIONS",
    "OLDEST_SUPPORTED_SCHEMA_VERSION",
    "Migration",
    "load_weights_and_state",
    "save_weights_and_state",
]

logger = logging.getLogger(__name__)

type WeightDict = dict[str, torch.Tensor]
type StateDict = dict[str, object]
type Migration = Callable[[WeightDict, StateDict], tuple[WeightDict, StateDict]]

CURRENT_SCHEMA_VERSION: int = 1
OLDEST_SUPPORTED_SCHEMA_VERSION: int = 1  # v1 supports only itself
assert OLDEST_SUPPORTED_SCHEMA_VERSION <= CURRENT_SCHEMA_VERSION, (
    "OLDEST_SUPPORTED_SCHEMA_VERSION must be <= CURRENT_SCHEMA_VERSION"
)

# Registry: (from_version, to_version) -> migration function.
# Empty in v1; first entry lands when v1.1 needs to migrate v1 saves.
MIGRATIONS: dict[tuple[int, int], Migration] = {}

_WEIGHTS_FILE = "weights.safetensors"
_STATE_FILE = "state.json"


def _migrate(weights: WeightDict, state: StateDict) -> tuple[WeightDict, StateDict]:
    """Step ``(weights, state)`` forward until the schema is current.

    Advances ``state['schema_version']`` one registered step at a time
    until it equals :data:`CURRENT_SCHEMA_VERSION`. Each step must mutate
    ``schema_version`` to a strictly larger value; a step that does not
    advance it would spin the loop forever, so the post-step value is
    checked and a non-advancing step is rejected.

    Raises:
        PredictionError: The checkpoint schema is newer than this
            library supports, older than the oldest supported schema,
            has no registered migration step, or a registered step did
            not advance ``schema_version``.
    """
    raw_src = state.get("schema_version", 0)
    src = raw_src if isinstance(raw_src, int) else 0
    if src > CURRENT_SCHEMA_VERSION:
        raise PredictionError(
            f"checkpoint schema {src} newer than library "
            f"v1 (max {CURRENT_SCHEMA_VERSION}); upgrade seq-sklearn"
        )
    if src < OLDEST_SUPPORTED_SCHEMA_VERSION:
        raise PredictionError(
            f"checkpoint schema {src} older than oldest supported "
            f"({OLDEST_SUPPORTED_SCHEMA_VERSION})"
        )
    while src < CURRENT_SCHEMA_VERSION:
        try:
            step = MIGRATIONS[(src, src + 1)]
        except KeyError:
            raise PredictionError(
                f"no migration registered from schema {src} to {src + 1}"
            ) from None
        weights, state = step(weights, state)
        post = state.get("schema_version", src)
        if not isinstance(post, int) or post <= src:
            raise PredictionError(
                f"migration step ({src}, {src + 1}) did not advance "
                f"schema_version from schema {src} to {src + 1} "
                f"(got {post}, expected > {src}); the migration callable "
                f"must mutate state['schema_version'] to a strictly larger "
                f"value before returning"
            )
        src = post
    return weights, state


def save_weights_and_state(
    path: str | Path,
    weights: WeightDict,
    state: StateDict,
) -> None:
    """Write ``weights.safetensors`` and ``state.json`` under ``path``.

    Creates ``path`` (and parents) if absent. ``weights`` is a flat
    string-keyed tensor dict; ``state`` is the JSON-serializable config /
    fit-state / metadata block.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(path / _WEIGHTS_FILE))
    (path / _STATE_FILE).write_text(json.dumps(state, indent=2))


def load_weights_and_state(
    path: str | Path,
) -> tuple[WeightDict, StateDict]:
    """Read and schema-migrate a persisted ``(weights, state)`` pair.

    Loads the safetensors archive and the JSON state block, warns once
    when the persisting ``seq_sklearn_version`` differs from the
    installed one, then runs :func:`_migrate` before returning so the
    caller always consumes a current-schema pair.

    Raises:
        PredictionError: The schema cannot be migrated to the current
            version (propagated from :func:`_migrate`).
    """
    path = Path(path)
    weights = load_file(str(path / _WEIGHTS_FILE))
    state = json.loads((path / _STATE_FILE).read_text())
    persisted_version = state.get("seq_sklearn_version")
    if persisted_version != __version__:
        warnings.warn(
            f"checkpoint version mismatch: saved with seq_sklearn "
            f"{persisted_version}, loading with {__version__}",
            UserWarning,
            stacklevel=2,
        )
    return _migrate(weights, state)
