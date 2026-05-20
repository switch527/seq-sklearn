"""CLI entry point for the benchmark harness.

Usage:

    python -m benchmarks.run --config <config.toml> --dry-run

Phase B0 ships the argparse + config-load + banner; later phases
wire `--experiment` to the per-experiment driver modules
(`benchmarks/experiments/raw_loss.py`, `ensemble.py`, etc.). The
`--dry-run` flag exits 0 after validation without running anything,
and is the path the scaffold test exercises.
"""

import argparse
import logging
import sys
import tomllib
from pathlib import Path

from pydantic import ValidationError

from benchmarks.config import BenchmarkConfig
from benchmarks.registry import list_datasets, list_models

logger = logging.getLogger(__name__)


class _ConfigLoadError(Exception):
    """Wrapping exception for the three `_load_config` failure modes
    (file missing, TOML malformed, schema failure) so the CLI main()
    can map them to a controlled exit code with a clean error message
    instead of a full Python traceback."""


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchmarks.run",
        description="seq-sklearn comparative benchmark harness",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to the TOML benchmark config",
    )
    parser.add_argument(
        "--experiment",
        choices=["raw_loss", "ensemble", "training_time", "hpo_uplift", "all"],
        default="all",
        help="experiment to run; 'all' runs every experiment in the config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="override output directory (default: BenchmarkConfig.output_dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the config and report registry counts, then exit 0",
    )
    return parser.parse_args(argv)


def _load_config(path: Path) -> BenchmarkConfig:
    """Load and validate a TOML config.

    Wraps the three observable failure modes (`FileNotFoundError`,
    `tomllib.TOMLDecodeError`, pydantic `ValidationError`) in
    `_ConfigLoadError` so the CLI can map them to a clean message
    and a controlled exit code.
    """
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise _ConfigLoadError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise _ConfigLoadError(f"config file is not valid TOML: {path}: {exc}") from exc
    try:
        return BenchmarkConfig.model_validate(raw)
    except ValidationError as exc:
        raise _ConfigLoadError(f"config schema validation failed: {path}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code.

    Exit codes:
      - ``0``: dry-run validation succeeded.
      - ``1``: config load / validation failed.
      - ``2``: non-dry-run path reached before the experiment driver
        lands in Phase B5+; the dry-run flag is the only supported
        invocation in Phase B0.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        config = _load_config(args.config)
    except _ConfigLoadError as exc:
        logger.error("%s", exc)
        return 1

    if args.experiment != "all" and args.experiment not in {e.kind for e in config.experiments}:
        logger.error(
            "--experiment=%s but the config declares only %s; pass "
            "--experiment=all to run every kind in the config, or choose "
            "one of the declared kinds",
            args.experiment,
            sorted({e.kind for e in config.experiments}),
        )
        return 1

    logger.info("seq-sklearn benchmark harness (Phase B0 scaffold)")
    logger.info("  config:         %s", args.config)
    logger.info("  experiment:     %s", args.experiment)
    logger.info(
        "  datasets cfg:   %d (registered: %d)",
        len(config.datasets),
        len(list_datasets()),
    )
    logger.info(
        "  models cfg:     %d (registered: %d)",
        len(config.models),
        len(list_models()),
    )
    logger.info(
        "  experiments:    %s",
        [e.kind for e in config.experiments],
    )

    if args.dry_run:
        logger.info("dry-run: config validates; exiting before any experiment runs")
        return 0

    # Phase B5+ will wire the per-experiment driver dispatch here.
    logger.error(
        "experiment driver not yet implemented (Phase B5+); pass --dry-run "
        "until then"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
