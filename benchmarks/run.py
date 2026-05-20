"""CLI entry point for the benchmark harness.

Usage:

    python -m benchmarks.run --config <config.toml> --dry-run
    python -m benchmarks.run --config <config.toml> --experiment=raw_loss

Phase B0 shipped the argparse + config-load + banner; Phase B5
wires `--experiment=raw_loss` to the raw-loss experiment driver
and writes a Markdown leaderboard next to the manifest shards.
Phases B6-B8 will add the remaining dispatch arms.

The `--dry-run` flag exits 0 after validation without running
anything, and is the path the scaffold test exercises.
"""

import argparse
import logging
import sys
import tomllib
from pathlib import Path

from pydantic import ValidationError

# Side-effect import: pulling `benchmarks.adapters` triggers each
# adapter module's `register_model(...)` + `register_adapter_factory
# (...)` calls so the registry is populated before the experiment
# driver inspects it. The same import pattern lives in the scaffold
# test.
import benchmarks.adapters  # pyright: ignore[reportUnusedImport]
import benchmarks.datasets  # noqa: F401  # pyright: ignore[reportUnusedImport]
from benchmarks.config import BenchmarkConfig
from benchmarks.experiments import build_run_environment, run_raw_loss
from benchmarks.registry import list_datasets, list_models
from benchmarks.report.raw_loss import render_from_dir

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
      - ``0``: dry-run validation succeeded, or `--experiment=raw_loss`
        completed (the leaderboard is at `output_root/leaderboard.md`).
      - ``1``: config load / validation failed.
      - ``2``: an experiment kind other than `raw_loss` was requested
        but not yet implemented (Phase B6+).
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

    logger.info("seq-sklearn benchmark harness (Phase B5 raw-loss)")
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

    output_root: Path = args.output if args.output is not None else config.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    requested = args.experiment
    declared_kinds = {e.kind for e in config.experiments}
    kinds = sorted(declared_kinds) if requested == "all" else [requested]

    for kind in kinds:
        if kind != "raw_loss":
            logger.error(
                "experiment driver for kind=%s not yet implemented "
                "(Phase B6+); only `raw_loss` ships in Phase B5",
                kind,
            )
            return 2
        env = build_run_environment(profile="standard")
        result = run_raw_loss(config, output_root=output_root, env=env)
        logger.info(
            "raw_loss complete: %d cells run, %d task-mismatch, "
            "%d quantile-followup, %d adapter-error, %d already-complete "
            "(run_id=%s)",
            result.cells_attempted,
            result.cells_skipped_task_mismatch,
            result.cells_skipped_quantile_followup,
            result.cells_skipped_adapter_error,
            result.cells_already_complete,
            result.run_id,
        )
        leaderboard_md = render_from_dir(output_root)
        leaderboard_path = output_root / "leaderboard.md"
        leaderboard_path.write_text(leaderboard_md, encoding="utf-8")
        logger.info("leaderboard written to %s", leaderboard_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
