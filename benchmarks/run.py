"""CLI entry point for the benchmark harness.

Usage:

    python -m benchmarks.run --config <config.toml> --dry-run
    python -m benchmarks.run --config <config.toml> --experiment=raw_loss

Phase B0 shipped the argparse + config-load + banner. Phases B5,
B6, and B7 wire `--experiment=raw_loss`, `--experiment=ensemble`,
and `--experiment=training_time` to their drivers and write the
corresponding Markdown reports next to the manifest shards.
Phase B8 (`hpo_uplift`) is the last dispatch arm; until it ships,
`--experiment=hpo_uplift` exits 2.

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
from benchmarks.experiments import (
    build_run_environment,
    run_ensemble,
    run_raw_loss,
    run_training_time,
)
from benchmarks.registry import list_datasets, list_models
from benchmarks.report.ensemble import render_from_dir as render_pairwise_from_dir
from benchmarks.report.raw_loss import render_from_dir as render_leaderboard_from_dir
from benchmarks.report.training_time import render_from_dir as render_training_time_from_dir

logger = logging.getLogger(__name__)


# Topological dispatch order for `--experiment=all`: report-only
# experiments (`ensemble`, `training_time`) read the B5 manifest
# written by `raw_loss`, so `raw_loss` must run first. Alphabetical
# sort would put `ensemble` ahead of `raw_loss` and the dependent
# drivers would crash on an empty manifest. `hpo_uplift` is here
# as a placeholder; B8 wires its driver.
_DISPATCH_ORDER: tuple[str, ...] = (
    "raw_loss",
    "ensemble",
    "training_time",
    "hpo_uplift",
)


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
      - ``0``: dry-run validation succeeded, or every requested
        experiment kind completed (the relevant Markdown report
        is at `output_root/<kind>.md`).
      - ``1``: config load / validation failed.
      - ``2``: an experiment kind was requested whose driver has
        not yet shipped (only `hpo_uplift` today).
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

    logger.info("seq-sklearn benchmark harness")
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
    if requested == "all":
        kinds = [k for k in _DISPATCH_ORDER if k in declared_kinds]
    else:
        kinds = [requested]

    for kind in kinds:
        if kind == "raw_loss":
            env = build_run_environment(profile="standard")
            result = run_raw_loss(config, output_root=output_root, env=env)
            logger.info(
                "raw_loss complete: %d cells run, %d task-mismatch, "
                "%d proba-unavailable, %d proba-runtime-unavailable, "
                "%d quantile-followup, %d adapter-error, %d already-complete "
                "(run_id=%s)",
                result.cells_attempted,
                result.cells_skipped_task_mismatch,
                result.cells_skipped_proba_unavailable,
                result.cells_skipped_proba_runtime_unavailable,
                result.cells_skipped_quantile_followup,
                result.cells_skipped_adapter_error,
                result.cells_already_complete,
                result.run_id,
            )
            leaderboard_md = render_leaderboard_from_dir(output_root)
            leaderboard_path = output_root / "leaderboard.md"
            leaderboard_path.write_text(leaderboard_md, encoding="utf-8")
            logger.info("leaderboard written to %s", leaderboard_path)
        elif kind == "ensemble":
            env = build_run_environment(profile="standard")
            pair_result = run_ensemble(config, output_root=output_root, env=env)
            logger.info(
                "ensemble complete: %d pairs run, %d predictions-missing, "
                "%d empty-join, %d already-complete (run_id=%s)",
                pair_result.pairs_attempted,
                pair_result.pairs_skipped_predictions_missing,
                pair_result.pairs_skipped_empty_join,
                pair_result.pairs_already_complete,
                pair_result.run_id,
            )
            pairwise_md = render_pairwise_from_dir(output_root)
            pairwise_path = output_root / "pairwise.md"
            pairwise_path.write_text(pairwise_md, encoding="utf-8")
            logger.info("pairwise report written to %s", pairwise_path)
        elif kind == "training_time":
            env = build_run_environment(profile="standard")
            tt_result = run_training_time(config, output_root=output_root, env=env)
            logger.info(
                "training_time complete: %d groups evaluated, %d fully skipped (run_id=%s)",
                tt_result.groups_evaluated,
                tt_result.groups_fully_skipped,
                tt_result.run_id,
            )
            training_time_md = render_training_time_from_dir(output_root)
            training_time_path = output_root / "training_time.md"
            training_time_path.write_text(training_time_md, encoding="utf-8")
            logger.info("training-time report written to %s", training_time_path)
        else:
            logger.error(
                "experiment driver for kind=%s not yet implemented; "
                "only `raw_loss`, `ensemble`, and `training_time` "
                "ship today (Phase B8 brings `hpo_uplift`)",
                kind,
            )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
