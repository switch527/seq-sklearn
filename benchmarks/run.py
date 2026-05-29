"""CLI entry point for the benchmark harness.

Usage:

    python -m benchmarks.run --config <config.toml> --dry-run
    python -m benchmarks.run --config <config.toml> --experiment=raw_loss

Phase B0 shipped the argparse + config-load + banner. Phases B5,
B6, B7, and B8 wire `--experiment=raw_loss`, `--experiment=ensemble`,
`--experiment=training_time`, and `--experiment=hpo_uplift` to
their drivers and write the corresponding Markdown reports next to
the manifest shards.

The `--dry-run` flag exits 0 after validation without running
anything, and is the path the scaffold test exercises.
"""

import argparse
import logging
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

# Side-effect import: pulling `benchmarks.adapters` triggers each
# adapter module's `register_model(...)` + `register_adapter_factory
# (...)` calls so the registry is populated before the experiment
# driver inspects it. The same import pattern lives in the scaffold
# test.
import benchmarks.adapters  # pyright: ignore[reportUnusedImport]
import benchmarks.datasets  # noqa: F401  # pyright: ignore[reportUnusedImport]
from benchmarks.bootstrap_manifest import aggregator_failed_sentinel_path, rollup_path
from benchmarks.config import BenchmarkConfig
from benchmarks.experiments import (
    RunEnvironment,
    build_run_environment,
    run_ensemble,
    run_ensemble_lift,
    run_hpo_uplift,
    run_raw_loss,
    run_training_time,
)
from benchmarks.manifest import atomic_write_bytes
from benchmarks.registry import list_datasets, list_models
from benchmarks.report.bootstrap_rollup import (
    RawRollupError,
    aggregate_bootstrap_rollup,
    is_rollup_enabled,
)
from benchmarks.report.ensemble import render_from_dir as render_pairwise_from_dir
from benchmarks.report.ensemble_lift import render_ensemble_lift_markdown
from benchmarks.report.hpo_uplift import render_from_dir as render_hpo_uplift_from_dir
from benchmarks.report.raw_loss import render_from_dir as render_leaderboard_from_dir
from benchmarks.report.render import REPORT_FILENAME, render_report_from_dir
from benchmarks.report.training_time import render_from_dir as render_training_time_from_dir
from benchmarks.run_manifest import (
    build_run_manifest,
    load_run_manifest,
    run_manifest_path,
    write_run_manifest,
)

logger = logging.getLogger(__name__)


# Topological dispatch order for `--experiment=all`: the report-only
# kinds (`ensemble`, `training_time`) and the HPO uplift kind all
# read the B5 manifest written by `raw_loss`, so `raw_loss` must
# run first. Alphabetical sort would put `ensemble` ahead of
# `raw_loss` and the dependent drivers would crash on an empty
# manifest.
_DISPATCH_ORDER: tuple[str, ...] = (
    "raw_loss",
    "ensemble",
    "training_time",
    "hpo_uplift",
    "ensemble_lift",
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
        choices=[
            "raw_loss",
            "ensemble",
            "training_time",
            "hpo_uplift",
            "ensemble_lift",
            "all",
        ],
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


def _dispatch_kinds(
    kinds: list[str],
    *,
    config: BenchmarkConfig,
    output_root: Path,
    env: RunEnvironment,
) -> int:
    """Run every experiment kind in `kinds` in order, writing each
    per-experiment Markdown report next to its manifest shards.

    Returns the process exit code: 0 on full success, 2 on an
    unrecognized kind (the canary branch). `ValueError` from any
    driver propagates to the caller; `main()` catches it and exits
    1 with a clean message.
    """
    for kind in kinds:
        if kind == "raw_loss":
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
            _run_bootstrap_rollup(config, env=env, output_root=output_root)
            leaderboard_md = render_leaderboard_from_dir(output_root)
            leaderboard_path = output_root / "leaderboard.md"
            leaderboard_path.write_text(leaderboard_md, encoding="utf-8")
            logger.info("leaderboard written to %s", leaderboard_path)
        elif kind == "ensemble":
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
        elif kind == "hpo_uplift":
            hpo_result = run_hpo_uplift(config, output_root=output_root, env=env)
            logger.info(
                "hpo_uplift complete: %d cells tuned, %d task-mismatch, "
                "%d proba-unavailable, %d proba-runtime-unavailable, "
                "%d quantile-followup, %d hpo-family-not-registered, "
                "%d hpo-budget-zero, %d hpo-all-trials-pruned, "
                "%d adapter-error, %d already-complete (run_id=%s)",
                hpo_result.cells_attempted,
                hpo_result.cells_skipped_task_mismatch,
                hpo_result.cells_skipped_proba_unavailable,
                hpo_result.cells_skipped_proba_runtime_unavailable,
                hpo_result.cells_skipped_quantile_followup,
                hpo_result.cells_skipped_hpo_family_not_registered,
                hpo_result.cells_skipped_hpo_budget_zero,
                hpo_result.cells_skipped_hpo_all_trials_pruned,
                hpo_result.cells_skipped_adapter_error,
                hpo_result.cells_already_complete,
                hpo_result.run_id,
            )
            hpo_md = render_hpo_uplift_from_dir(output_root)
            hpo_path = output_root / "hpo_uplift.md"
            hpo_path.write_text(hpo_md, encoding="utf-8")
            logger.info("hpo_uplift report written to %s", hpo_path)
        elif kind == "ensemble_lift":
            lift_result = run_ensemble_lift(config, output_root=output_root, env=env)
            logger.info(
                "ensemble_lift complete: %d datasets aggregated; "
                "wilcoxon p=%s holm_adj=%s (n_datasets=%d, "
                "family_size=%d) (run_id=%s)",
                len(lift_result.rows),
                lift_result.wilcoxon.p_value,
                lift_result.wilcoxon.holm_adjusted_p_value,
                lift_result.wilcoxon.n_datasets,
                lift_result.wilcoxon.family_size,
                lift_result.run_id,
            )
            lift_md = render_ensemble_lift_markdown(lift_result)
            lift_path = output_root / "ensemble_lift.md"
            atomic_write_bytes(lift_md.encode("utf-8"), lift_path)
            logger.info("ensemble_lift report written to %s", lift_path)
        else:
            logger.error(
                "experiment driver for kind=%s not yet implemented; "
                "every shipped kind (raw_loss, ensemble, training_time, "
                "hpo_uplift, ensemble_lift) is dispatched above",
                kind,
            )
            return 2
    return 0


def _run_bootstrap_rollup(
    config: BenchmarkConfig,
    *,
    env: RunEnvironment,
    output_root: Path,
) -> None:
    """B13 bootstrap-CI rollup step. Runs AFTER raw_loss completes.

    Per B13.5 + the R5 CLI-wrapper design:
    - Skips when `is_rollup_enabled(config)` is False (the
      `ExperimentSpec(kind="raw_loss", bootstrap_rollup_enabled=False)`
      opt-out path).
    - Catches `RawRollupError` and deletes any half-written
      `bootstrap_rollup.parquet` so the renderer doesn't pick up a
      partial file; the run continues with exit code 0 and the
      leaderboard falls back to the std variant with a "Bootstrap
      aggregator failed" footnote.
    """
    if not is_rollup_enabled(config):
        logger.info(
            "bootstrap_rollup: skipped (no raw_loss ExperimentSpec has "
            "`bootstrap_rollup_enabled=True`)"
        )
        return
    if not run_manifest_path(output_root).exists():
        logger.warning(
            "bootstrap_rollup: skipped (run_manifest.json absent at %s)",
            output_root,
        )
        return
    try:
        manifest = load_run_manifest(output_root)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        logger.warning(
            "bootstrap_rollup: skipped (failed to load run_manifest.json: %s: %s)",
            type(exc).__name__,
            exc,
        )
        return
    try:
        rows = aggregate_bootstrap_rollup(
            config, output_root=output_root, env=env, manifest=manifest
        )
        # Successful aggregate: unlink any stale failure sentinel
        # from a prior run so the renderer surfaces the CI variant
        # rather than the std + footnote (B13.0 R2 arch-C1 close).
        stale = aggregator_failed_sentinel_path(output_root)
        if stale.exists():
            stale.unlink(missing_ok=True)
        logger.info(
            "bootstrap_rollup: %d rollup rows written to %s",
            len(rows),
            rollup_path(output_root),
        )
    except RawRollupError as exc:
        logger.warning(
            "bootstrap_rollup: aggregator failed (%s); deleting any partial "
            "rollup shard. The leaderboard will fall back to the std variant.",
            exc,
        )
        path = rollup_path(output_root)
        if path.exists():
            path.unlink(missing_ok=True)
        # Drop a sentinel file so the renderer surfaces the
        # "Bootstrap aggregator failed: <class>" footnote
        # (Gemini-C2 wrapper case + R1 code-I1 wire-up).
        sentinel = aggregator_failed_sentinel_path(output_root)
        sentinel.write_text(type(exc).__name__, encoding="utf-8")


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
        not yet shipped. Every kind in the `ExperimentKind`
        literal currently dispatches; this branch is the canary
        for future kinds.
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

    # Build the RunEnvironment ONCE so every experiment kind in this
    # invocation shares the same run_id. The per-cell ResultRow.run_id
    # then matches the RunManifest.run_id and the B9 reproducibility
    # contract holds (the manifest's run_id is the join key into the
    # `results/` shards).
    fresh_env = build_run_environment(profile="standard")

    # Resume-after-crash: if `output_root/run_manifest.json` already
    # exists with `completed_at_utc=None`, the previous invocation
    # crashed mid-run and per-cell shards on disk carry its run_id.
    # Issuing a fresh run_id here would break the
    # `ResultRow.run_id == RunManifest.run_id` join the
    # methodology page names as the cross-layer key. Resume the
    # prior manifest's run_id + started_at_utc; the fresh
    # environment otherwise replaces it (machine config may have
    # changed across the crash boundary). A manifest with
    # `completed_at_utc` populated indicates a previous CLEAN
    # exit; in that case we also reuse the run_id, treating
    # re-runs as an idempotent append (the per-cell sentinel
    # contract already de-dupes work).
    env: RunEnvironment
    started_at_utc: str
    if run_manifest_path(output_root).exists():
        try:
            prior = load_run_manifest(output_root)
        except (ValueError, FileNotFoundError) as exc:
            logger.error(
                "existing run_manifest.json failed to load (%s); "
                "refusing to overwrite. Delete the file or pass a "
                "fresh --output to start over.",
                exc,
            )
            return 1
        logger.info(
            "resuming run_id=%s (started_at_utc=%s) from existing manifest at %s",
            prior.run_id,
            prior.started_at_utc,
            run_manifest_path(output_root),
        )
        env = RunEnvironment(
            run_id=prior.run_id,
            library_git_sha=fresh_env.library_git_sha,
            hardware_tier=fresh_env.hardware_tier,
            profile=fresh_env.profile,
        )
        started_at_utc = prior.started_at_utc
    else:
        env = fresh_env
        started_at_utc = datetime.now(UTC).isoformat()

    # Write the run manifest BEFORE any experiment touches the
    # filesystem. A crashed run still leaves a manifest of intent
    # on disk; the post-experiment rewrite (below) stamps the
    # completion timestamp.
    manifest = build_run_manifest(
        config,
        run_id=env.run_id,
        library_git_sha=env.library_git_sha,
        profile=env.profile,
        hardware_tier=env.hardware_tier,
        output_root=output_root,
        started_at_utc=started_at_utc,
    )
    write_run_manifest(output_root, manifest)

    requested = args.experiment
    declared_kinds = {e.kind for e in config.experiments}
    if requested == "all":
        kinds = [k for k in _DISPATCH_ORDER if k in declared_kinds]
    else:
        kinds = [requested]

    # Cross-driver `ValueError` -> exit 1 (B7 + B8 Gemini-deferral
    # carryover). Every driver raises `ValueError` on the "config
    # declares no <kind> experiment" / "cache_dir is None" / "empty
    # manifest" failure modes. Without this catch, the CLI emits a
    # Python traceback; with it, the user sees a one-line message
    # and a clean exit-1 (matches the config-load-failure exit code).
    # The catch ALSO wraps the post-loop completion-stamp + report
    # assembly so a crash there (e.g., a corrupt manifest the
    # `render_report_from_dir` reload tripped on) doesn't leak a
    # traceback either.
    try:
        rc = _dispatch_kinds(kinds, config=config, output_root=output_root, env=env)
        if rc != 0:
            return rc
        # Stamp the completion timestamp atomically; a crash between
        # the dispatch and this point leaves the original manifest
        # of intent on disk for the next invocation to resume.
        completed = build_run_manifest(
            config,
            run_id=env.run_id,
            library_git_sha=env.library_git_sha,
            profile=env.profile,
            hardware_tier=env.hardware_tier,
            output_root=output_root,
            started_at_utc=manifest.started_at_utc,
            completed_at_utc=datetime.now(UTC).isoformat(),
            environment=manifest.environment,
        )
        write_run_manifest(output_root, completed)
        # Assemble the cross-experiment `report.md` atomically.
        # `report.md` is fully derived from `run_manifest.json` +
        # the per-experiment Markdown files, so a crash between the
        # manifest rewrite and this write is benign: the next clean
        # invocation re-derives the same `report.md`.
        report_md = render_report_from_dir(output_root)
        report_path = output_root / REPORT_FILENAME
        atomic_write_bytes(report_md.encode("utf-8"), report_path)
        logger.info("assembled report written to %s", report_path)
    except ValueError as exc:
        logger.error("benchmark run failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
