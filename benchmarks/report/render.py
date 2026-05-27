"""Cross-experiment report assembly (Phase B9 / B8.1).

Reads the four per-experiment Markdown reports + the run manifest
under `output_root` and produces the assembled `report.md`
deliverable described by the impl plan:

- One-paragraph executive summary that names the run profile, the
  hardware tier, and the SHA pair (library + benchmarks).
- Run-time metadata block referencing the manifest path + key
  environment fields.
- Each per-experiment report rendered as a section in the order
  B5 -> B6 -> B7 -> B8 (matches the impl plan's deliverable
  ordering).
- A methodology footer that links the static index page under
  `docs/explanation/benchmarks/`.

Missing per-experiment files (e.g., the run was `--experiment=
raw_loss` only) render as a one-line "_<kind> not run in this
manifest_" notice so the assembled report is always shaped the
same way regardless of which subset of experiments fired.
"""

import logging
from pathlib import Path

from benchmarks.run_manifest import RunManifest, load_run_manifest, run_manifest_path

logger = logging.getLogger(__name__)


__all__ = [
    "REPORT_FILENAME",
    "assemble_report",
    "render_report_from_dir",
]


REPORT_FILENAME = "report.md"


# Per-experiment Markdown filenames the assembler reads, in the
# canonical B5 -> B6 -> B7 -> B8 order. Each maps to the section
# heading the assembled report uses.
_EXPERIMENT_FILES: tuple[tuple[str, str], ...] = (
    ("leaderboard.md", "Raw loss leaderboard (B5 / B6.1)"),
    ("pairwise.md", "Ensemble complementarity (B6 / B6.2)"),
    ("ensemble_lift.md", "Ensemble lift Wilcoxon (B11 / B6.2.5)"),
    ("training_time.md", "Training-time table (B7 / B6.3)"),
    ("hpo_uplift.md", "HPO uplift + Friedman/Holm (B8 / B6.4)"),
)


def _render_executive_summary(manifest: RunManifest) -> str:
    """One-paragraph executive summary at the top of report.md."""
    env = manifest.environment
    completed = manifest.completed_at_utc or "in-progress"
    return (
        f"Run `{manifest.run_id}` on profile `{manifest.profile}`, "
        f"hardware tier `{manifest.hardware_tier}`, started "
        f"`{manifest.started_at_utc}` (completed `{completed}`). "
        f"Library + benchmarks SHA `{manifest.library_git_sha}`. "
        f"Platform `{env.platform}`. "
        f"Datasets ({len(manifest.datasets)}): "
        f"`{', '.join(manifest.datasets) or 'none'}`. "
        f"Models ({len(manifest.models)}): "
        f"`{', '.join(manifest.models) or 'none'}`."
    )


def _render_manifest_block(manifest: RunManifest, *, manifest_relpath: str) -> str:
    """Markdown block summarizing the env fingerprint + manifest
    pointer (the source-of-truth lives in the JSON; the report
    surfaces the key fields)."""
    env = manifest.environment
    pkg_lines: list[str] = []
    for name, ver in sorted(env.package_versions.items()):
        if ver is None:
            pkg_lines.append(f"- `{name}`: _not installed_")
        else:
            pkg_lines.append(f"- `{name}`: `{ver}`")
    parts = [
        "## Run metadata",
        "",
        f"Manifest: [`{manifest_relpath}`]({manifest_relpath})",
        "",
        f"- Python: `{env.python_version}`",
        f"- CPU: `{env.cpu_model or 'unknown'}`",
        f"- GPU: `{env.gpu_model or 'none'}`",
        f"- CUDA runtime: `{env.cuda_runtime or 'none'}`",
        f"- CUDA driver: `{env.cuda_driver or 'unknown'}`",
        "",
        "### Package versions",
        "",
        *pkg_lines,
        "",
    ]
    return "\n".join(parts)


def _render_experiment_section(output_root: Path, filename: str, heading: str) -> str:
    """Render one per-experiment section.

    Missing files surface as a one-line notice; present files are
    quoted verbatim under their own `##`-level heading.
    """
    path = output_root / filename
    if not path.exists():
        return f"## {heading}\n\n_`{filename}` not present in this manifest._\n"
    body = path.read_text(encoding="utf-8")
    # The per-experiment renderers emit their own `# Title` line
    # at the top; the assembler replaces it with the section
    # heading so the assembled report has a single `# Report`
    # level-1 line plus level-2 section headings.
    body = body.lstrip()
    if body.startswith("# "):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.lstrip()
    return f"## {heading}\n\n{body}\n"


def _render_methodology_footer() -> str:
    """Footer linking the static methodology index page.

    The link is the in-tree doc path (no leading `..` or absolute
    URL); how a reader follows it depends on where `report.md`
    lands in their checkout, but the path itself stays stable
    relative to the repo root so a `find docs/explanation/
    benchmarks/index.md` from anywhere resolves it.
    """
    return (
        "## Methodology\n\n"
        "Per-experiment design notes live in the repo at "
        "`docs/explanation/benchmarks/index.md`. Each per-experiment "
        "section above quotes its renderer's Markdown verbatim; "
        "the full per-cell `ResultRow` shards remain under "
        "`results/` for downstream audit.\n"
    )


def assemble_report(
    output_root: Path,
    *,
    manifest: RunManifest | None = None,
) -> str:
    """Assemble `report.md` content from `output_root`.

    `manifest` defaults to the result of `load_run_manifest
    (output_root)`; the caller can override to inject a manifest
    for testing or to re-render from a partial state. When the
    manifest file is missing AND `manifest is None`, raises
    `FileNotFoundError` (the caller must run something before the
    assembler can attribute the report).
    """
    if manifest is None:
        manifest = load_run_manifest(output_root)
    parts: list[str] = [
        "# seq-sklearn benchmark report",
        "",
        _render_executive_summary(manifest),
        "",
        _render_manifest_block(manifest, manifest_relpath=run_manifest_path(output_root).name),
    ]
    for filename, heading in _EXPERIMENT_FILES:
        parts.append(_render_experiment_section(output_root, filename, heading))
    parts.append(_render_methodology_footer())
    return "\n".join(parts)


def render_report_from_dir(output_root: Path) -> str:
    """Convenience wrapper: load manifest + assemble report.

    Mirrors the per-experiment `render_from_dir` shape so the CLI
    can call it the same way it calls the leaderboard / pairwise /
    training-time / HPO-uplift renderers.
    """
    return assemble_report(output_root)
