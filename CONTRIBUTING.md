# Contributing to seq-sklearn

Thanks for the interest. The project is in early development; the
contributor workflow below is provisional and may change before v1.

## Local setup

```
uv venv --python 3.14 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Optional extras: `[onnx]`, `[mlflow]`, `[wandb]`, `[docs]`.

## Gates

Every PR must pass:

```
ruff check .
ruff format --check .
pyright
pytest -m "not slow and not perf and not gpu" --cov
```

CI runs all four on every PR plus a wheel-install smoke test and a
snapshot-marker guard. See `.github/workflows/pr.yml`.

## Review workflow

Non-trivial designs and code changes go through a multi-agent review
loop documented in `CLAUDE.md`:

- `/design-review <doc>` iterates the Claude swarm on a design doc.
- `/review [diff-spec]` iterates the swarm on a code diff.
- `/gemini-final-pass <kind> <target>` runs Gemini's reviewer after the
  swarm reaches consensus. Gemini capacity is scarce; reserved for
  post-consensus.

See `docs/requirements.md` and `docs/architecture.md` for the v1
contract and design; `CHANGELOG.md` records what shipped.

## Commit conventions

- Conventional-Commits-style prefix is encouraged but not enforced.
- Commits touching `tests/_snapshots/` MUST carry a `SNAPSHOT_REVIEWED:`
  marker line in the commit message (the pre-commit hook and CI guard
  enforce this).

## Pre-commit hooks

The `.pre-commit-config.yaml` is shipped but not autoinstalled. Enable
locally with:

```
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

## Style

See the anti-tell rules in `CLAUDE.md` and `.claude/rules/style.md`.
