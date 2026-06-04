# Setup on a fresh Ubuntu 26.04 LTS machine

End-to-end instructions for bootstrapping `seq-sklearn` on a clean
machine. Written so an agent can follow it without prior context.

The repo has two parts:
- **`seq-sklearn`** — the library + benchmark harness + tests (this repo).
- **`seq-sklearn-meta`** — agent configuration (`CLAUDE.md`, `.claude/`,
  `GEMINI.md`, etc.). Lives in a separate repo and is mounted at `.meta/`
  via symlinks. Kept separate so this repo's git history stays clean of
  agent-config churn.

Both repos belong to `https://github.com/switch527/`.

## 1. System prerequisites

```bash
# Git + curl + build essentials (Python may need to compile native extensions)
sudo apt update && sudo apt install -y \
    git curl build-essential pkg-config \
    libssl-dev libffi-dev \
    unzip
```

For GPU work (the TFT family uses CUDA), install NVIDIA drivers from the
Ubuntu graphics-drivers PPA. The hardware this was developed on:

- NVIDIA RTX PRO 6000 Blackwell Workstation (96 GB VRAM)
- Driver `590.48.01` or newer
- CUDA 13.0 (PyTorch 2.12 ships with cu130 wheels — no separate CUDA
  toolkit install needed for runtime; install only if you want to build
  custom kernels)

CPU-only also works; the test suite runs without CUDA and the benchmark
harness routes the TSC + GBM families through CPU.

## 2. Install `uv`

`uv` is the package manager + Python installer + venv tool used by this
repo. One-shot install:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Restart shell or `source ~/.bashrc` so PATH picks up ~/.local/bin/uv
```

`uv --version` should print `>= 0.5`.

## 3. Clone both repos

The main repo first; meta into `.meta/` inside it:

```bash
git clone https://github.com/switch527/seq-sklearn.git
cd seq-sklearn
git clone https://github.com/switch527/seq-sklearn-meta.git .meta
```

## 4. Wire the agent config symlinks

The meta repo's bootstrap script creates symlinks at the parent root so
Claude Code and Gemini CLI discover `CLAUDE.md`, `.claude/`, `GEMINI.md`,
`.gemini/`, `.geminiignore` at the standard locations:

```bash
bash .meta/bootstrap.sh
```

Idempotent and re-runnable. The script aborts rather than clobber if any
of those paths exist as non-symlinks.

## 5. Install Python + create the venv

The project targets `>=3.12,<3.15`. The development hardware uses Python
3.14.

```bash
uv python install 3.14
uv venv --python 3.14
source .venv/bin/activate
```

## 6. Install the project + extras

The repo carries a committed `uv.lock` (652 KB) that pins every
transitive dependency. `uv sync` reproduces the environment exactly.

```bash
# All extras except the optional ones that need manual install
uv sync --extra dev --extra benchmarks --extra docs --extra onnx
```

Available extras (`pyproject.toml [project.optional-dependencies]`):
- **`dev`** — ruff, pyright, pytest, hypothesis, pre-commit
- **`benchmarks`** — pyarrow, lightgbm, xgboost, catboost, openml
- **`docs`** — sphinx + theme + numpydoc + gallery
- **`onnx`** — onnx + onnxruntime + onnxscript
- **`mlflow`**, **`wandb`** — experiment trackers (optional)

### Manual extras (not in `pyproject.toml`)

Two libraries can't go through `pyproject.toml` for environmental reasons.
Install only if you need them:

```bash
# Required for the TSC adapter family (ROCKET, MultiRocket, Catch22)
# Excluded from pyproject because aeon 0.11.x pins scikit-learn < 1.6 and
# the library requires >= 1.6. aeon 1.4+ resolves the conflict.
uv pip install "aeon>=1.4"

# Required for the amex_default loader (benchmarks/datasets/amex_default.py)
uv pip install kaggle
```

## 7. Configure Kaggle (optional, only if running amex_default)

Sign in at `kaggle.com` → Account → API → **Create New Token**. Kaggle's
2026 API token format starts with `KGAT_`. Save it to
`~/.kaggle/access_token`:

```bash
mkdir -p ~/.kaggle
echo "KGAT_your_token_here" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Verify: `kaggle competitions list` should return a list, not an auth error.

## 8. Verify the install

```bash
# Targeted gate: benchmarks suite (~3 minutes)
uv run pytest tests/benchmarks/ -q

# Full library suite (~20-30 minutes on CPU; faster on GPU)
uv run pytest -q
```

Expected: 1132+ passed, 2 skipped (aeon-gated TSC smokes if aeon isn't
installed; gpu-marked perf tests).

Lint + type check:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

All three should pass cleanly on `main`.

## 9. Optional: dataset caches

The benchmark harness uses `.cache/benchmarks/` for downloaded archives
and generated panel parquets. These are gitignored. Cleanest paths:

- **Synthetic datasets** (binary, multiclass, regression, imbalanced
  50:1, imbalanced 100:1) — no download needed; loaded from the
  `SyntheticPanelGenerator` DGP. Just work.
- **C-MAPSS FD001-FD004** — `curl` the archive once:
  ```bash
  mkdir -p .cache/benchmarks/archives/c_mapss_fd001
  curl -sSL -o /tmp/cmapss_outer.zip \
      "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
  unzip -p /tmp/cmapss_outer.zip "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip" \
      > .cache/benchmarks/archives/c_mapss_fd001/CMAPSSData.zip
  ```
- **UEA datasets** (`uea_basic_motions`, `uea_heartbeat`, `uea_face_detection`,
  `uea_motor_imagery`, `uea_pems_sf`, `uea_ethanol_concentration`,
  `uea_self_regulation_scp1`) — aeon downloads them on first loader call.
  Cached under `.cache/aeon/`.
- **`amex_default`** — gated. Requires Kaggle creds + manual download:
  ```bash
  kaggle competitions download -c amex-default-prediction \
      -p .cache/benchmarks/archives/amex_default/
  ```

## 10. IDE workspace

Open the parent `seq-sklearn/` folder in your IDE. The `.meta/` symlink
gives access to the agent config files at the standard locations.

### VS Code

The repo includes `.vscode/seq-sklearn.code-workspace` (if not, create
it) — a multi-root workspace pinning both `seq-sklearn` and
`seq-sklearn-meta` as separate roots. That lets you `git add` /
`git commit` in either repo from the same window. To open:

```bash
code .vscode/seq-sklearn.code-workspace
```

If you'd rather just open the parent folder, `code .` works too; the
symlinks resolve transparently, but git operations on meta files require
`cd .meta && git ...`.

### JetBrains (PyCharm)

Open the parent `seq-sklearn/` folder. Mark `.venv/` as Python SDK. Add
`.meta/` as a content root via Settings → Project Structure if you want
to edit agent config alongside library code.

## 11. Running the test suite + benchmark configs

```bash
# Quick gate
uv run pytest tests/benchmarks/ -q

# A single benchmark config (synthetic binary classification, all 7 models)
uv run python -m benchmarks.run \
    --config configs/synthetic_binary_smoke.toml \
    --experiment raw_loss

# HPO uplift on synthetic datasets (TFT-only, GPU)
uv run python -m benchmarks.run \
    --config configs/synthetic_hpo_tft_parallel.toml \
    --experiment all
```

Run outputs land under `.cache/benchmarks/runs/<config-name>/`.

## 12. Sanity check that everything moved

After cloning + bootstrapping + installing, this checklist should all
pass:

- [ ] `git -C . status` → clean
- [ ] `git -C .meta status` → clean
- [ ] `ls -la CLAUDE.md` → symlink to `.meta/CLAUDE.md`
- [ ] `ls -la .claude` → symlink to `.meta/.claude/`
- [ ] `uv run python -c "import seq_sklearn; print(seq_sklearn.__version__)"` → prints
- [ ] `uv run python -c "import benchmarks.adapters, benchmarks.datasets; from benchmarks.registry import list_datasets, list_models; print(len(list_datasets()), len(list_models()))"` → prints `17 11` (the side-effect imports of `benchmarks.adapters` + `benchmarks.datasets` are what populate the registry)
- [ ] `uv run pytest tests/benchmarks/ -q` → 1132+ passed
- [ ] If GPU: `uv run python -c "import torch; print(torch.cuda.is_available())"` → `True`
