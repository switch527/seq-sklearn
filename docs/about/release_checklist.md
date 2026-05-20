# Release checklist (v1.0.0)

The acceptance-criteria-driven checklist a release engineer
executes to ship a v1.0.0 (or any subsequent) release. Each item
maps to an owning CI job or artifact, so "done" is verifiable, not
a checkbox.

## Library-wide criteria (every release)

| # | Criterion | Owner |
|---|---|---|
| 1 | `ruff check`, `ruff format --check`, `pyright` strict pass | `pr.yml` `lint` + `type` jobs |
| 2 | `pytest -m "not slow and not perf"` passes the 85% line / 80% branch coverage gates | `pr.yml` `test-unit` job (with `--cov`) |
| 3 | `tests/deploy/` smoke test passes against the built wheel | `pr.yml` `test-deploy` job |
| 4 | The `/design-review` loop has reached consensus on changes since the previous release | The relevant `docs/phaseN_*.md` Tracking section (final round must show APPROVE from arch + style + qa, zero CRITICAL, every IMPROVEMENT resolved or deferred) |
| 5 | The `style-reviewer` agent reports zero CRITICAL findings | The final round of `/design-review` and `/review` ledgers in the phase docs |
| 6 | `CHANGELOG.md` is updated | `docs/about/changelog`'s `[{version}] - {date}` entry exists, with at least one of `Added`/`Changed`/`Deprecated`/`Removed`/`Fixed`/`Security` |
| 7 | A release-candidate wheel installs from TestPyPI and runs a minimal end-to-end script | Manual: `pip install -i https://test.pypi.org/simple/ seq-sklearn=={version}` + `python -c "from seq_sklearn import TFTClassifier; print(TFTClassifier)"` |

## v1-specific criteria (TFT release)

| # | Criterion | Owner |
|---|---|---|
| 8 | All F1-F11 requirements are implemented and tested | The corresponding test files per `tests/unit/`, `tests/integration/`, `tests/e2e/`, `tests/deploy/`. Spot-check: `tests/unit/test_check_estimator.py` (F1.1 sklearn contract), `tests/e2e/test_quickstart.py` (N1 quickstart-in-CI), `tests/integration/test_onnx_parity.py` (F1 ONNX export), `tests/perf/test_n7_absolute.py` (criterion 9, this list, item 9). |
| 9 | All N1-N7 requirements are met | N1-N6 are CI-gated on every PR (see items 1-3 above). N7 absolute budgets split by hardware. Three of the four numbers (GPU peak memory, training wall-clock, GPU inference latency) are validated by `tests/perf/test_n7_absolute.py::test_n7_absolute_budgets` (marked `gpu` and `slow`; skips unless `SEQ_SKLEARN_N7_GPU=1` so the strict per-batch budgets never assert incidentally on a non-reference CUDA device). The fourth (CPU inference latency) is validated by `tests/perf/test_n7_absolute.py::test_n7_cpu_inference_latency` (marked `slow`; skips unless `SEQ_SKLEARN_N7_CPU=1`). **Release-checklist step:** run `SEQ_SKLEARN_N7_GPU=1 pytest -m "gpu and slow" tests/perf/test_n7_absolute.py` once on an A100/T4/4090 to record the three GPU/training numbers; run `SEQ_SKLEARN_N7_CPU=1 pytest -m slow tests/perf/test_n7_absolute.py::test_n7_cpu_inference_latency` once on the release reference CPU to record the CPU latency; record all four in the `CHANGELOG.md` v1.0.0 entry. |
| 10 | Two quickstart examples exist and pass in CI | `tests/e2e/test_quickstart.py` (binary classifier ≥ 0.75 three-seed median per N1) + the quantile regressor example in the gallery (80% interval coverage in `[0.75, 0.85]` after conformal calibration). |
| 11 | The `/gemini-final-pass design` against requirements + architecture surfaces no new CRITICAL | The recorded Gemini tally in `docs/phase12_docs_release.md`'s Tracking section (if Gemini was run for the release). For v1.0.0, Gemini was deferred per the user-ratified Claude-only consensus; this criterion is documented as "Claude-only consensus" in the CHANGELOG. |

## Repo-settings (human-verified, not a test)

- The `docs` PR job is registered in the repo's branch-protection
  required status checks (so the `sphinx-build -W` gate cannot be
  bypassed by merging without waiting). Verified by visiting the
  GitHub branch-protection settings page once per release.

## Release-time mechanical steps

1. **Bump the version.** Set `pyproject.toml` `version =
   "{version}"`. `seq_sklearn.__version__` reads it via
   `importlib.metadata`; you don't edit `__init__.py`.
2. **Convert `[Unreleased]` to `[{version}] - {date}`** in
   `CHANGELOG.md`, with the full Keep-a-Changelog section set.
   Add a fresh empty `[Unreleased]` at the top.
3. **Record the N7 numbers** from the criterion-9 manual GPU run
   in the new CHANGELOG entry.
4. **Build the wheel.** `uv build --wheel`. Sanity-check:
   `unzip -l dist/seq_sklearn-{version}-*.whl | head`.
5. **Upload to TestPyPI** and run the criterion-7 minimal e2e.
6. **Tag and push.** `git tag v{version} && git push --tags`.
7. **Upload to PyPI.** `uv publish` (with credentials).
8. **Activate the new RTD version** in the Read the Docs dashboard
   (the new tag triggers a build automatically; flip it to
   `Active` + `Public`).
9. **Cut a GitHub release** with the CHANGELOG entry's content as
   the release notes.
