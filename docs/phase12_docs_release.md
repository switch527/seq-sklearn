# Phase 12: Documentation and release prep (plan)

> **Stack decision (Phase 12 design-review R1, user-ratified):**
> the docs toolchain is **Sphinx + numpydoc + autosummary/autodoc +
> sphinx-gallery, PyData Sphinx Theme, Read the Docs**, NOT mkdocs.
> The earlier mkdocs resolution (architecture A12, requirements
> Q16) is overturned. The contradicting authoritative docs are
> already reconciled in THIS change set (arch-C1 fixed by landing,
> not asserting): `docs/architecture.md` A12 rewritten to Sphinx;
> `docs/requirements.md` N6 + open-Q12 + Q16 + the `[docs]` extra
> reconciled; `docs/readme_and_docs_plan.md` marked the ratified
> stack-rationale doc; `docs/docs_strategy_research.md` tooling
> verdict annotated superseded (its framework-agnostic findings
> stand). The existing `docs/conf.py` skeleton + `docs/api/
> generated/*.rst` stubs are the foundation, not discarded.

## Requirements

The grading rubric. Every finding must trace to one of these, to a
fundamental correctness concern, or to a cited framework-agnostic
finding in `docs/docs_strategy_research.md` (RS-Cn / RS-Dn / RS-Sn).

- **R1** Ship the Sphinx site per architecture A12 (rewritten):
  `docs/conf.py` carrying the A12 extension recipe (autodoc,
  autosummary, napoleon, intersphinx, doctest, numpydoc,
  autodoc-pydantic, sphinx-gallery, sphinx-copybutton, myst-parser),
  the `[docs]` extra pinned exactly as A12 now lists, PyData Sphinx
  Theme, building on the existing `docs/conf.py` + `docs/api/
  generated/*.rst` skeleton.
- **R2** `sphinx-build -W --keep-going` passes (warnings-as-errors:
  broken xrefs, unknown autosummary targets, toctree omissions) and
  is a REAL required PR-CI gate replacing the `pr.yml` no-op docs
  job. The job's required-status-check registration is enumerated in
  the release checklist (qa-C5: a job not in branch protection is
  advisory, not a gate).
- **R3** Every documentation code snippet is executed in CI (RS-C2),
  by FOUR mechanisms with no silent-skip path: (a) docstring
  snippets via `pytest --doctest-modules src/seq_sklearn/`
  (requirements N1), added to the `pr.yml` unit job invocation
  (qa-N2); (b) reST/MyST prose snippets via `sphinx.ext.doctest`
  (`.. testcode::` / `.. doctest::`) run by `sphinx-build -b doctest`
  in the docs job; (c) `docs/examples/*.py` executed at build by
  `sphinx-gallery` (a failing example fails the build); (d)
  `README.md` (repo root, OUTSIDE `docs/`, so b/c do not reach it):
  `test_quickstart_readme_block` (PD.4) EXTRACTS the `## Quickstart`
  fenced block and `exec`s it in an isolated namespace
  (`importorskip("seq_sklearn")`, CPU-forced), asserting it runs
  without exception, NOT merely that it parses (qa R3-C1: this is
  the named mechanism the R4 "CI-executed" claim traces to; the
  legible fenced shape stays legible, no `>>>` doctest reformat,
  while still being genuinely executed). A bounded
  non-executable-block meta-test (PD.4) caps how much can opt out.
- **R4** The N1 quickstart is the single executable source and
  cannot silently rot. Ground truth (verified): `README.md:76` is a
  `## Planned API (not yet released)` section with THREE illustrative
  python fences and NO working quickstart; `examples/quickstart.py`
  is a `def run_quickstart() -> float` whose own docstring states
  "the README snippet is the legible shape; this is the executable
  form" - so README and the file are INTENTIONALLY not identical and
  an AST-equivalence test (the R1 attempt) is wrong by the file's own
  contract (qa R2-C1/C2 corrected). Phase 12 mechanism:
  - Phase 12 REWRITES `README.md` from "Planned API" stubs into a
    real one-screen quickstart under a single `## Quickstart`
    heading (the disambiguator: the anti-drift test targets the
    first ```` ```python ```` fence after the `## Quickstart`
    heading, not "the fenced block"; qa R2-C2).
  - `examples/quickstart.py` stays the SINGLE executable source,
    N1-tested by `tests/e2e/test_quickstart.py` (unchanged).
  - `docs/index.md` embeds `examples/quickstart.py` via
    `literalinclude` (zero drift by construction; the SITE quickstart
    is the real tested code, strictly better than a legible-only
    snippet for RS-C1/C2).
  - `README.md`'s `## Quickstart` fenced block is CI-EXECUTED by R3
    mechanism (d): `test_quickstart_readme_block` (PD.4) extracts
    the block and `exec`s it (isolated namespace,
    `importorskip("seq_sklearn")`, CPU-forced), so it genuinely
    cannot rot even though it is the legible shape and not
    byte-identical to the file. The test ALSO asserts the block
    exists under `## Quickstart`, parses, and imports the same
    public symbols `examples/quickstart.py` uses (structural pins
    layered on top of real execution, NOT a substitute for it; the
    qa R3 critique that import-set alone is too weak is answered by
    the exec). Two quickstart examples exist and pass in CI
    (acceptance criterion 10).
- **R5** Diátaxis four-mode IA (RS-C3): `tutorial/` `how-to/`
  `reference/` `explanation/` `about/` `design/` toctrees, every page
  with exactly one home, dense bidirectional cross-linking via
  `intersphinx` + `:ref:` (RS-SD).
- **R6** Time-to-first-success (RS-C1) + first-screen value prop
  (RS-C5): `docs/index.md` opens with the one-screen value prop then
  the runnable zero-setup `fit`/`predict` (bundled/synthetic data, no
  download, GPU, or credentials) above the fold, embedded from
  `examples/quickstart.py`.
- **R7** Reference quadrant auto-generates: pydantic config classes
  as `autodoc-pydantic` field/validator tables (the dominant
  reference surface; `:inherited-members:` across `TFTConfig <-
  BaseModelConfig <- BaseTrainingConfig`); the estimator API via
  `autosummary` + `numpydoc` with the Lim et al. 2021 TFT citation
  in the TFT class `References`; a Glossary as the canonical
  sklearn-contract page API + guides `:ref:` into (RS-SE).
- **R8** The domain misuse-risk pages exist (RS-D*): data-format
  contract; explicit not-a-forecaster / classification-first framing;
  time-series splitting & leakage (Wrong-vs-Right, RS-SA); a
  standalone determinism/reproducibility reference; ONNX export
  round-trip how-to; goal-framed tuning guide; the F11
  `observability` event reference.
- **R9** Release maturity signals (RS-C6): `CHANGELOG.md` v1.0.0
  entry (Keep a Changelog, in use); `about/versioning` deprecation
  policy (public-API SemVer guarantee + window); the migration-guide
  TEMPLATE (1.0.0 has nothing to migrate); RTD versioned hosting
  configured (`.readthedocs.yaml` + stable/latest), which is RTD's
  native versioning (NOT deferred, unlike the mkdocs `mike` path).
- **R10** Repo rules: ruff/pyright clean on new `.py` (example
  scripts, conf.py); no `print` in library code (example scripts may
  print); docs build does not regress `pytest -m "not slow and not
  perf"`; the criterion-9 N7-absolute release-checklist step (wired
  in Phase 11) is enumerated with `test_n7_absolute.py` existence
  machine-gated (qa-I4).
- **R11** Acceptance criterion 11 (Gemini design pass surfaces no new
  CRITICAL) is made verifiable, not a bare checkbox (qa-I3): the
  Phase 12 plan's own Gemini final-pass is recorded in this doc's
  Tracking section with its tally; the release checklist references
  that recorded result.
- **R12 (done-when, implementation_plan Phase 12)** the site renders,
  the API reference shows the pydantic field tables via
  autodoc-pydantic, `pytest --doctest-modules src/seq_sklearn/`
  passes, the quickstart-in-CI passes, and the release checklist
  (acceptance criteria 1-11, including the criterion-9 step) is
  enumerated with every item mapped to an owning CI job or artifact.
- **R13 Public-API façade + version (release-prep; the unowned
  Phase 8<->12 seam, user-ratified into Phase 12).** Ground truth at
  S6 start: `src/seq_sklearn/__init__.py` is the Phase-0 placeholder
  (`__all__ = []`, `__version__ = "0.0.0"`); the documented
  `from seq_sklearn import TFTClassifier` surface does not exist; no
  test in the 95-file suite imports the façade (every test uses the
  deep module path), so 11 phases of governance never flagged it
  (it had no owning test; coverage measures lines-executed not
  surface-correctness; `check_estimator` validates classes handed to
  it directly). Phase 12 wires `seq_sklearn/__init__.py` to the
  EXACT spec-defined surface in architecture **A3** ("Public-API
  surface", `docs/architecture.md:219-251`; the literal `__all__`
  block at `:240`: `TFTClassifier`,
  `TFTRegressor`, `TabularToSequence`, `TabularToSequenceConfig`,
  `TFTConfig`, `EntityTimeSeriesSplit`, `HardwareTier`, `detect`,
  the six error classes, `AttentionOutput`,
  `RegressionAttentionOutput`, `suggest_params`,
  `optuna_trial_guard`), with the matching imports, AND sets the
  single-sourced version (`pyproject.toml` `version` ->
  release version; `seq_sklearn.__version__` read via
  `importlib.metadata.version`, not a hardcoded literal that can
  drift). This is re-export of already-implemented,
  already-`check_estimator`-passing classes (no behavior change), and
  it is what makes R4/R6 (a working quickstart) and a non-`0.0.0`
  v1 release possible. Scope is EXACTLY the A3 list, not a wider
  surface; INTERNAL-tier symbols (e.g. `RecurrentSequenceEstimator`,
  `seq_sklearn._*`) stay absent per the requirements stability
  tiers.

## Scope and non-goals

- **Scope**: a shippable v1.0.0 Sphinx site (Diátaxis IA, PyData
  theme), every snippet CI-executed, reference auto-generated from
  pydantic + docstrings, the domain pages, the README/index rewrite,
  the v1.0.0 CHANGELOG + versioning/deprecation policy + migration
  template, RTD config, and the enumerated release checklist.
- **NG1** No content rewrite of `requirements.md` / `architecture.md`
  / phase plans / research docs; hosted as-is under `design/` with an
  upfront "internal governance, not user docs" callout (qa-N1
  default). Their prose is out of scope for style review here.
- **NG2** Migration codemod (a `bump-pydantic` analog): nothing to
  migrate at 1.0.0; template + deprecation policy ship now, codemod
  at the first breaking change. Deferred D1.
- **NG3** Social-proof / who-uses-it grid: no adopters; placeholder
  only. Deferred D2.
- **NG4** Release-highlights narrative posts: start at the first
  minor after 1.0.0 (RS-C6 norm). Deferred D3.
- **NG5** The v1.0.0 RELEASE itself (TestPyPI RC, tag, PyPI publish,
  gh-pages/RTD publish trigger): the follow-on to this phase. Phase
  12 ships the build + strict gate + RTD config + the checklist; the
  release executes the checklist. Deferred D4.

Note vs the prior (mkdocs) plan: the executable examples GALLERY is
NO LONGER deferred. `sphinx-gallery` provides it natively, so v1.0.0
ships a real rendered, executed gallery (this was the single biggest
mkdocs caveat; the Sphinx decision removes it).

## P-0: Stack reconciliation (arch-C1, enumerated and LANDED)

The R1 swarm flagged the Sphinx-vs-mkdocs contradiction as live in
main. It is resolved by editing every authoritative spec site, with
each site enumerated (the R2 arch-C1/C2 lesson: "land, do not
assert", and enumerate so a re-review can verify). Landed in this
change set (commit-of-this-revision):

- `docs/architecture.md`: A12 prose (Sphinx recipe + pin list) `:31`
  Q12-resolution pointer, `:~149` layout (`autosummary`/
  sphinx-gallery), A18 `[docs]` extra block (was a duplicate stale
  mkdocs block), A19 CI table (`sphinx-build -W`), A20 item 4 +
  the Deferred entry (`autodoc-pydantic`). No mkdocs/mkdocstrings/
  griffe language remains except correct "Sphinx analog of
  griffe-pydantic" comparisons.
- `docs/requirements.md`: N6, open-question 12 + 16, the `[docs]`
  extra pin line, the release-workflow item `:~1904`, the
  changelog-tail `Q12` entries (`:~2435`, `:~2585`) all reconciled
  to Sphinx; Q12-OPEN vs Q16-mkdocs no longer contradict.
- `docs/implementation_plan.md`: Phase 12 Modules + Deliverable
  tests + Done-when, the Phase 0 `:~113` docs-job note, the R8 risk
  entry, and the Phase-0 retro entry all reconciled (Phase 12's own
  R12 grading source is now Sphinx, closing R2 arch-C2).
- `docs/readme_and_docs_plan.md` marked the ratified stack doc;
  `docs/docs_strategy_research.md` verdict annotated superseded.

A `test_no_mkdocs_residue_in_specs` (PD.4) greps the four spec docs
for unreconciled `mkdocs`/`mkdocstrings`/`griffe-pydantic` tokens
(allowing only the explicitly-marked "superseded/analog" mentions
via an allowlist of line-substrings) so the reconciliation cannot
silently regress.

The scaffold files (`docs/conf.py`, `pyproject.toml [docs]`,
`.readthedocs.yaml`, `.gitignore`, `.github/workflows/pr.yml`,
`docs/conf.py:exclude_patterns`) are NOT spec contradictions, they
are already Sphinx, but they need concrete Phase-12 deltas; those
are S6 work, enumerated precisely in P-A so the plan drives them
rather than asserting them done.

## P-A: Site scaffold (R1 / R2), enumerated S6 deltas

Ground truth (verified): `pyproject.toml:52-59` `[docs]` is ALREADY
Sphinx with SIX base pins (`sphinx>=8.1,<9`,
`pydata-sphinx-theme>=0.16,<0.17`, `numpydoc>=1.8,<2`,
`myst-parser>=4.0,<5`, `sphinx-copybutton>=0.5,<0.6`,
`sphinx-sitemap>=2.6,<3`). Architecture A12/A18 was R2-rewritten to
the EIGHT-pin POST-Phase-12 TARGET (the six plus `sphinx-gallery` +
`autodoc-pydantic`); the live pyproject has six; PA.2 closes the
gap. This divergence is intentional and enumerated (arch R3-I1),
bounded by `test_docs_extra_is_target_or_live` (PD.4).
`docs/conf.py:24-33` already loads autodoc/autosummary/intersphinx/
napoleon/numpydoc/myst/copybutton/sitemap. `docs/api/generated/`
holds 33 UNTRACKED autosummary stubs (regenerated by
`autosummary_generate`). `.readthedocs.yaml` has
`fail_on_warning: false` and no version config. `.gitignore` has no
`docs/_build/` entry (nothing under `docs/_build/` is git-tracked;
arch R2-I2). The deltas:

- **PA.1** `docs/conf.py`: ADD the three missing A12 extensions
  (`sphinx.ext.doctest`, `sphinxcontrib.autodoc_pydantic`,
  `sphinx_gallery.gen_gallery`), `autosummary_generate=True`,
  `numpydoc_show_class_members=False`, autodoc-pydantic field/
  validator summary flags, the `intersphinx_mapping`
  (sklearn/numpy/torch/pydantic), `sphinx_gallery_conf` ->
  `docs/examples/`, `nitpicky=True`. `sphinx_copybutton`/
  `sphinx_sitemap`/`myst_parser` already present (no change). The
  A12 recipe is aligned to keep `sphinx-sitemap` (it is in the real
  pyproject and aids SEO/adoption; arch R2-C3/N1).
- **PA.2** `pyproject.toml [docs]`: ADD exactly two pins
  (`sphinx-gallery>=0.18,<0.19`, `autodoc-pydantic>=2.2,<3`), bounds
  matching the repo N3 upper-bound convention the existing pins use
  (arch R2-N1). The other six already match A12; no churn.
- **PA.3** Diátaxis toctree: the existing MyST `docs/index.md`
  becomes the value-prop + above-the-fold quickstart landing; the
  33 (not 35; arch R2-I3) autosummary *targets* are relocated via
  the `reference/` toctree (the stubs are untracked build output,
  regenerated, not git-moved); new `tutorial/ how-to/ explanation/
  about/ design/` trees; root toctree wires every page (no orphans).
- **PA.4** `docs/conf.py:exclude_patterns` currently EXCLUDES
  `requirements.md`/`architecture.md`/`implementation_plan.md`/
  research from the build (arch R2-I1). Phase 12 MOVES those into a
  `design/` subtree and REMOVES them from `exclude_patterns` so the
  `design/` quadrant renders with an "internal governance, not user
  docs" callout (Q3 resolution). `.gitignore` gains `docs/_build/`
  (arch R2-I2: additive only, nothing committed to remove; the
  earlier "remove committed _build" wording was wrong).
- **PA.5** `.readthedocs.yaml`: flip `fail_on_warning: true`
  (mirrors the CI `-W` gate) and add RTD versioning (stable /
  latest). R9's versioned hosting, RTD-native (no `mike`).

## P-B: The four Diátaxis quadrants

- **PB.1 Tutorial** `tutorial/first_classifier.{md,rst}`: one linear
  finishable spine on a bundled/synthetic panel (config -> fit ->
  predict/proba -> evaluate -> read variable-selection), labeled
  "this alone is enough". Distinct from the index quickstart (the
  30-second hook); Q1 below asks whether to keep both or use one
  progressive example (FastAPI single-spine, the R1-qa lean).
- **PB.2 How-to** `how-to/*`, task-named: `configure_training`,
  `imbalanced_classes`, `extract_attention`, `tune_with_optuna`,
  `export_onnx` (RS-DG, the Phase 10 capability's user guide: full
  export -> reload-in-onnxruntime round trip), `persist_a_model`
  (RS-SC: joblib vs ONNX decision + "never load untrusted pickles" +
  version-pin/containerize), `panel_data` (RS-DA, the data-format
  contract: exact shape, why consecutive rows are consecutive
  periods, ragged case, PRINTED-shape worked example, top misuse
  source), `time_series_splitting` (RS-DD/SA: entity-time CV, why
  random splits leak on multi-entity panels, Wrong-vs-Right),
  `tuning` (RS-DH: goal-framed, names the ~5 fields that matter,
  inline win/lose tradeoffs, points at the Optuna integration),
  `performance` (RS-DF: precision, GPU/CPU, batch/throughput/memory,
  few-line wins).
- **PB.3 Reference** (auto-generated, R7): `reference/config` via
  `autodoc-pydantic` over the config classes (field tables: type /
  default / constraint / description / validators, inherited-members
  across the config chain); `reference/api` via `autosummary` +
  `numpydoc` over the estimator public API (the existing docstring
  convention; Lim et al. 2021 in the TFT `References`);
  `reference/glossary` the canonical sklearn-contract page;
  `reference/observability` the F11 event-payload reference
  (implementation_plan Phase 12 deliverable).
- **PB.4 Explanation** `explanation/*`:
  `why_tft_for_classification` (with the explicit "seq-sklearn is NOT
  a forecasting library" statement, RS-DB, also a first-screen
  `index` line), `how_variable_selection_and_attention_work`,
  `design_sklearn_api_over_lightning`,
  `determinism_and_precision` (the model behind the RS-DE reference
  page); `reference/determinism` is the standalone reproducibility
  reference (seeding, strict-mode side effects, CUDA flags, the
  "not bit-exact across versions/platforms" caveat + perf cost).

## P-C: Examples gallery (sphinx-gallery, now in scope)

- **PC.1** `docs/examples/` holds runnable `.py` scripts in the
  sphinx-gallery format (a module docstring header + `# %%` cells).
  `examples/quickstart.py` (the N1 source) is surfaced here too.
  v1.0.0 ships a SMALL curated set (quickstart binary classifier,
  quantile regressor, attention extraction, imbalanced) - quality
  over count; the grid grows post-v1.
- **PC.2** sphinx-gallery executes every example at build; a failing
  example fails `sphinx-build -W`. This IS R3 mechanism (c) and
  replaces the prior plan's deferred/curated-scripts compromise.
  Gallery examples force CPU + a tiny synthetic panel +
  `max_epochs=1` so the build stays inside the docs-CI budget
  (RK1).

## P-D: Snippet-execution gate + its own tests (R3, the trust gate)

- **PD.1** `pytest --doctest-modules src/seq_sklearn/` is ADDED to
  the `pr.yml` unit-job command (qa-N2: it is currently absent from
  `pr.yml:56`; N1 mandates it). Required.
- **PD.2** `sphinx-build -b doctest docs/` runs in the docs job;
  every `.. testcode::`/`.. doctest::` block executes. Illustrative
  non-runnable code uses a plain `.. code-block:: python` (no skip
  marker needed, it is simply not a doctest directive).
- **PD.3** The docs `pr.yml` job (was a no-op) does:
  `uv sync --extra docs --extra dev`; `sphinx-build -W --keep-going
  -b html docs/ docs/_build/html`; `sphinx-build -W -b doctest docs/
  docs/_build/doctest`. Both required. Q4 asks single-vs-split job;
  default single (one `uv sync`, fits the 5-minute budget; build
  ~tens of seconds with tiny examples). `linkcheck` runs on the
  nightly schedule, not the PR gate (network-flaky).
- **PD.4 Tests for the gate itself** (so it cannot silently rot),
  under `tests/docs/`, NON-perf/slow, fast, `pytest.importorskip`
  (visible skip, never silent) where the `docs` extra is needed:
  - `test_sphinx_build_warns_as_errors`: subprocess `sphinx-build -W
    -b html` into a tmp dir, assert exit 0. `importorskip("sphinx")`
    so the skip is VISIBLE in output (qa-C2: no silent no-op).
  - `test_autodoc_pydantic_field_tables_rendered`: after the build,
    parse `reference/config` HTML, assert known `TFTConfig` field
    names (e.g. `hidden_size`, `attention_heads`) AND a validator
    name appear, so a misconfigured autodoc-pydantic that emits an
    empty Fields section fails (qa-C3: `-W` alone does not catch an
    empty-but-valid render).
  - `test_quickstart_readme_block` (R4 + R3 mechanism (d); qa
    R2-C1/C2 + R3-C1 + R4-I2/N1): locate the FIRST ```` ```python ````
    fence after the `## Quickstart` heading in `README.md` (fails,
    not skips, if the heading or fence is absent, the forcing
    function for the PE.0 "Planned API" -> real-quickstart rewrite).
    BEFORE exec: `ast`-scan the block and assert it constructs the
    estimator with `max_epochs <= _DOC_MAX_EPOCHS` (a pinned module
    constant, the same bound the gallery scan uses, qa R4-N1: a
    heavy README fit cannot hang the unit job; the block mirrors the
    N1-tiny `examples/quickstart.py`). Then `exec` the source in an
    isolated namespace, applying the EXACT `_force_cpu` monkeypatch
    set from `tests/e2e/test_quickstart.py` (patch
    `seq_sklearn.training.trainer.detect` -> `HardwareTier.CPU`,
    `torch.cuda.is_available` -> `False`, `torch.cuda.device_count`
    -> `0`; reuse that helper, do not reinvent it; qa R4-I2: CPU is
    ENFORCED by contract, not by analogy), `importorskip("seq_sklearn")`,
    asserting it runs without exception. ALSO assert it parses and
    imports the same public symbols `examples/quickstart.py` imports
    (cheap structural pins LAYERED ON the exec, not a substitute).
    This is the README's real execution gate; `docs/index` uses
    `literalinclude` of `examples/quickstart.py` so it is the real
    tested code and needs no test beyond the build.
  - `test_no_mkdocs_residue_in_specs` (P-0 guard): grep
    `docs/architecture.md`, `docs/requirements.md`,
    `docs/implementation_plan.md`, `docs/readme_and_docs_plan.md`
    for `mkdocs`/`mkdocstrings`/`griffe-pydantic`; a hit is ALLOWED
    when the hit line OR either ADJACENT line (a +/-2-line window,
    case-insensitive) contains any of a NAMED anchor set:
    `superseded`, `overturned`, `reconciled`, `ratified`,
    `interim`, `earlier mkdocs`, `not mkdocs`, `sphinx analog`,
    `resolved to sphinx`, and `mkdocs<2` (the upper-bound-policy
    example, which is about the pin policy, not the docs stack).
    The window + case-insensitive + broadened anchors close qa
    R4-I1 (the prior per-line 6-anchor form covered under half the
    real hits and would have failed day 1). The anchor list is
    enumerated HERE so it is auditable at design time, and
    `test_no_mkdocs_residue_in_specs` itself asserts the current
    spec corpus passes (so the anchor set is proven sufficient on
    the day it lands, not aspirationally).
  - `test_docs_extra_is_target_or_live` (arch R3-C1/I1): assert the
    architecture A12/A18 `[docs]` pin list equals EITHER the live
    `pyproject.toml [docs]` extra OR that list plus exactly
    `{sphinx-gallery, autodoc-pydantic}` (the PA.2 adds), so the
    intentional pre-PA.2 divergence is bounded and a third state
    (real drift) fails. Parses both pin lists, compares the package
    name sets.
  - `test_every_gallery_script_has_max_epochs_1` (qa R2-C3, static
    pre-execution scan): `ast`-parse every `docs/examples/*.py`,
    assert each estimator construction passes
    `max_epochs <= _DOC_MAX_EPOCHS` (the SAME single pinned module
    constant the README-block scan uses, so the gallery and the
    README quickstart share one bound), so an over-heavy script is
    caught BEFORE the build runs it, not only after via the
    wall-clock `test_doc_snippet_suite_fast`.
  - `test_every_doc_has_a_toctree_home`: assert every `docs/**/*.{md,
    rst}` (excluding `_build`, `design/` hosted-as-is) is reachable
    from a toctree (explicit orphan check complementing `-W`).
  - `test_nonexecutable_block_ratio_bounded`: across `docs/`, assert
    `plain code-block python` / (all python blocks) < 0.25 (a pinned
    `_MAX_NONEXEC_RATIO`), so authors cannot silently make the whole
    suite non-executable (qa-C1: concrete bound, named test).
  - `test_doc_snippet_suite_fast`: assert the `-b doctest` build
    wall-clock < a pinned `DOCS_DOCTEST_BUDGET_S` (e.g. 120 s) on the
    CI tier, so one hanging TFT-fit block cannot blow the PR budget
    (qa-I1).
  - `test_n7_absolute_test_present_and_unskipped`: assert
    `tests/perf/test_n7_absolute.py` exists and has a test function
    not under an unconditional `pytest.mark.skip` (qa-I4: the
    criterion-9 release step has a real artifact, not just an
    enumerated line).
  - `test_public_api_surface` (R13/PE.4; the owning test the prior
    11 phases lacked; runs in the UNIT job's default
    `pytest -m "not slow and not perf and not gpu"` sweep via
    `testpaths = ["tests"]`, so it gates EVERY PR going forward, qa
    delta-N1). Assertions:
    (a) Spec-literal source: `_SPEC_PUBLIC_API` is EXTRACTED from
    `docs/architecture.md` at test-module scope by REGEX over the A3
    `__all__` block. The pattern uses tolerant whitespace
    (`r"__all__\s*=\s*\[(.*?)\]"` with `re.DOTALL`) so future
    formatting edits to the spec block cannot silently produce an
    empty set; on a non-matching file the test fails LOUDLY with a
    parse error, not a silent zero-element compare (qa confirming
    delta-I1). Strip whitespace/quotes from the captured names;
    parse to a set. NOT a hand-typed Python literal (qa delta-C1: a
    hand-typed copy is what kept the 11-phase drift class open).
    Assert `set(seq_sklearn.__all__) == _SPEC_PUBLIC_API`.
    (b) Every name in `__all__` is a real attribute on
    `seq_sklearn` AND `getattr(seq_sklearn, name) is`
    (identity-compare) the deep-path import, so the façade
    re-exports the real symbols, not stubs.
    (c) Version single-source: `seq_sklearn.__version__ ==
    importlib.metadata.version("seq-sklearn")`; on
    `PackageNotFoundError` (a non-editable invocation, e.g.
    `PYTHONPATH=src pytest`), `pytest.fail("seq-sklearn not
    installed; run uv sync or pip install -e .")` so a misconfigured
    CI cell fails LOUDLY, not with an unhandled exception (qa
    delta-I1).
    (d) INTERNAL-tier forbidden set (qa delta-I2; "anything `_*`" is
    a tautology since `__all__` cannot legitimately carry private
    names): assert the ENUMERATED INTERNAL-tier names from A3
    (`RecurrentSequenceEstimator`,
    `RecurrentSequenceEstimatorConfig`, per
    `docs/architecture.md:253-257`) are NOT in
    `seq_sklearn.__all__` AND `getattr(seq_sklearn, name, _MISSING)
    is _MISSING`. Plus a structural rule: every non-underscore,
    non-dunder name in `vars(seq_sklearn)` (NOT `dir`, which
    surfaces inherited module-type attrs the test does not own)
    MUST be in `__all__` (no namespace leakage via `getattr` outside
    the documented surface; qa confirming delta-NIT).
    (e) Subprocess façade execution: in a subprocess via
    `sys.executable` (same venv), `import seq_sklearn` then
    `from seq_sklearn import <each_A3_name>` (one statement per
    name, so a failure identifies which name is broken); assert
    returncode 0 and no traceback. Layered on (b) which checks
    identity in the parent process; the subprocess proves the
    package is genuinely installed-and-importable, not just
    monkey-imported in the test session.
    This test, NOT a phase plan, is what OWNS the façade going
    forward, closing the seam that hid PE.4 for 11 phases.

- **PE.0 README rewrite (qa R2-C1, an explicit deliverable, not
  implied).** `README.md`'s `## Planned API (not yet released)`
  section (currently three illustrative non-working fences) is
  rewritten into a real `## Quickstart` with one runnable one-screen
  example whose code is CI-executed (R3/R4). The "Planned API"
  framing is removed (v1.0.0 ships the API; it is no longer
  "planned"). Without this, `test_quickstart_readme_block` fails by
  construction, which is the intended forcing function.
- **PE.1** `CHANGELOG.md`: `[Unreleased]` -> `[1.0.0] - <date>` with
  the full Keep-a-Changelog section set, summarizing F1-F11 + N1-N7;
  fresh empty `[Unreleased]` on top. `test_changelog_has_1_0_0_entry`
  asserts the section + SemVer header shape (qa-N3, a cheap regex
  guard, not a full linter).
- **PE.2** `about/versioning`: public-API SemVer guarantee (public =
  estimator classes + pydantic configs + documented functions;
  `_`-prefixed private), deprecation window (deprecate in a minor
  with a runtime `DeprecationWarning` naming the replacement; remove
  no earlier than the next major). `about/migration_template`: the
  empty-but-structured template (changes grouped by surface area,
  before/after per renamed symbol, deprecation-warning policy,
  codemod slot).
- **PE.3** `about/release_checklist`: acceptance criteria 1-11
  enumerated, EACH mapped to an owning CI job or artifact, including:
  the criterion-9 manual `pytest -m "gpu and slow"
  tests/perf/test_n7_absolute.py` step (Phase 11-wired); criterion 11
  pointing at THIS doc's recorded Gemini-final-pass tally (R11); and
  an explicit "the `docs` PR job is registered in branch-protection
  required-status-checks" line (qa-C5, a human-verified repo-settings
  item, not a test).
- **PE.4 Public-API façade + version wiring (R13).** Implement the
  spec-defined re-export block in `src/seq_sklearn/__init__.py`
  EXACTLY per architecture A3 ("Public-API surface",
  `docs/architecture.md:219-251`): the
  imports of `TFTClassifier`/`TFTRegressor` from their module paths,
  `TabularToSequence`/`TabularToSequenceConfig`, `TFTConfig`,
  `EntityTimeSeriesSplit`, `HardwareTier`/`detect`, the six error
  classes from `seq_sklearn.errors`, `AttentionOutput`/
  `RegressionAttentionOutput`, `suggest_params`,
  `optuna_trial_guard`; then `__all__` = that list (the spec
  literal). `__version__` is read via
  `importlib.metadata.version("seq-sklearn")` (NOT a hardcoded
  literal that drifts from `pyproject.toml`); a single source of
  truth. `pyproject.toml` `version` is bumped from `"0.0.0"` to the
  v1.0.0 release version at release time (PE.1 records it; the
  CHANGELOG entry and `__version__` will then match by
  construction). The classes being re-exported are
  already-implemented and already pass the Phase-9 N1
  `check_estimator` suite; PE.4 is re-export + version, no behavior
  change.

## Resolved questions (R2, both reviewers converged)

- **Q1 RESOLVED** ONE progressive example (FastAPI single-spine): the
  `index` 30-second hook and `tutorial/first_classifier` are the same
  canonical `examples/quickstart.py` surfaced at two depths
  (`literalinclude` on index; the gallery-rendered, narrated run as
  the tutorial), not two maintained snippets. One executable source,
  zero drift.
- **Q2 RESOLVED** Double-surfacing is correct, not confusing: the
  index is the above-the-fold hook, the gallery is the executed,
  output-bearing full run. Both point at the one
  `examples/quickstart.py`; there is nothing to keep in sync.
- **Q3 RESOLVED** Host the internal design docs in-site under
  `design/` WITH an "internal governance, not user docs" callout
  (NG1), conditional on PA.4 removing them from
  `conf.py:exclude_patterns` (they are currently excluded; the plan
  now lands that flip).
- **Q4 RESOLVED** Single docs CI job (one `uv sync --extra docs
  --extra dev`, then `-b html` and `-b doctest` sequentially); the
  ~40s build fits the 5-minute budget; revisit only if breached.

## Risks

- **RK1** Building the gallery executes real TFT fits; a too-large
  example blows the docs-CI budget. Mitigated by CPU + tiny synthetic
  panel + `max_epochs=1` in every gallery script, and
  `test_doc_snippet_suite_fast` (PD.4) catching regressions.
- **RK2** `sphinx-build -W` is brittle: every unresolved xref fails.
  Mitigated by `nitpicky` + building incrementally +
  `test_sphinx_build_warns_as_errors` catching it pre-merge.
- **RK3** autodoc-pydantic may mis-render the deep config inheritance
  chain. Mitigated by `test_autodoc_pydantic_field_tables_rendered`
  asserting real field + validator names in the HTML, not just a
  clean build.
- **RK4** The existing `docs/conf.py`/`docs/api/generated/*.rst`
  skeleton may carry stale assumptions from before the stack was
  ratified. Mitigated by PA.1 treating conf.py as extend-not-trust
  and the strict build surfacing any stale stub.

## Deferred (v1.1+, with reason)

- **D1** Migration codemod: nothing to migrate at 1.0.0; template +
  policy ship now.
- **D2** Social-proof / who-uses-it grid: no adopters yet.
- **D3** Release-highlights narrative posts: start at the first minor
  after 1.0.0 (RS-C6 norm).
- **D4** The v1.0.0 release execution (TestPyPI RC, tag, PyPI/RTD
  publish): the follow-on to this phase, not Phase 12's deliverable.
- **D5** Full large-count example gallery: v1.0.0 ships a small
  curated executed set; breadth grows post-v1 (quality over count).

## Tracking (review loop)

Addressed and deferred items are maintained here so successive swarm
runs see prior decisions and do not re-raise resolved points.

Round 1 (architecture 2C/5I/2N REQUEST_CHANGES; qa 5C/4I/3N
REQUEST_CHANGES; style 0/0/0 APPROVE):

- Addressed (CRITICAL):
  - arch-C1 (Sphinx-vs-mkdocs unreconciled contradiction +
    competing scaffold): RESOLVED by user decision (Sphinx) and the
    contradiction LANDED-reconciled in this change set, A12 rewritten,
    requirements N6/Q12/Q16 + `[docs]` extra reconciled,
    readme_and_docs_plan ratified, research verdict annotated, the
    existing scaffold adopted (P-0). The whole plan is rewritten
    Sphinx-native.
  - arch-C2 / qa-C4 (verbatim mirror impossible vs the real
    `run_quickstart()` shape): R4/PD.4 replaced with
    `literalinclude` (index, zero-drift) + AST-equivalence README
    test, not substring matching.
  - qa-C1 (skip-marker meta-test unspecified/unbounded): PD.4
    `test_nonexecutable_block_ratio_bounded` with a pinned
    `_MAX_NONEXEC_RATIO`.
  - qa-C2 (strict-build test silently skipped): PD.4
    `test_sphinx_build_warns_as_errors` uses `importorskip`
    (visible) + PE.3 enumerates the branch-protection registration.
  - qa-C3 (field tables: clean build != rendered content): PD.4
    `test_autodoc_pydantic_field_tables_rendered` asserts real field
    + validator names in the HTML.
  - qa-C5 (docs job not enforced): PE.3 explicit
    branch-protection-required-check line + R2.
- Addressed (IMPROVEMENT):
  - qa-I1 (snippet budget): PD.4 `test_doc_snippet_suite_fast` with
    a pinned `DOCS_DOCTEST_BUDGET_S`.
  - qa-I2 (Q1 runner choice): moot under Sphinx, the runner is
    `sphinx.ext.doctest` + sphinx-gallery, not mktestdocs/
    pytest-markdown-docs; PD.2 states it.
  - qa-I3 (criterion 11 not machine-checkable): R11 + PE.3 point the
    checklist at this doc's recorded Gemini-pass tally.
  - qa-I4 (`test_n7_absolute` not gated): PD.4
    `test_n7_absolute_test_present_and_unskipped`.
  - arch I-set (5): subsumed by the Sphinx rewrite + the above
    (the mkdocs-specific arch IMPROVEMENTs no longer apply); any
    still-live ones re-surface in R2 against the rewritten plan.
- Addressed (NITPICK):
  - qa-N1 (host design docs default): NG1 = host under `design/`
    with an "internal governance" callout.
  - qa-N2 (`--doctest-modules` absent from pr.yml): PD.1 adds it to
    the unit job.
  - qa-N3 (CHANGELOG unlinted): PE.1 `test_changelog_has_1_0_0_entry`
    regex guard.
- Style: APPROVE 0/0/0 (re-graded in R2 against the rewrite).

Round 2 (architecture 3C/4I/2N REQUEST_CHANGES; qa 5-R1-closed +
3-new-C; style 0/0/1 APPROVE accepted-house-style):

- Addressed (CRITICAL):
  - arch-C1 (P-0 "landed" was only partial: mkdocs residue in
    architecture.md A18/A19/`:31`/`:149`/A20.4/Deferred,
    requirements `:1904`/`:2435`/`:2585`, all of
    implementation_plan Phase 12): every site now ENUMERATED in P-0
    and LANDED this revision; `test_no_mkdocs_residue_in_specs`
    (PD.4) guards regression.
  - arch-C2 (implementation_plan Phase 12, R12's grading source, was
    all-mkdocs): reconciled (Modules/Deliverable-tests/Done-when +
    `:113` + R8 risk + Phase-0 retro).
  - arch-C3 (PA.1/2/5 misdescribed the real scaffold): P-A rewritten
    to ground truth, A12 pin list aligned to the real pyproject
    (keeps `sphinx-sitemap`), deltas are ADD-only and enumerated.
  - qa R2-C1 (README has no `run_quickstart` body, only "Planned
    API" stubs): R4 rewritten + PE.0 makes the README
    Planned-API->Quickstart rewrite an explicit deliverable;
    `test_quickstart_readme_block` fails-by-construction until it
    lands.
  - qa R2-C2 ("the fenced block" ambiguous, README has 3): R4/PD.4
    disambiguate to the first python fence after `## Quickstart`.
  - qa R2-C3 (max_epochs prose-only): PD.4
    `test_every_gallery_script_has_max_epochs_1` static AST scan,
    pre-execution.
  - The R1 AST-equivalence anti-drift (itself wrong by the file's
    legible-vs-executable contract) is RETRACTED; replaced by
    literalinclude (index, executable) + structural README block
    test + R3 execution.
- Addressed (IMPROVEMENT/NITPICK):
  - arch R2-I1 (conf.py excludes design docs): PA.4 flips
    `exclude_patterns` so the `design/` quadrant renders.
  - arch R2-I2 (.gitignore, nothing committed to remove): PA.4
    additive-only, the wrong "remove committed _build" wording
    dropped.
  - arch R2-I3 (33 not 35 stubs, untracked): PA.3 corrected.
  - arch R2-I4 (pr.yml not enumerated): P-0 names it; PD.3 owns the
    edit, S6.
  - arch R2-N1 (pin bounds): PA.2 uses the repo N3 bound convention.
  - Q1-Q4 resolved in-doc (both reviewers converged).
- Style: the one R2 NITPICK was accepted house style (inline
  enumeration hyphens), no change.

Round 3 (architecture 1C/1I/1N REQUEST_CHANGES; qa 1C/1I/1N
REQUEST_CHANGES; style 0/0/0 APPROVE consensus). All R2 findings
verified genuinely closed; the two R3 CRITICALs were self-inflicted
regressions from the R2 edits, both doc-only, both fixed:

- Addressed (CRITICAL):
  - arch R3-C1 (A18 prose falsely claimed byte-identity with the
    live pyproject; contradicted PA.2): architecture.md A18 prose
    reworded to "POST-Phase-12 TARGET; live pyproject has the six
    base pins; PA.2 adds the two; intentionally not byte-identical
    until PA.2"; requirements.md `:1937` aligned to capped bounds +
    the same six-live/two-added framing; PD.4
    `test_docs_extra_is_target_or_live` bounds the divergence to
    exactly {live} or {live + sphinx-gallery + autodoc-pydantic}.
  - qa R3-C1 (README block had no real execution path: R3's three
    mechanisms do not reach repo-root `README.md`): R3 now has a
    FOURTH mechanism (d), `test_quickstart_readme_block` EXTRACTS
    and `exec`s the `## Quickstart` block (importorskip, CPU-forced)
    so it genuinely cannot rot; R4/PD.4 reworded to trace the
    "CI-executed" claim to it. The legible fenced shape is kept (no
    `>>>` reformat).
- Addressed (IMPROVEMENT/NITPICK):
  - arch R3-I1: P-A "Ground truth" note records the A12/A18
    eight-pin-target vs live-six divergence as intentional +
    enumerated + bounded by the new PD.4 test.
  - arch R3-N1: requirements.md `[docs]` example aligned to the
    capped-bound N3 convention.
  - qa R3-I1: collapses, the exec gate makes import-set a layered
    pin, not the sole guard (stated in R4/PD.4).
  - qa R3-N1: `test_no_mkdocs_residue_in_specs` allowlist anchors
    are now NAMED in PD.4 (`superseded`, `Sphinx analog of
    griffe-pydantic`, `interim mkdocs`, `NOT mkdocs`, `the 2.0`,
    `ratified`), auditable at design time.
- Style: APPROVE 0/0/0, consensus on style.

Round 4 (FINAL, 4-round cap; architecture 0C/0I/0N APPROVE; style
0C/0I/0N APPROVE; qa 0C/2I/1N APPROVE). Zero CRITICAL. All R3
findings verified closed at source by arch (not just plan claims).
qa's 2 IMPROVEMENTs + 1 NITPICK were mechanical doc-precision items,
RESOLVED in this round rather than deferred:

- qa R4-I1: `test_no_mkdocs_residue_in_specs` switched from a
  per-line 6-anchor allowlist (which covered under half the real
  hits, day-1 failure) to a +/-2-line case-insensitive window with
  a broadened enumerated anchor set, and the test asserts the
  current spec corpus passes so the anchor set is proven sufficient
  on landing.
- qa R4-I2: `test_quickstart_readme_block` CPU-forcing is now a
  contract (reuse `tests/e2e/test_quickstart.py`'s exact `_force_cpu`
  monkeypatch set), not "like the e2e test" by analogy.
- qa R4-N1: the README block gets the SAME static
  `max_epochs <= _DOC_MAX_EPOCHS` pre-exec scan as the gallery
  (one shared pinned constant), so a heavy README fit cannot hang
  the unit job.

CONSENSUS REACHED after 4 rounds. architecture + style APPROVE with
zero findings; qa APPROVE with zero CRITICAL and every IMPROVEMENT
resolved in-doc (none deferred). The Sphinx-stack reconciliation is
landed and verified across the spec corpus; the plan is internally
consistent end to end. NITPICKs permitted to remain (none
outstanding). Cleared for the gated Gemini design final-pass.

Gemini design final-pass: deferred until Gemini quota resets
(blocked earlier this session by `429 QUOTA_EXHAUSTED`); per
R11/PE.3 its tally is recorded here before the release checklist
references it. A new Gemini CRITICAL would reopen consensus for one
more `/design-review` round.

## S7 implementation-review ledger (post-consensus)

The bulk Phase 12 landing (commit `0cf6d89`) went through `/review`
(Claude-only swarm: code + arch + qa + style) per the user direction
to drive to consensus with Gemini deferred. Findings and resolutions
below; the plan-doc enumeration of the 18-name façade above is
historical (PE.4 / R13) and has been superseded at the file level
by the 24-name surface in `seq_sklearn/__init__.py` and
architecture A3.

### S7 Round 1 (commit `2f83ab7`)

Resolved CRITICAL findings:

- code-C1 / arch-C2 / qa-C1 (n7 skip-detection De Morgan):
  `tests/docs/test_misc_gates.py` rewritten to a regex anchored to
  top-of-line `@pytest.mark.skip`; bare-decorator silencing is now
  caught.
- code-C2 (test ruff+pyright): `tests/docs/test_docs_extra_matches_spec.py`
  UP036, SIM109, RUF100 + pyright Optional member access resolved.
- code-C3 (wrong attention axis): gallery + how-to + explanation
  corrected from `out.attention_weights.mean(axis=-1)` to
  `out.attention_weights[:, :, -1, :].mean(axis=1)`; shape comments
  updated to the actual `(N, n_heads, L, L)`.
- arch-C1 / code-I1 (A3-vs-estimator-API seam): promoted the six
  `*Params` adapter classes to STABLE. `seq_sklearn.config._adapters.py`
  renamed to `adapters.py`; `__all__` extended 18 → 24 names;
  architecture A3 + requirements per-module stability table + A4
  step-3 file-path reference all updated; ~50 import sites repointed.
- style-CRITICAL (8 prose em-dashes): rewritten to periods,
  semicolons, or parens across six user-facing docs.

Resolved IMPROVEMENT findings: criterion-9 CPU N7 artifact added
(`tests/perf/test_n7_absolute.py::test_n7_cpu_inference_latency`,
opt-in via `SEQ_SKLEARN_N7_CPU=1`); architecture A12 stale
post-PA.2 prose updated; `_spec_public_api` regex anchored to
`## A3:`; README structural test now validates every `ImportFrom`
resolution; pr.yml docs job timeout-minutes added.

### S7 Round 2

Resolved CRITICAL findings:

- arch-C1: `docs/requirements.md:233` updated `docs/api/` →
  `docs/reference/` (Phase 12 R1 collapsed the path; the normative
  bullet had not followed).
- arch-C2 / qa-C1 (N7 per-sample-vs-per-batch math):
  `tests/perf/test_n7_absolute.py` dropped the `/ len(batch)`
  division on both functions; the `*_INFER_MS` constants are now the
  per-batch budgets per `requirements.md:2100-2101`. BOTH functions
  gated behind env vars (`SEQ_SKLEARN_N7_GPU=1` and
  `SEQ_SKLEARN_N7_CPU=1`) so the strict per-batch budgets never
  assert incidentally on a non-reference device; the release engineer
  sets the env vars on the recording run.
- arch-C3: `docs/architecture.md:51` A1 layout tree updated
  `_adapters.py` → `adapters.py` with a STABLE-per-A3 callout.
- style-C1 + style-C2: two prose em-dashes in
  `docs/explanation/design_sklearn_api_over_lightning.md` rewritten.

Resolved IMPROVEMENT findings: `docs/reference/api.md` autoclasses
the six adapter classes; CHANGELOG `Added` entry calls out the
24-name surface and the adapter promotion; README + 3 how-to pages
+ 3 gallery scripts + `examples/quickstart.py` switched from
`from seq_sklearn.config.adapters import ...` to
`from seq_sklearn import ...` (the documented STABLE form); this
ledger added; `__init__.py` docstring updated to reference the
`## A3:` heading rather than a stale line range.

### S7 Round 3

Architecture-reviewer raised two CRITICAL, two IMPROVEMENT, one
NITPICK; code + qa + style all APPROVE with zero findings. Resolved
in commit `cfec103`:

- arch-C1 (CPU N7 silently records GPU number): added a
  `torch.cuda.is_available()` skip-gate inside the CPU test so it
  refuses to run on a CUDA-equipped box; release_checklist + CHANGELOG
  both document `CUDA_VISIBLE_DEVICES=''` as mandatory.
- arch-C2 (`imbalanced_classes.md` wrong adapter): the human-reading
  python snippet imported `SamplerConfig` (frozen pydantic family
  sub-config) instead of `SamplerParams` (BaseEstimator adapter), so
  `clone()` + `to_pydantic()` would have raised. Rewrote the import,
  the kwarg, and the prose references.
- arch-I1 (A1 docs tree stale): rewrote `docs/architecture.md:143-159`
  to match the live Diátaxis filesystem.
- arch-I2 (INTERNAL-tier reconciliation): added an explicit row to
  the requirements per-module stability table declaring
  `seq_sklearn.config.{optimizer, scheduler, loss, sampler}` INTERNAL.
- arch-N1 (test_public_api_surface line range): rewrote the
  `_INTERNAL_TIER_FORBIDDEN` docstring to reference the `## A3:`
  section semantically rather than coordinates.

### S7 Round 4 (CONSENSUS REACHED)

All four agents APPROVE with zero CRITICAL. Two NITPICKs cleared:

- code-N1: `test_n7_cpu_inference_latency` docstring command line
  prepended with `CUDA_VISIBLE_DEVICES=''` to match the skip-message
  and release_checklist.
- arch-N1: A1 docs-tree labels rewritten from `(design/)` to
  `(toctreed under design/)` since the files physically live at
  `docs/` top-level and `docs/design/index.md` is a toctree facade.

Two IMPROVEMENTs deferred with reasons:

- qa-I1 (extend `test_quickstart_block_imports_real_public_symbols`
  to all how-to plain-python blocks): deferred. Reason: the facade
  test + adapter unit tests + Sphinx-doctest `{testcode}` blocks
  already catch any rename in the named symbols. Adding a parametrized
  AST walker over `docs/how-to/*.md` would tighten coverage on
  illustrative blocks but the failure mode (a stale import in a
  how-to plain block) is low-severity. Track as a follow-up.
- arch-I1 (`docs/reference/config.md` callout): pre-existing,
  pre-R3. The pydantic family sub-config classes (`OptimizerConfig`,
  `SchedulerConfig`, `LossConfig`, `SamplerConfig`) are rendered via
  `autopydantic_model` on the reference page; with the new R3
  INTERNAL-tier row, a reader could mistake them for importable
  STABLE classes. A 2-3 line callout clarifying that these are the
  *spec* for the STABLE `*Params` adapter fields would close the
  loop. Deferred to the next docs-polish pass.

CONSENSUS REACHED after 4 rounds. Final tally: 0 CRITICAL,
0 IMPROVEMENT outstanding (2 deferred with reasons), 0 NITPICK.
Cleared for the gated Gemini code final-pass (deferred per user
direction); v1.0.0-blocking work on Phase 12 is complete.
