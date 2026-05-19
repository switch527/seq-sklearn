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
  by THREE mechanisms with no silent-skip path: (a) docstring
  snippets via `pytest --doctest-modules src/seq_sklearn/`
  (requirements N1), added to the `pr.yml` unit job invocation
  (qa-N2); (b) reST/MyST prose snippets via `sphinx.ext.doctest`
  (`.. testcode::` / `.. doctest::`) run by `sphinx-build -b doctest`
  in the docs job; (c) `docs/examples/*.py` executed at build by
  `sphinx-gallery` (a failing example fails the build). A bounded
  non-executable-block meta-test (PD.4) caps how much can opt out.
- **R4** The N1 quickstart anti-drift holds WITHOUT fragile
  verbatim-substring matching (arch-C2: `examples/quickstart.py` is a
  `def run_quickstart() -> float` consumed by `importlib` in
  `tests/e2e/test_quickstart.py`; a print-ending flat script it is
  not). Mechanism: `examples/quickstart.py` is the SINGLE source;
  `docs/index.md` embeds it by reference via
  `literalinclude`/`literal-block` (zero drift by construction);
  `README.md` carries the same code in a fenced block and
  `test_quickstart_readme_in_sync` asserts the README block is
  AST-equivalent to the `run_quickstart` body (normalized compare,
  not whitespace-exact). Two quickstart examples exist and pass in
  CI (acceptance criterion 10).
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

## P-0: Stack reconciliation (arch-C1, landed in this change set)

The Sphinx-vs-mkdocs contradiction the R1 swarm flagged is resolved
by editing the authoritative docs, not by the plan asserting a
winner: A12 rewritten to Sphinx; requirements N6/Q12/Q16 + `[docs]`
extra reconciled; `readme_and_docs_plan.md` marked ratified;
`docs_strategy_research.md` verdict annotated superseded. The
existing `docs/conf.py` + `docs/api/generated/*.rst` are kept and
extended. `docs/_build/` is added to `.gitignore` (a build artifact
must never be committed; PA.4).

## P-A: Site scaffold (R1 / R2)

- **PA.1** `docs/conf.py`: extend the existing skeleton to the A12
  recipe (extensions list, `autosummary_generate=True`,
  `numpydoc_show_class_members=False`, autodoc-pydantic field/
  validator summaries on, the `intersphinx_mapping` for
  sklearn/numpy/torch/pydantic, `sphinx_gallery_conf` pointing at
  `docs/examples/`, copybutton, myst-parser, `nitpicky=True` so
  missing xrefs are warnings `-W` turns to errors).
- **PA.2** `[docs]` extra in `pyproject.toml` pinned EXACTLY as A12
  now lists (`sphinx>=8,<9`, `pydata-sphinx-theme>=0.16`,
  `numpydoc>=1.8`, `sphinx-gallery>=0.18`, `autodoc-pydantic>=2.2`,
  `sphinx-copybutton>=0.5`, `myst-parser>=4`).
- **PA.3** Diátaxis toctree (A12 layout): the existing MyST
  `docs/index.md` becomes the value-prop + quickstart landing; the
  35 existing `docs/api/generated/*.rst` autosummary stubs move under
  `reference/`; new `tutorial/ how-to/ explanation/ about/ design/`
  trees. Root `index.md` toctree wires every page (no orphans).
- **PA.4** `.gitignore` gains `docs/_build/`; the committed
  `docs/_build/html/` tree is removed (build output, never tracked).
- **PA.5** `.readthedocs.yaml`: build config (python version, the
  `[docs]` extra, `sphinx` builder, `fail_on_warning: true` mirroring
  the CI `-W` gate) + RTD project versioning (stable / latest). This
  is R9's versioned hosting; RTD does it natively (no `mike`).

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
  - `test_quickstart_readme_in_sync`: parse the fenced python block
    in `README.md`, parse `examples/quickstart.py`, assert the
    README block is AST-equivalent to the `run_quickstart` body
    (normalized, not whitespace-exact; arch-C2/qa-C4). Fails (not
    skips) if either is missing. `docs/index` uses `literalinclude`
    so it cannot drift and needs no test beyond the build.
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

## P-E: Release readiness (R9 / R11 / R12)

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

## P-F: Open questions for the swarm

- **Q1** Keep a separate 15-minute `tutorial/first_classifier` on top
  of the 30-second `index` quickstart, or one progressive example
  serving both (FastAPI single-spine; the R1-qa lean was single)?
- **Q2** `docs/index` and the gallery both surface
  `examples/quickstart.py` via `literalinclude`/gallery; is that
  double-surfacing confusing, or correct (hook on index, full run in
  gallery)?
- **Q3** Host the large internal design docs (`requirements.md`,
  `architecture.md`, phase plans) in-site under `design/` (NG1) with
  a callout, or exclude from the user site entirely and keep
  repo-only? R1-qa/arch leaned host-with-callout.
- **Q4** Single docs CI job (one `uv sync`, build + doctest
  sequential) vs split (parallel, two syncs)? Default single; revisit
  only if the 5-minute budget is breached.

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

Gemini design final-pass: deferred until quota resets; recorded here
when run (R11).
