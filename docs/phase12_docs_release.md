# Phase 12: Documentation and release prep (plan)

## Requirements

The grading rubric. Every swarm finding must trace to one of these,
to a fundamental correctness concern, or to a cited convergent finding
in `docs/docs_strategy_research.md` (the research synthesis this plan
is modeled on; referenced below as RS-Cn / RS-Dn / RS-Sn).

- **R1** Ship the `mkdocs.yml` site per architecture A12 (mkdocs +
  mkdocs-material + mkdocstrings[python] + griffe-pydantic +
  gen-files + literate-nav + section-index), the `[docs]` extra
  pinned exactly as A12 specifies, and the A12 mkdocstrings options
  recipe.
- **R2** `mkdocs build --strict` passes (A12 `validation:` block: nav
  omitted/not_found/absolute, links not_found/anchors/absolute,
  unresolved mkdocstrings xrefs) and is wired as a REAL PR-CI gate
  replacing the current `pr.yml` no-op docs job.
- **R3** Every documentation code snippet is executed in CI (RS-C2):
  `pytest --doctest-modules src/seq_sklearn/` (requirements N1) for
  docstrings PLUS a Markdown-snippet runner for the prose pages; a
  green snippet suite is a required PR gate. No stale-example path.
- **R4** The N1 quickstart-in-CI contract holds: `docs/index.md` and
  `README.md` mirror the single `examples/quickstart.py` that
  `tests/e2e/test_quickstart.py` imports; the three cannot drift.
  Two quickstart examples exist and pass in CI (acceptance criterion
  10: a binary classifier recovering accuracy >= 0.75 three-seed
  median; a quantile regressor recovering 80% interval coverage in
  [0.75, 0.85]).
- **R5** The site is organized on the Diátaxis four-mode IA
  (RS-C3): tutorial / how-to / reference / explanation, every page
  with exactly one home, dense bidirectional cross-linking (RS-SD).
- **R6** Time-to-first-success (RS-C1) and first-screen value prop
  (RS-C5): `docs/index.md` opens with the runnable zero-setup
  `fit`/`predict` (bundled/synthetic data, no download, no GPU, no
  credentials) above the fold, preceded by a one-screen value prop.
- **R7** The reference quadrant auto-generates: pydantic config
  classes as griffe-pydantic field/validator tables; the estimator
  API in the established docstring convention with the Lim et al.
  2021 TFT citation; a Glossary as the canonical sklearn-contract
  page that API and guides link into (RS-SE).
- **R8** The domain pages that address seq-sklearn's specific misuse
  risks exist: data-format contract (RS-DA), explicit
  "not-a-forecaster" / classification-first framing (RS-DB),
  time-series splitting & leakage (RS-DD), determinism/reproducibility
  reference (RS-DE), ONNX export round-trip how-to (RS-DG),
  goal-framed tuning guide (RS-DH), `observability.md` F11 event
  reference (implementation_plan Phase 12).
- **R9** Release maturity signals (RS-C6): `CHANGELOG.md` carries the
  v1.0.0 entry (Keep a Changelog format, already in use); a
  versioning + deprecation-policy page states the public-API SemVer
  guarantee and deprecation window; the migration-guide TEMPLATE
  exists (even though 1.0.0 has nothing to migrate). README badges,
  CONTRIBUTING, LICENSE, SECURITY already shipped in Phase 0.
- **R10** Repo rules hold: ruff/pyright clean on any new `.py`
  (example scripts, gen-files script); no `print` in library code
  (examples may print); the docs build does not regress the
  `pytest -m "not slow and not perf"` default suite; the
  criterion-9 N7-absolute release-checklist step (wired in Phase 11)
  is enumerated in the release checklist.
- **R11 (done-when, implementation_plan Phase 12)** The docs site
  renders, the API reference shows the pydantic field tables via
  griffe-pydantic, `pytest --doctest-modules src/seq_sklearn/`
  passes, the quickstart-in-CI passes, and the release checklist
  (acceptance criteria 1-11, including the criterion-9 step) is
  enumerated and each item has an owning artifact or CI job.

## Scope and non-goals

- **Scope**: a v1.0.0 documentation site (Diátaxis IA, mkdocs-material
  A12 stack), every snippet CI-tested, the reference auto-generated
  from pydantic + docstrings, the domain pages that prevent misuse,
  the README/index rewrite, the v1.0.0 CHANGELOG + versioning &
  deprecation policy + migration template, and the enumerated release
  checklist. This phase produces a SHIPPABLE v1.0.0 doc set.
- **NG1 Rendered executable examples GALLERY** (sphinx-gallery-style
  thumbnail grid with captured plots). The one gap in the mkdocs
  stack (RS tooling verdict). v1.0.0 ships a SMALL curated set of
  runnable, CI-tested example scripts under `examples/` surfaced as
  plain doc pages; the rendered gallery (mkdocs-gallery / notebook
  pages) is Deferred D1.
- **NG2 Versioned doc site via `mike`** + version selector. v1.0.0 is
  a single version; `site_url` is set now so the selector can be
  added with no content change at the first post-1.0 release.
  Deferred D2.
- **NG3 Migration codemod** (a `bump-pydantic` analog). Nothing to
  migrate at 1.0.0; the migration-guide template + deprecation policy
  ship now, the codemod at the first breaking change. Deferred D3.
- **NG4 Social-proof / "who uses it"** logo grid. No adopters yet;
  a placeholder section only. Deferred D4.
- **NG5 Hosting/deploy of the site** (gh-pages publish workflow). The
  build + strict gate is in scope; the publish-to-gh-pages action is
  a release-time op, Deferred D5 (the v1.0.0 release itself is the
  follow-on to this phase, not Phase 12's deliverable).
- **NG6** No content rewrite of `requirements.md` / `architecture.md`
  / the phase plans / research docs; they are hosted as-is under a
  "Design & decisions" section per A12 (`docs/research/*` already
  exists). Their prose is out of scope for style review here.

## P-A: Site scaffold (R1 / R2 / A12)

- **PA.1** `mkdocs.yml` at repo root: `site_name`, `site_url` (set to
  the eventual canonical URL now so NG2's selector is a no-op
  content-wise later), `theme: material` (instant nav, copy buttons,
  dark/light toggle, content tabs, admonitions, built-in search),
  `repo_url`, the A12 `plugins:` block VERBATIM (mkdocstrings python
  handler with the A12 options recipe incl. `extensions:
  [griffe_pydantic]`, gen-files, literate-nav, section-index), and
  the A12 `validation:` block so `--strict` fails on
  nav/link/xref defects.
- **PA.2** The `[docs]` extra in `pyproject.toml` pinned EXACTLY as
  A12 lists (mkdocs>=1.6,<2; mkdocs-material>=9.7,<10;
  mkdocstrings[python]>=0.27; griffe-pydantic>=1.3;
  mkdocs-gen-files>=0.5; mkdocs-literate-nav>=0.6;
  mkdocs-section-index>=0.3). Plus the Markdown-snippet test runner
  dep (PD.2) in the `dev` extra.
- **PA.3** Site tree per A12 layout, organized by Diátaxis (R5):
  - `index.md` (value prop + above-the-fold quickstart, R6)
  - `tutorial/` (one linear finishable spine)
  - `how-to/` (task-named guides)
  - `reference/` (auto-generated API + config tables + glossary)
  - `explanation/` (the "why" docs)
  - `about/` (changelog, versioning & deprecation policy, migration
    template, release checklist)
  - `design/` (requirements.md, architecture.md, research/*, phase
    plans hosted as-is, NG6)
  The Diátaxis bucket names are the nav sections; A12's flat
  `guides/` list maps INTO `how-to/` + `explanation/`.

## P-B: The four Diátaxis quadrants

- **PB.1 Tutorial** `tutorial/first_classifier.md`: one linear
  top-to-bottom narrative, no decisions asked, on a bundled/synthetic
  panel: build config -> fit -> predict/predict_proba -> evaluate ->
  read variable-selection weights. Ends labeled "this alone is
  enough; advanced topics are in How-to/Explanation." Every block
  CI-tested (R3). This is distinct from the index quickstart (which
  is the 30-second hook); the tutorial is the 15-minute teach.
- **PB.2 How-to** `how-to/*.md`, task-named (RS Keras convention),
  each a focused recipe with a tested snippet: `configure_training`
  (the pydantic config), `imbalanced_classes`, `extract_attention`
  (variable-selection / attention surfaces), `tune_with_optuna`,
  `export_onnx` (RS-DG, the Phase 10 capability's user guide:
  full export -> reload-in-onnxruntime round trip),
  `persist_a_model` (RS-SC: joblib vs ONNX decision + the "never
  load untrusted pickles" + version-pin/containerize warnings),
  `choose_lookback_and_min_periods`.
- **PB.3 Reference** (R7, auto-generated, neutral):
  - `reference/config.md` literate-nav + mkdocstrings +
    griffe-pydantic over the pydantic config classes (the dominant
    reference surface). Field tables: name / type / default /
    constraint / `Field(description=...)` / validators, respecting
    the `TFTConfig <- BaseModelConfig <- BaseTrainingConfig`
    inheritance via `inherited_members: true`.
  - `reference/api.md` mkdocstrings over the estimator public API;
    the docstring convention already in `src/` (Parameters /
    Attributes / Returns / Raises / Examples), the Lim et al. 2021
    TFT paper cited in the TFT class docstring References.
  - `reference/glossary.md` the canonical sklearn-contract page:
    fit/predict/predict_proba, trailing-`_` fitted attrs,
    `classes_`, target types, `random_state`/`n_jobs`, panel terms
    (entity, period, lookback, min_periods). API + guides link here
    (RS-SD/SE).
- **PB.4 Explanation** `explanation/*.md`: `why_tft_for_classification`
  (with the explicit "seq-sklearn is NOT a forecasting library"
  statement, RS-DB), `how_variable_selection_and_attention_work`,
  `design_sklearn_api_over_lightning`, `determinism_and_precision`
  (the model behind RS-DE's reference page).

## P-C: Domain pages (R8, the misuse-risk set)

- **PC.1 `how-to/panel_data.md` (data-format contract, RS-DA, top
  priority).** The exact panel shape, WHY consecutive rows are
  consecutive periods regardless of wall-clock (requirements F2),
  the ragged / variable-history case, a worked example with PRINTED
  shapes. The single biggest misuse source.
- **PC.2 Not-a-forecaster framing (RS-DB).** A first-screen line in
  `index.md` AND a section in
  `explanation/why_tft_for_classification.md`. The README already
  states this; carry it into the site prominently.
- **PC.3 `how-to/time_series_splitting.md` (RS-DD).** Entity-time
  expanding-window CV, why random splits leak on multi-entity panels
  (requirements F2), how to use the library's splitter; Wrong-vs-Right
  runnable form (RS-SA: this IS the Common-Pitfalls page, temporal
  leakage first).
- **PC.4 `reference/determinism.md` (RS-DE).** Standalone page:
  seeding, the strict-mode side effects, CUDA flags, the explicit
  "not bit-exact across versions/platforms" caveat + the perf cost.
- **PC.5 `how-to/performance.md` (RS-DF).** Scannable few-line-wins:
  precision, GPU vs CPU, batch size / throughput / memory.
- **PC.6 `how-to/tuning.md` (RS-DH).** Goal-framed ("better accuracy
  / faster / less overfitting"), names the ~5 config fields that
  matter most, every tradeoff an inline win/lose sentence, points at
  the Optuna integration rather than reinventing HPO advice. Pairs
  with the auto-generated `reference/config.md` (split per RS-DH).
- **PC.7 `reference/observability.md`.** The F11 event-payload
  reference (implementation_plan Phase 12 deliverable).

## P-D: Snippet-execution gate (R3, the trust gate)

- **PD.1** Docstring snippets: `pytest --doctest-modules
  src/seq_sklearn/` (requirements N1) runs in the default PR suite.
  Already a requirement; this phase ensures every example added to a
  docstring is doctest-valid and the job is enforced.
- **PD.2** Prose snippets: a Markdown-snippet runner
  (`pytest-markdown-docs` or `mktestdocs`, RS evidence) collects
  every fenced ```python block under `docs/` and runs it as a test,
  with `memory=True` semantics for sequential stateful blocks. Added
  to the `dev` extra and to the PR suite. Snippets that are
  illustrative-not-runnable use a documented fence info-string the
  runner skips (e.g. ```python title="..." or a `notest` marker),
  and a meta-test asserts the skip-marker count is bounded so authors
  cannot silently opt everything out.
- **PD.3** CI: the `pr.yml` docs job stops being a no-op. It
  `uv sync --extra docs --extra dev`, runs `mkdocs build --strict`,
  `pytest --doctest-modules src/seq_sklearn/`, and the Markdown
  snippet suite. All three are required. Respects the 5-minute PR
  budget (build + doctests are fast; no perf/e2e-slow here; the
  quickstart e2e already runs in the unit job per N1).
- **PD.4 Tests for the gate itself** (so it cannot silently rot):
  - `test_mkdocs_build_strict`: invokes `mkdocs build --strict` in a
    tmp dir, asserts exit 0 (skipped if the `docs` extra absent).
  - `test_quickstart_mirrors_index_and_readme`: asserts the code in
    `examples/quickstart.py` appears verbatim in both `docs/index.md`
    and `README.md` (R4 anti-drift, mechanical).
  - `test_every_doc_has_a_diataxis_home`: asserts every `docs/**/*.md`
    is reachable from `mkdocs.yml` nav (no orphan pages; complements
    `--strict` nav validation with an explicit assertion).

## P-E: Release readiness (R9 / R11)

- **PE.1** `CHANGELOG.md`: convert `[Unreleased]` to `[1.0.0] -
  <date>` with the full Keep-a-Changelog section set
  (Added/Changed/Deprecated/Removed/Fixed/Security), summarizing
  F1-F11 + N1-N7 delivery. A new empty `[Unreleased]` header on top.
- **PE.2** `about/versioning.md`: public-API SemVer guarantee, the
  deprecation window (deprecate in a minor with a runtime
  `DeprecationWarning` pointing at the replacement; remove no earlier
  than the next major), what counts as public API (the estimator
  classes + pydantic configs + documented functions; `_`-prefixed is
  private). `about/migration_template.md`: the empty-but-structured
  template (changes grouped by surface area, before/after per renamed
  symbol, deprecation-warning policy, optional codemod slot) so the
  first breaking release has a form to fill, not a blank page.
- **PE.3** `about/release_checklist.md`: acceptance criteria 1-11
  enumerated, each mapped to its owning CI job or artifact, INCLUDING
  the criterion-9 manual `pytest -m "gpu and slow"
  tests/perf/test_n7_absolute.py` step wired in Phase 11 and recorded
  in implementation_plan Phase 12. This is the document the v1.0.0
  release (the follow-on, NG5) executes.

## P-F: Open questions for the swarm

- **Q1** Markdown-snippet runner choice: `pytest-markdown-docs` vs
  `mktestdocs`. Both run fenced blocks under pytest; which has the
  cleaner stateful-block + skip-marker story for our prose pages?
  (PD.2)
- **Q2** Tutorial vs index-quickstart split: is a separate 15-minute
  `tutorial/first_classifier.md` worth its maintenance cost on top of
  the 30-second `index.md` quickstart, or does one canonical example
  with progressive sections serve both (FastAPI single-spine model)?
  (PB.1)
- **Q3** Should `requirements.md` / `architecture.md` (large internal
  design docs) be hosted in the site at all (NG6 says yes, as-is
  under design/), or excluded from the user site and left repo-only
  to avoid confusing users with internal governance prose?
- **Q4** Does `mkdocs build --strict` belong in the same PR job as
  the snippet tests, or a separate job? Trade-off: one
  `uv sync --extra docs` vs CI parallelism within the 5-minute
  budget. (PD.3)

## Risks

- **RK1** `mkdocs build --strict` is brittle: every unresolved
  mkdocstrings xref fails the build. Mitigated by building
  incrementally and the `test_mkdocs_build_strict` gate catching it
  pre-merge, not at release.
- **RK2** Doctest snippets that need a fitted model are slow (TFT fit
  on CPU). Mitigated by docstring examples using the smallest
  synthetic panel + `max_epochs=1`, and the heavy end-to-end staying
  in the existing e2e quickstart (not duplicated into docstrings).
- **RK3** griffe-pydantic rendering of the deep config inheritance
  chain may mis-render `inherited_members`; A12 already pins the
  recipe, PB.3 verifies the rendered table in the strict build.

## Deferred (v1.1+, with reason)

- **D1** Rendered executable examples gallery (mkdocs-gallery /
  notebook pages): the one mkdocs-stack gap; v1.0.0 ships curated
  CI-tested example scripts instead, gallery is non-blocking polish.
- **D2** `mike` versioned site + selector: one version at 1.0.0;
  `site_url` wired now so it is a zero-content-change add later.
- **D3** Migration codemod: nothing to migrate at 1.0.0; template +
  policy ship now.
- **D4** Social-proof / who-uses-it grid: no adopters yet.
- **D5** gh-pages publish workflow: a release-time op, belongs to the
  v1.0.0 release that follows this phase, not Phase 12.
- **D6** Release-highlights narrative posts: start at the first minor
  after 1.0.0 (RS-C6 norm).

## Tracking (review loop)

Addressed and deferred items are maintained here so successive swarm
runs see prior decisions and do not re-raise resolved points.

(populated by `/design-review`)
