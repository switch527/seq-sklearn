# Documentation strategy research (drives the Phase 12 plan)

Synthesis of four parallel research passes on the documentation of the
most-adopted ML / Python libraries: scikit-learn; Hugging Face
Transformers, FastAPI, Pydantic; PyTorch / Lightning / Keras / XGBoost
/ LightGBM / sktime / aeon / Darts; and cross-cutting framework +
tooling + adoption-mechanics evidence. This document is the grading
rubric the Phase 12 plan is modeled on. Every recommendation traces to
a cited source in the agent reports.

## What every successful library converges on

These appeared independently in all four passes. They are not
optional; they are the price of adoption.

- **C1 Time-to-first-success above the fold.** A runnable, copy-paste
  `fit`/`predict` on a BUNDLED or synthetic dataset, on the homepage,
  with no credentials, no dataset download, no GPU. Under a minute from
  landing to "it worked" is the single strongest adoption predictor
  (sklearn Getting Started; Pydantic get-started; FastAPI first-steps;
  Stripe/Twilio teardowns). HF Transformers is the cautionary case: it
  buries the win behind account + token + 4 installs.
- **C2 Every documentation snippet is CI-executed.** The #1 trust
  killer and the #1 OSS-doc complaint is stale/broken examples. sklearn
  doctests every class snippet; FastAPI's tutorial blocks "are actually
  tested Python files"; NumPy/pandas enforce `--doctest-modules`. This
  is a hard gate, not a nicety. seq-sklearn already mandates
  `pytest --doctest-modules` (requirements N1), extend it to the
  Markdown prose snippets too.
- **C3 Diátaxis information architecture.** tutorial / how-to /
  reference / explanation, four modes, each page has exactly one home.
  Adopted explicitly by Canonical, Django, LangChain, HF Transformers.
  Prevents the "API-reference-only, no narrative" failure that kills
  evaluation.
- **C4 Tooling: mkdocs-material + mkdocstrings + griffe-pydantic +
  mike.** Both tooling passes converged here independently. Decisive
  reason: seq-sklearn is pydantic-v2-config-heavy and `griffe-pydantic`
  renders config schemas as field/constraint/validator tables with
  near-zero effort, the dominant reference surface for this library.
  It is also the proven stack of our nearest dependency-graph peers
  (pydantic, FastAPI). This CONFIRMS architecture A12's existing
  choice. One accepted caveat: no turnkey executable gallery (see D-list).
- **C5 Value proposition in the first screen.** One sharp claim + one
  quantified proof + recognizable social proof. Pydantic leads with
  download counts + a logo grid; FastAPI with benchmark numbers +
  named-company quotes. seq-sklearn's analog: "scikit-learn-compatible
  Temporal Fusion Transformer for time-series classification" + a
  benchmark table vs GBM/classical-MTSC baselines + the interpretable
  variable-selection/attention surfaces as the differentiator.
- **C6 Versioned docs + Keep-a-Changelog + public SemVer + deprecation
  policy.** Practitioners assess adoption RISK from these before they
  adopt. Shipping 1.0.0 is itself a maturity signal. `mike` gives a
  versioned site + selector with a one-line CI step.

## The seq-sklearn information architecture (Diátaxis, concrete)

- **Tutorial (learning, one linear finishable spine, labeled
  "this alone is enough"):** "Train your first TFT classifier on a
  bundled toy multivariate panel, evaluate it, and read its
  variable-selection weights." Runs top to bottom, asks the reader no
  decisions. FastAPI's single-spine model; advanced TFT internals
  quarantined out of it.
- **How-to (work, task-named not API-named):** configure the pydantic
  training config; handle imbalanced classes; extract attention /
  variable-selection; tune with the Optuna integration; export to
  ONNX; persist/load a fitted model; pick lookback / min_periods. Keras
  "Developer Guides" naming convention.
- **Reference (auto-generated, neutral):** the pydantic config classes
  as griffe-pydantic field tables (default / constraint / effect);
  the estimator API (numpydoc sections: Parameters / Attributes /
  See Also / References / Examples; inline `versionadded` /
  `versionchanged` / `deprecated`; the Lim et al. 2021 TFT paper in
  References); a **Glossary** as the canonical contract (fit/predict/
  predict_proba, `classes_`, trailing-underscore attrs, target types,
  random_state/n_jobs semantics) that API + guides link into.
- **Explanation (study, the "why"):** why a TFT for classification (and
  the explicit "we are NOT a forecaster" statement); how
  variable-selection and attention work; the scikit-learn-API-over-
  Lightning design; the determinism/precision model.

## Domain-specific pages the generic frameworks do not name

From the DL / time-series pass. These address seq-sklearn's exact
user-confusion points and are higher-impact than generic polish:

- **D-A Data-format contract page (top priority).** Exact panel shape,
  WHY consecutive rows are treated as consecutive periods regardless of
  wall-clock (requirements F2), the ragged / unequal-length case, and a
  worked example with PRINTED shapes (aeon's model). This is our single
  biggest source of misuse.
- **D-B Classification-first, task-partitioned IA** with an explicit
  up-front "seq-sklearn is not a forecasting library" (aeon/sktime
  prevent the category error this way).
- **D-C Named feature taxonomy + support matrix:** static vs
  time-varying (real/categorical), one table mapping each to where it
  goes in the config, plus documented entity-time alignment (Darts
  covariates model).
- **D-D Dedicated "splitting time-series data" page:** the entity-time
  expanding-window CV, leakage, why random splits leak on multi-entity
  panels (requirements F2). Its own first-class topic.
- **D-E Standalone Reproducibility/Determinism reference page:**
  seeding, the strict-mode side effects, CUDA flags, the "not
  bit-exact across versions/platforms" caveat + the perf cost (PyTorch
  `notes/randomness` is the model; this is also acceptance criterion 9
  adjacent).
- **D-F Performance/precision tuning as a single scannable how-to:**
  mixed precision, GPU vs CPU, batch size / throughput, memory, framed
  as few-line wins (PyTorch tuning-guide recipe).
- **D-G ONNX export guide with a full runnable export -> reload-in-
  onnxruntime round trip** (Phase 10 shipped the capability; this is
  its user-facing guide).
- **D-H Config docs = flat auto-generated Parameter Reference +
  separate goal-framed Tuning Guide.** Reference is griffe-pydantic
  (free). Tuning guide organized by GOAL ("better accuracy / faster /
  less overfitting"), names the ~5 fields that matter most, every
  tradeoff stated as an inline win/lose sentence (LightGBM house
  style), and points at the Optuna integration rather than reinventing
  HPO advice.
- **D-I Progressive-disclosure skill ladder:** beginners never see
  precision/multi-GPU/determinism; these surface at an
  intermediate/advanced tier (Lightning's explicit level ladder).

## scikit-learn-specific (our API is sklearn-compatible)

- **S-A "Common Pitfalls", Wrong-vs-Right runnable form,** leading with
  TEMPORAL leakage and CV-on-sequences (sklearn's page leads with
  leakage; ours must, given the domain).
- **S-B "Developing a compatible / interoperable estimator" page:**
  documents the `__init__`-stores-verbatim / `fit`-returns-self-sets-
  `*_` / get_params / estimator-tags / `check_estimator` contract. For
  a third-party sklearn-compatible library this IS the interoperability
  promise; pair it with the shipped `check_estimator` test.
- **S-C Model-persistence page with a decision tree + explicit security
  ("never load untrusted pickles") and version-pin / containerize
  warnings.** Lightning/torch models are heavyweight; users need
  joblib vs ONNX guidance up front.
- **S-D Dense bidirectional cross-linking** API <-> guide <->
  example <-> glossary. The compounding effect that makes a doc set
  feel authoritative.
- **S-E numpydoc section order + inline version directives** so the
  zero-learning-curve sklearn audience is immediately at home.

## Adoption / trust mechanics

- Homepage: task-oriented funnel (capability + "who/why"), not a doc
  tree. One quantified proof (the benchmark table) when available.
- One-click copy buttons on every snippet; downloadable complete
  samples; a Colab/notebook badge on the quickstart to remove local-
  install friction for the first run.
- Migration-guide template ready before the first breaking release:
  changes grouped by surface area, before/after for every renamed
  symbol, a deprecation-warning policy (old API warns + points at the
  replacement), optionally a codemod (pydantic's `bump-pydantic` is the
  model). Designed now even though v1.0.0 has nothing to migrate yet.
- Project-health signals shipped at 1.0.0: CI + coverage badges,
  CONTRIBUTING, LICENSE, issue templates, a visible changelog and
  version selector. Empirically separates adopted from abandoned OSS.

## Phase 12 priority (what to build now) vs deferred (v1.1+)

**Build in Phase 12 (ranked by adoption impact):**

1. C2 every snippet CI-tested (doctest-modules + a Markdown-snippet
   runner) as a required gate, plus `mkdocs build --strict`.
2. C1 homepage + Getting Started: zero-setup runnable `fit`/`predict`
   above the fold; C5 value-prop first screen.
3. C4 mkdocs-material + mkdocstrings + griffe-pydantic site scaffold
   (confirms A12); C3 Diátaxis nav.
4. Reference: griffe-pydantic config tables + numpydoc estimator API +
   Glossary (S-E, reference quadrant).
5. D-A data-format contract + D-B classification-first / not-a-
   forecaster + D-D splitting page (our top misuse risks).
6. The how-to set: config, imbalance, attention extraction, Optuna,
   ONNX (D-G), persistence (S-C); D-H tuning guide.
7. S-A Common Pitfalls (temporal leakage first); S-B compatible-
   estimator page; D-E determinism page; D-F perf page.
8. C6 Keep-a-Changelog + SemVer statement + deprecation-policy page;
   README rewrite (one-screen quickstart that the e2e test imports);
   the criterion-9 N7-absolute release-checklist step (already wired
   in Phase 11).

**Deferred to v1.1+ with reason (carry in the plan's Deferred
section):**

- Versioned doc site via `mike` + selector: valuable but v1.0.0 has a
  single version; wire the `site_url` now, add `mike` at the first
  post-1.0 release.
- Executable examples GALLERY (mkdocs-gallery / notebook pages): the
  one gap in the mkdocs stack and the hardest piece. v1.0.0 ships a
  small curated set of runnable example scripts under `docs/examples/`
  (tested by C2); the rendered thumbnail gallery is v1.1.
- Migration codemod (`bump-pydantic` analog): nothing to migrate at
  1.0.0; ship the migration-guide TEMPLATE + deprecation policy now,
  the codemod at the first breaking change.
- Social-proof logo grid / "who uses it": no adopters yet; leave a
  placeholder section, populate post-release.
- Release-highlights narrative posts: starts at the first minor after
  1.0.0.

## Tooling verdict

> SUPERSEDED at Phase 12 design-review R1. The research below leaned
> mkdocs-material; the user RATIFIED **Sphinx + numpydoc +
> sphinx-gallery + PyData theme + Read the Docs** instead, on the
> "scientific-Python credibility optics" argument this section itself
> names: the direct peer cluster (scikit-learn, sktime, aeon, skorch,
> tslearn, darts) is near-unanimously Sphinx, so matching it is a
> first-order adoption signal to exactly the target audience, and
> sphinx-gallery closes the executable-gallery gap that was the main
> mkdocs caveat. Every FRAMEWORK-AGNOSTIC finding above
> (time-to-first-success, CI-tested snippets, Diátaxis IA,
> first-screen value prop, versioned/changelog/SemVer, the domain
> pages, the sklearn-parity pages) stands unchanged; only the
> rendering toolchain flips. pydantic field tables now render via
> `autodoc-pydantic` (Sphinx analog of griffe-pydantic). See
> architecture A12 + requirements Q12/Q16 (reconciled).

Original (pre-decision) verdict, retained for the record:
**mkdocs-material + mkdocstrings + griffe-pydantic + (mike
deferred)**, matching the then-current architecture A12. Two caveats
were flagged: no turnkey `sphinx-gallery`, and HTML-only output.
Sphinx was named the fallback "if the project later needs
scientific-Python credibility optics or a very large executable
gallery", the user invoked exactly that.
