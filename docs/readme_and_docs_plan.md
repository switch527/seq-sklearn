# README and documentation plan: seq-sklearn

Goal: maximize GitHub stars, real adoption, and durable discoverability
for a library that fills a genuine gap (sklearn-compatible deep sequence
models for supervised classification and regression on tabular panel
data, not forecasting). Plan is evidence-based: two research passes over
22 high-traction ML repos and their doc sites. Sources cited inline.

## 1. Strategic framing

The growth flywheel for this specific niche is not generic. The
time-series-ML cluster (sktime, darts, tslearn, pyts, aeon, river,
skorch) all grow through the same loop: a sharp gap-filling hook in the
README, a sklearn-compatible `.fit().predict()` proof, a citable paper
plus DOI, and inclusion in the sklearn ecosystem surfaces. seq-sklearn
maps onto this loop almost exactly, with one extra asset competitors
lack: TFT's interpretable variable-selection and attention surfaces, a
figure that sells the library on sight.

Two hard constraints shape the plan:

1. The repo is pre-implementation (phase-1 foundation). The
   highest-leverage README assets (a runnable <15-line quickstart, a
   rendered attention figure, a paper-backed number) cannot be faked.
   The plan is therefore **staged**: a credible pre-1.0 README now, a
   launch README at v1 ship.
2. The "narrative" the research recommends is benefit-led and confident,
   not flowery. Target register is ruff's README: zero banned
   vocabulary, no em dashes, declarative. This is compatible with the
   repo's anti-tell style rules; do not relax them for marketing copy.

## 2. README structure (v1 launch target)

Exact top-to-bottom order. Items 1-6 must sit above the first scroll.

1. **Wordmark / logo**, centered. A simple SVG wordmark is enough.
   Absence reads as immature (19 of 22 studied repos have one).
2. **Tagline, one sentence.** skorch pattern (API + niche in one
   breath), differentiator first. The tagline must name the *library*,
   not the first model; TFT is v1's first concrete model, not the
   library's identity. Draft:
   > Modern deep sequence models, as easy to use as scikit-learn: one
   > fit / predict API for classification and regression on multivariate
   > time series, across the transformer and recurrent model families.
   TFT-specific strengths (interpretable variable selection, attention)
   belong in the v1 features list, not the headline.
3. **Badge row.** PyPI version, Python versions, License, CI, coverage,
   and a **DOI/Zenodo badge from day one**. The DOI is not decoration:
   a BibTeX + DOI citation loop is the dominant growth mechanic in this
   exact niche (darts, sktime, tslearn, pyts, aeon, river all do it; the
   "papers with code" diffusion study finds ~20% higher citation rate
   once a repo exists). Omit a downloads badge until numbers are
   non-trivial; a low number signals neglect.
4. **One-paragraph what + why (3-4 sentences), narrative tone.** State
   the gap explicitly and concretely: TFT is published and powerful, but
   using it normally means dozens of lines of dataloader/trainer wiring
   (literally true of pytorch-forecasting's ~95-line quickstart).
   seq-sklearn collapses that to `fit`/`predict` while preserving the
   interpretability surfaces. Use aeon's credibility move: grounded in
   the published TFT architecture (Lim et al., 2021).
5. **Install.** Single `pip install seq-sklearn`. Conda line once
   conda-forge recipe lands. Note the `[onnx]` extra.
6. **Quickstart, the load-bearing section.** One copy-pasteable block,
   strictly under 15 lines, full lifecycle on the churn use case,
   ending in a concrete printed metric (river's `Accuracy: 0.89`
   pattern). Must literally show sklearn idioms:

   ```python
   from seq_sklearn import TFTClassifier

   clf = TFTClassifier(lookback=12, hidden_size=128)
   clf.fit(X_train, y_train)               # X: tidy panel DataFrame
   proba = clf.predict_proba(X_test)
   print(f"AUC: {roc_auc_score(y_test, proba[:, 1]):.3f}")
   ```

   If this block exceeds ~15 lines the library fails to demonstrate its
   own reason to exist. Highest-leverage decision in the README. Add a
   second 3-line block showing `Pipeline` / `GridSearchCV` composition,
   since sklearn-compatibility is the core claim and must be proven, not
   just stated.
7. **Interpretability figure.** One rendered attention-heatmap or
   variable-selection-weight plot from a real run, embedded inline.
   For an interpretability-focused library this outweighs any benchmark
   table (darts embeds a forecast plot, optuna a dashboard GIF for the
   same reason). This is the asset no competitor wrapper has.
8. **Why / features, bullets.** sklearn API (`Pipeline`,
   `GridSearchCV`, `cross_val_score`, Optuna), pytorch-lightning backend
   (CPU/GPU automatic), interpretable attention + variable selection,
   pydantic-typed configs, calibrated probabilities / conformal
   quantiles, ONNX export. Name the ecosystems you plug into (sklearn,
   lightning, Optuna) for reciprocal discovery.
9. **Capability matrix, short table.** Classifier vs regressor, static /
   known / observed covariates, probabilistic output, calibration. Scans
   in two seconds; every adjacent lib carries one.
10. **One paper-backed benchmark sentence, not a table.** Copy
    pytorch-forecasting's exact device: cite the TFT paper's headline
    result with the arXiv link. Do **not** build an in-README accuracy
    leaderboard; the entire ML-lib set deliberately avoids them (stale,
    contestable, credibility liability). A reproducible benchmark script
    lives in `docs/examples/`, not the README.
11. **Docs / tutorials link**, including the churn end-to-end example.
12. **Citation.** Full BibTeX for seq-sklearn plus a cite to Lim et al.
    2021. Niche's primary academic flywheel; not optional.
13. **Contributing + community.**
14. **License.**

Deliberately omitted: in-README accuracy leaderboard, star-history
chart, fabricated "used by" logos. Add a "Used by" section (the
strongest non-benchmark trust device, per ruff) only once real adopters
exist; it cannot be faked.

### Interim README (now, pre-implementation)

The current README is honest but sells nothing. Replace with: logo,
tagline (item 2), badge row minus coverage/downloads, the what+why
paragraph (item 4), the gap narrative, the roadmap table (keep), and a
**"Planned API" preview block** clearly labeled as not-yet-released
showing the target `fit`/`predict` snippet. This builds the hook and
lets early visitors star for the promise (sktime/aeon did this
pre-1.0). Swap to the full launch README at v1 ship. Do not show fake
benchmarks or a fake figure; label the API preview as forthcoming.

## 3. Documentation site plan

### Stack decision (resolves requirements.md N6 open question)

**Sphinx + numpydoc + autosummary/autodoc + sphinx-gallery, PyData
Sphinx Theme, NumPy-style docstrings, hosted on Read the Docs.**

Rationale: the sklearn-compatible, time-series-ML cluster is
near-unanimously Sphinx (sklearn, sktime, aeon, skorch, tslearn, darts,
pytorch-forecasting, Optuna). NumPy docstring style + numpydoc is the de
facto ecosystem standard and effectively mandatory to read as a sklearn
citizen. `intersphinx` gives free cross-references into sklearn / numpy /
torch / pandas object inventories, which a wrapper-style library needs
constantly and which mkdocstrings cannot match. `sphinx-gallery`
executes every example on build (docs + free regression test) and is
sklearn's single biggest long-tail SEO asset. RTD removes the
versioning CI burden for a small maintainer. MkDocs Material is the
better general-purpose 2025 choice but loses on every factor that
matters for *this* niche. This decision should feed the architecture
doc and the `/design-review` loop per the repo workflow, not be silently
adopted.

### Information architecture (the sklearn triad)

Install → User Guide → API Reference → Examples Gallery, plus Changelog
and Contributing. This is the structure sktime, aeon, skorch, tslearn,
darts, pytorch-forecasting all clone from sklearn; copying it signals
ecosystem membership to users who already know sklearn's docs.

- **User Guide**: concept-first prose. How TFT is adapted from
  forecasting to classification/regression, variable selection, the
  attention surfaces, calibration story, the panel data contract.
- **API Reference**: autogenerated from numpydoc docstrings
  (already required by repo rule 2).
- **Examples Gallery**: task-titled runnable scripts ("Churn
  classification with attention", "Calibrated quantile regression",
  "TFT in a sklearn Pipeline"). Task-shaped titles capture long-tail
  search; this is the biggest organic-traffic lever.

### Minimal viable v1 vs mature

- **v1 launch (~1-2 days):** Sphinx + pydata theme + numpydoc +
  autosummary; IA above with 3-5 User Guide pages; 3-5 runnable example
  scripts executed in CI (no full gallery yet); RTD on `*.readthedocs.io`
  with default stable/latest; `sphinx-sitemap` + Search Console.
- **Mature:** full sphinx-gallery thumbnail gallery grouped by task;
  topic-organized User Guide + theory + glossary + FAQ; custom domain on
  RTD; intersphinx mappings; example-execution gate in CI.

Versioning burden is essentially eliminated by choosing RTD over GitHub
Pages. Recurring cost after setup is near-zero because autosummary
regenerates from docstrings and RTD rebuilds on tag.

## 4. Promotion / launch checklist (ranked by ROI)

Tier 1, at or before launch:

1. **PyPI with full metadata** (project_urls, classifiers, keywords;
   README renders as the project page). Non-negotiable.
2. **scikit-learn "Related Projects" PR.** Highest ROI for this exact
   library; aeon, sktime, darts, tslearn, skorch are all listed there.
   Use the scikit-learn-contrib template conventions.
3. **conda-forge recipe.** Scientific-Python users default to conda;
   bot-automated upkeep after one submission.
4. **JOSS paper + arXiv preprint.** DOI + Google Scholar + Papers With
   Code; the compounding citation channel that is the durable growth
   loop for research software. Also cite Lim et al. 2021 throughout.

Tier 2, launch week:

5. best-of-ml-python + awesome-machine-learning / awesome-time-series
   PRs (passive, low effort).
6. Papers With Code (once benchmark results exist).
7. Show HN + r/MachineLearning + r/Python posts, hook framed as the
   precise gap sklearn/sktime/pytorch-forecasting leave open.

Tier 3, ongoing:

8. PyData / SciPy talk or poster.
9. Kaggle demo notebook on a known churn dataset.
10. Blog / dev.to / X amplification; sklearn Discord mention once
    listed in Related Projects.

## 5. Sequencing against the roadmap

- **Now (phase-1):** ship the interim README (section 2). Stand up the
  v1-minimal Sphinx + RTD skeleton so docstrings accrue into a real API
  ref as code lands (rule 2 already requires the docstrings). Reserve
  the PyPI name. Draft the JOSS paper outline alongside the design doc.
- **At v1 code-complete:** swap to the launch README; produce the real
  attention figure and the churn end-to-end example from passing code;
  fill the paper-backed benchmark sentence from a reproducible script.
- **At v1 ship:** execute Tier 1 promotion in order; submit JOSS;
  request sklearn Related Projects listing.
- **Post-v1:** sphinx-gallery, "Used by" section once real adopters
  exist, Tier 2/3 channels.

## 6. Open decisions for the review loop

- Stack choice (Sphinx over mkdocs+mkdocstrings) should be ratified in
  the architecture doc and run through `/design-review`, since
  requirements.md N6 left it open and the architecture phase owns it.
- Logo/wordmark: needs a design pass; placeholder text wordmark
  acceptable for the interim README.
- JOSS vs arXiv-only vs both: recommend both; confirm with maintainer
  given the writing cost.
