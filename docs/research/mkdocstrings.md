# mkdocstrings + pydantic v2 documentation research

Scope: verify the seq-sklearn doc-toolchain choice (mkdocs + mkdocs-material + mkdocstrings, with griffe parsing pydantic v2 configs) in 2026, and pin concrete versions.

## Source citations (URLs)

1. mkdocstrings-python on PyPI: https://pypi.org/project/mkdocstrings-python/
2. mkdocstrings-python changelog: https://mkdocstrings.github.io/python/changelog/
3. mkdocs-material on PyPI: https://pypi.org/project/mkdocs-material/
4. Material for MkDocs blog, "What MkDocs 2.0 means": https://squidfunk.github.io/mkdocs-material/blog/2026/02/18/mkdocs-2.0/
5. Material for MkDocs changelog: https://squidfunk.github.io/mkdocs-material/changelog/
6. ProperDocs warning re MkDocs 2.0: https://github.com/orgs/ProperDocs/discussions/33
7. griffe-pydantic overview: https://mkdocstrings.github.io/griffe-pydantic/
8. griffe-pydantic repo: https://github.com/mkdocstrings/griffe-pydantic
9. griffe official pydantic extension page: https://mkdocstrings.github.io/griffe/extensions/official/pydantic/
10. Pydantic documentation integration page: https://pydantic.dev/docs/validation/latest/integrations/dev-tools/documentation/
11. Pydantic's own mkdocs.yml: https://github.com/pydantic/pydantic/blob/main/mkdocs.yml
12. FastAPI's mkdocs.yml: https://github.com/fastapi/fastapi/blob/master/docs/en/mkdocs.yml
13. autodoc_pydantic releases: https://github.com/mansenfranzen/autodoc_pydantic/releases
14. autodoc_pydantic on PyPI: https://pypi.org/project/autodoc_pydantic/
15. mkdocstrings recipes (gen-files pattern): https://mkdocstrings.github.io/recipes/
16. mkdocs-gen-files manual: https://oprypin.github.io/mkdocs-gen-files/index.html
17. MkDocs configuration (strict + validation): https://www.mkdocs.org/user-guide/configuration/
18. MkDocs strict-mode discussion: https://github.com/mkdocs/mkdocs/discussions/3818
19. Astral / ruff docs site (uses MkDocs): https://docs.astral.sh/ruff/
20. ruff mkdocs template: https://github.com/astral-sh/ruff/blob/main/mkdocs.template.yml

## Version pin recommendations (with concrete lower bounds)

As of May 2026 the current stable releases are:

- `mkdocstrings-python` 2.0.3 (released 2026-02-20), Python >= 3.10 [1].
- `mkdocstrings` 1.0.4 (Apr 2026, per FastAPI search context).
- `mkdocs-material` 9.7.6 (released 2026-03-19); 9.7.5 pinned `mkdocs<2` to guard against the breaking 2.0 release [3, 5].
- `griffe-pydantic` 1.3.1 (released 2026-02-20) [8].
- `griffe` 1.x line (current 1.14.x per changelog).
- `mkdocs` 1.6.x. Do NOT track `mkdocs>=2`: the 2.0 release rewrote the plugin and theme systems with no migration path, and the wider ecosystem (Material, ProperDocs fork) has rejected it [4, 6].

Recommended `pyproject.toml [project.optional-dependencies] docs`:

```toml
docs = [
  "mkdocs>=1.6,<2",
  "mkdocs-material>=9.7,<10",
  "mkdocstrings>=1.0,<2",
  "mkdocstrings-python>=2.0,<3",
  "griffe>=1.5,<2",
  "griffe-pydantic>=1.3,<2",
  "mkdocs-gen-files>=0.5",
  "mkdocs-literate-nav>=0.6",
  "mkdocs-section-index>=0.3",
]
```

The `mkdocs<2` cap is load-bearing. Material 9.7.5 already added the same cap upstream [5].

## Pydantic v2 rendering capability in mkdocstrings 2026

mkdocstrings-python parses source via `griffe`. Out of the box it renders pydantic models as plain Python classes: it sees class-level type-annotated attributes and their `Field(...)` defaults, and emits an attribute table with name, type, default. It does NOT special-case `model_validator`, `field_validator`, `model_config`, computed fields, or the field's `description=` kwarg.

To get pydantic-shaped pages you enable the `griffe-pydantic` Griffe extension [7, 9]. It extracts model metadata into Griffe's `extra` slot and binds custom mkdocstrings templates: dedicated "Fields", "Validators", "Config" sections, with constraints (`ge`, `le`, `min_length`, etc.), aliases, and `frozen` shown explicitly [7]. It operates both statically and dynamically.

Pydantic itself, the canonical reference site, runs mkdocs + mkdocstrings WITHOUT griffe-pydantic [11]. It relies on attribute docstrings and standard Field metadata. That works because pydantic's public API surface is hand-curated and the team writes Google-style docstrings for every model. For a config-heavy library like seq-sklearn (many nested `BaseModel` configs, validators that enforce cross-field constraints), griffe-pydantic is the better default.

## Validators, field descriptions, inheritance behavior

- **Field descriptions from `Field(..., description="...")`**: rendered correctly by griffe-pydantic. Description text appears in the field table. Plain mkdocstrings-python only surfaces this if you also put the description in an attribute docstring (the line directly under the attribute assignment in PEP-257 style) [7, 11].
- **Validators**: griffe-pydantic shows `@field_validator` and `@model_validator` as a separate Validators section, linked back to the field(s) they bind to [7]. Plain mkdocstrings-python renders them as ordinary methods on the class, which is noisy.
- **Inheritance**: mkdocstrings-python supports the `inherited_members` option (introduced in 1.2.0) which controls whether parent attributes/methods appear on the child page [2]. FastAPI sets `inherited_members: true` [12]. For `TFTConfig <- BaseModelConfig <- BaseTrainingConfig`, with `inherited_members: true` the child page renders the full flattened field table. griffe-pydantic respects the same option and merges inherited fields into the per-class Fields section [7].
- **Caveat**: griffe-pydantic's static mode can miss validators registered via `__pydantic_decorators__` or constructed dynamically. For seq-sklearn's static, declarative configs this is a non-issue.

## Comparison with sphinx + autodoc_pydantic 2026 status

The architecture doc calls sphinx-autodoc + pydantic "more brittle." 2026 evidence:

- `autodoc_pydantic` last release is v2.2.0, dated 2024-04-26 [13]. No PyPI release in the past ~12 months, classified as low-maintenance / possibly discontinued by Snyk [14]. It does support pydantic >= 2.7 [13], but the package has not tracked pydantic 2.8-2.11 changes (e.g. computed fields, deprecated-field API, new validator hooks).
- Sphinx itself is healthy, but the autodoc + pydantic integration depends on a single third-party extension that is stalling. Replacing it requires writing custom Sphinx directives.
- mkdocstrings-python + griffe-pydantic both shipped point releases in February 2026 [1, 8] and are maintained by the same author (Timothée Mazzucotelli) as pydantic's parser-of-choice.

Net: in 2026 mkdocs is the actively maintained path for pydantic v2 API docs. The architecture doc's "more brittle" assessment is correct and has gotten worse, not better.

## Worked examples from comparable libraries

- **pydantic** (docs.pydantic.dev): mkdocs + mkdocs-material + mkdocstrings, NO griffe-pydantic [11]. Relies on attribute docstrings.
- **FastAPI** (fastapi.tiangolo.com): mkdocs + mkdocs-material + mkdocstrings, with `griffe_typingdoc` extension, `inherited_members: true`, `merge_init_into_class: true`, `docstring_section_style: spacy` [12].
- **Typer** (typer.tiangolo.com): mkdocs-material (same Tiangolo doc stack as FastAPI). Search context confirms Typer is on Material for MkDocs.
- **ruff** (docs.astral.sh/ruff): mkdocs + mkdocs-material [19, 20]. Ruff is not a pydantic-heavy project, so it does not exercise the pydantic rendering path.

Three of the four use mkdocstrings; ruff is mkdocs but documents a Rust core, so no Python autodoc. None use Sphinx. None use autodoc_pydantic.

## Recommended mkdocstrings options

Aligned with FastAPI [12] and pydantic [11], tuned for a config-heavy library:

```yaml
plugins:
  - mkdocstrings:
      handlers:
        python:
          options:
            extensions:
              - griffe_pydantic:
                  schema: true
            show_root_heading: true
            show_root_full_path: false
            show_signature_annotations: true
            separate_signature: true
            signature_crossrefs: true
            merge_init_into_class: true
            inherited_members: true
            members_order: source
            docstring_section_style: spacy
            show_symbol_type_heading: true
            show_symbol_type_toc: true
            show_source: false
            filters: ["!^_"]
            unwrap_annotated: true
```

Rationale per option:
- `show_root_full_path: false`: header reads `TFTConfig`, not `seq_sklearn.config.training.TFTConfig`. Less visual noise.
- `merge_init_into_class: true`: pydantic models do not define `__init__` by hand, but if a class does, fold params into the class page.
- `show_signature_annotations: true` + `separate_signature: true`: types render under the signature, not crowded into a one-liner.
- `show_source: false`: source view bloats every page; users who want source click through to GitHub.
- `inherited_members: true`: needed so `TFTConfig` shows fields from `BaseModelConfig` and `BaseTrainingConfig`.
- `filters: ["!^_"]`: hide private helpers.

## Build-time evidence

The ~3s mkdocs vs ~30s sphinx claim is order-of-magnitude correct but no public benchmark cites those exact numbers. Per writeups comparing the two: mkdocs' `serve` does incremental rebuilds with sub-second live reload; Sphinx requires full `make html` per change, with builds in the 10-60s range on medium projects. For seq-sklearn's expected size mkdocs cold builds typically land at 2-5s, Sphinx 20-40s. Treat the architecture numbers as illustrative; re-measure once content lands.

## mkdocs-gen-files for executable examples

`mkdocs-gen-files` runs Python scripts at build time and writes files into the docs tree without committing them [15, 16]. The canonical recipe pairs it with `mkdocs-literate-nav` and `mkdocs-section-index` for a generated API reference, documented in mkdocstrings' recipes [15]:

```yaml
plugins:
  - gen-files:
      scripts:
        - scripts/gen_api_pages.py
        - scripts/gen_examples.py
  - literate-nav: { nav_file: SUMMARY.md }
  - section-index
  - mkdocstrings
```

For seq-sklearn's executable examples, `gen_examples.py` imports each example module, runs it under a context that captures stdout and saved figures, and emits one markdown page per example. griffe-pydantic uses gen-files for its own docs [8].

## --strict gate behavior

`mkdocs build --strict` converts WARNING log entries into errors, exiting non-zero [17, 18]. The `validation:` block governs what raises a warning. Maximal-strictness profile gates on:

- `validation.nav.omitted_files`: file present in `docs/` but absent from `nav:`.
- `validation.nav.not_found`: nav references a missing file.
- `validation.nav.absolute_links`: nav uses absolute paths.
- `validation.links.not_found`: broken internal link.
- `validation.links.anchors`: link points at a missing anchor.
- `validation.links.absolute_links`, `validation.links.unrecognized_links`.

Plus plugin warnings: mkdocstrings emits warnings for unresolved cross-references, missing `preload_modules`, and griffe parse failures. Under `--strict` all fail the build, which is the intended CI behavior.

## Decisions implied for seq-sklearn

1. Keep the chosen stack: mkdocs 1.6.x + mkdocs-material 9.7.x + mkdocstrings 1.0.x + mkdocstrings-python 2.0.x. The choice has aged well into 2026.
2. Pin `mkdocs<2`. The 2.0 release is a hostile rewrite [4, 6].
3. Add `griffe-pydantic>=1.3` as a docs dep. Plain mkdocstrings-python does not surface validators or `Field(description=...)` cleanly enough for a config-heavy public API.
4. Set `inherited_members: true` to render the full `TFTConfig` field table including ancestors.
5. Reject sphinx + autodoc_pydantic: autodoc_pydantic is effectively dormant (last release April 2024) and the stack is no longer the safer bet [13, 14].
6. Use `mkdocs-gen-files` + `mkdocs-literate-nav` + `mkdocs-section-index` for both the API reference tree and the auto-rendered executable-example pages [15].
7. Keep `mkdocs build --strict` in CI with the maximal `validation:` profile [17].
8. Re-measure build time once content lands; do not over-rely on the ~3s vs ~30s claim.
