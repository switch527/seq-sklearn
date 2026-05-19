"""Sphinx configuration for the seq-sklearn documentation site (Phase 12 A12).

Stack: Sphinx + numpydoc + autosummary/autodoc + autodoc-pydantic +
sphinx-gallery + PyData Sphinx Theme + sphinx-copybutton + MyST +
sphinx-sitemap, hosted on Read the Docs. User-ratified at Phase 12
design-review R1 over the earlier interim mkdocs resolution.

Site IA is Diátaxis: ``tutorial/ how-to/ reference/ explanation/
about/ design/`` toctrees. ``sphinx-build -W --keep-going`` is the CI
gate; ``sphinx-build -b doctest`` executes ``.. testcode::`` /
``.. doctest::`` blocks; ``sphinx-gallery`` executes every
``docs/examples/*.py`` at build.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

project = "seq-sklearn"
author = "Steve"
copyright = "2026, seq-sklearn contributors"

try:
    release = _pkg_version("seq-sklearn")
except PackageNotFoundError:
    release = "0.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx_gallery.gen_gallery",
    "numpydoc",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_sitemap",
]

# Markdown source via MyST, alongside reST.
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"

# Design docs are hosted under `design/` (NG1: "internal governance,
# not user docs" callout) once the Diátaxis tree lands. The exclusions
# below are the build-output paths and the legacy single-source design
# docs that still live at `docs/` root (they will be moved into
# `design/` and removed from this list in a follow-up commit).
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Research briefs and external-paper PDFs are repo artifacts, not
    # published doc-site content. Everything else under docs/ is in a
    # Diátaxis toctree (design/ surfaces the internal-governance docs
    # with a callout).
    "research/**",
    "references/**",
    # sphinx-gallery consumes `examples/README.rst` as the gallery
    # source and renders the result at `examples_gallery/index`; the
    # source itself is not a published page (and would otherwise trip
    # the `toc.not_included` warning).
    "examples/README.rst",
]

# Generate the API reference stub pages from autosummary directives.
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
# `nitpicky = True` would surface every unresolved xref, which
# autodoc-pydantic emits prolifically for private validators
# (`_check_extra_not_reserved`, `PydanticUndefined`, etc.). Those
# are INTERNAL per A3 and not worth chasing; rely on `-W` to catch
# real defects (broken toctrees, missing files, dead internal links)
# without nitpicky's xref-strictness.
nitpicky = False

# numpydoc + autodoc-pydantic field/validator summaries for pydantic
# config classes (the dominant reference surface).
numpydoc_show_class_members = False
autodoc_pydantic_model_show_field_summary = True
autodoc_pydantic_model_show_validator_summary = True
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_member_order = "bysource"

# Only the genuinely-optional extras are mocked. Core runtime deps
# (torch, lightning, optuna, ...) are installed in any docs build
# environment.
autodoc_mock_imports = ["onnx", "onnxruntime"]

# sphinx-gallery: execute every example .py at build, capture
# stdout/plots, generate the thumbnail gallery. CPU + max_epochs=1
# per the plan; PG.4 static scan enforces.
sphinx_gallery_conf = {
    "examples_dirs": [str(Path(__file__).parent / "examples")],
    "gallery_dirs": ["examples_gallery"],
    "filename_pattern": r"/.*\.py$",
    "ignore_pattern": r"__init__\.py$",
    "remove_config_comments": True,
    "show_memory": False,
    "plot_gallery": True,
    "abort_on_example_error": True,
    "expected_failing_examples": [],
    "min_reported_time": 0,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

# Nitpicky-allowlist for cross-refs we deliberately do not link
# (third-party types not in intersphinx, transient internals).
nitpick_ignore_regex = [
    (r"py:.*", r"^pytorch_lightning\..*$"),
    (r"py:.*", r"^lightning\..*$"),
    (r"py:.*", r"^optuna\..*$"),
    (r"py:.*", r"^seq_sklearn\._.*$"),
]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "seq-sklearn"
html_baseurl = "https://seq-sklearn.readthedocs.io/en/latest/"
sitemap_url_scheme = "{link}"

html_theme_options = {
    "github_url": "https://github.com/switch527/seq-sklearn",
    "use_edit_page_button": False,
    "show_toc_level": 2,
}

myst_enable_extensions = ["colon_fence", "deflist"]
