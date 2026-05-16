"""Sphinx configuration for the seq-sklearn documentation.

Stack rationale and the full plan live in
``docs/readme_and_docs_plan.md``. Summary: the sklearn-compatible,
time-series-ML cluster is near-unanimously Sphinx + numpydoc +
autosummary on Read the Docs, with the sklearn User Guide / API
Reference / Examples triad. This skeleton is the minimal-viable v1
shape; the gallery and topic-organized guide come post-v1.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

project = "seq-sklearn"
author = "Steve"
copyright = "2026, seq-sklearn contributors"  # noqa: A001

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
    "numpydoc",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_sitemap",
]

# Markdown source via MyST, consistent with the repo's other docs.
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
# The docs/ folder also holds design docs and research notes that are
# not part of the published site. Exclude them so the build is just the
# curated skeleton; they remain readable on GitHub.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "requirements.md",
    "architecture.md",
    "implementation_plan.md",
    "hyperparameter_strategy.md",
    "phase_1_refactor_plan.md",
    "readme_and_docs_plan.md",
    "research/**",
    "references/**",
]

# Generate the API reference stub pages from autosummary directives.
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# numpydoc is the sklearn-ecosystem docstring standard. Validation stays
# off until the public API stabilizes so pre-implementation builds stay
# green; turn it on (numpydoc_validation_checks) at the v1 docstring pass.
numpydoc_show_class_members = False

# Heavy or not-yet-importable deps are mocked so Read the Docs builds
# stay fast and do not break before the package is code-complete.
autodoc_mock_imports = [
    "torch",
    "lightning",
    "pytorch_lightning",
    "optuna",
    "optuna_integration",
    "safetensors",
    "onnx",
    "onnxruntime",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

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
