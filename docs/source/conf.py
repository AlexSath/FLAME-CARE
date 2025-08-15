from docutils import nodes
from sphinx.application import Sphinx

project = "FLAME-CARE docs"
authors = "Alexandre R. Sathler"
release = "1.0"
copyright = "Lonlinear Optical Microscopy Lab @ UC Irvine"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Sphinx's own extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    # Our custom extension, only meant for Furo's own documentation.
    "furo.sphinxext",
    # External stuff
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_inline_tabs",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "env", ".tox", "README.md"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "myst": ("https://myst-parser.readthedocs.io/en/latest", None),
}


linkcheck_allowed_redirects = {
    # All HTTP redirections from the source URI to the canonical URI will be treated as "working".
    r"https://sphinx-doc\.org/.*": r"https://sphinx-doc\.org/en/master/.*"
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "CARE for FLAME"
html_theme_options = {
    "source_repository": "https://github.com/AlexSath/FLAME-CARE",
    "source_branch": "main",
    # "source_directory": "lib/esbonio/tests/workspaces/demo/",
}


def lsp_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Link to sections within the lsp specification."""

    anchor = text.replace("/", "_")
    ref = f"https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#{anchor}"

    node = nodes.reference(rawtext, text, refuri=ref, **options)
    return [node], []


def setup(app: Sphinx):
    app.add_role("lsp", lsp_role)
