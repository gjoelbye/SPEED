"""Sphinx configuration for SPEED documentation."""

project = "SPEED"
copyright = "2024, Anders Gjolbye, Lina Skerath, William Lehn-Schioler, Nicolas Langer, Lars Kai Hansen"
author = "Anders Gjolbye et al."
release = "2.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

# Napoleon settings (Google-style docstrings)
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "mne": ("https://mne.tools/stable/", None),
}
