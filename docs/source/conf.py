# Configuration file for the Sphinx documentation builder.

import os
import sys
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath('../../'))

# Mock optional dependencies for documentation build
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = [
    'streamlit',
    'plotly',
    'plotly.graph_objects',
    'plotly.express',
    'seaborn',
    'pandas',
    'matplotlib',           # ADD
    'matplotlib.pyplot',    # ADD
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()

# -- Project information -----------------------------------------------------
project = 'LRS-Agents'
copyright = '2024, LRS-Agents Contributors'
author = 'LRS-Agents Contributors'
release = '0.2.0'
version = '0.2.0'

# -- General configuration ---------------------------------------------------

# Core extensions that must be present
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'myst_parser',
]

# Optional extensions - add if available
try:
    import nbsphinx
    extensions.append('nbsphinx')
except ImportError:
    pass  # nbsphinx is optional

templates_path = ['_templates']

exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    'old',
    '**/old/**',
    '*.ipynb_checkpoints',
]

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = []

html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'imported-members': False,
}

# Don't fail on warnings
nitpicky = False
suppress_warnings = ['ref.python', 'ref.doc', 'ref.ref']

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

# nbsphinx settings (only if nbsphinx is available)
if 'nbsphinx' in extensions:
    nbsphinx_execute = 'never'
    nbsphinx_allow_errors = True

# Autosummary
autosummary_generate = True
autosummary_imported_members = False
