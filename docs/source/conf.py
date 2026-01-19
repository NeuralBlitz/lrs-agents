# Configuration file for the Sphinx documentation builder.

import os
import sys
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath('../../'))

# Mock optional dependencies
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
    'matplotlib',
    'matplotlib.pyplot',
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()

# Project information
project = 'LRS-Agents'
copyright = '2024-2025, LRS-Agents Contributors'
author = 'LRS-Agents Contributors'
release = '0.2.1'
version = '0.2.1'

# Extensions
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

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = []

html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}

# Suppress warnings
nitpicky = False
suppress_warnings = ['ref.python', 'ref.doc', 'ref.ref']

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# MyST
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

# Autosummary
autosummary_generate = True
autosummary_imported_members = False
