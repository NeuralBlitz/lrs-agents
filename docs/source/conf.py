# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from unittest.mock import MagicMock

# -- Path setup --------------------------------------------------------------

# Add project root to path so we can import lrs
sys.path.insert(0, os.path.abspath('../../'))

# -- Mock optional dependencies ----------------------------------------------

class Mock(MagicMock):
    """Mock class for optional dependencies."""
    
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


# Mock modules that are optional or cause import issues during doc build
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

# -- Project information -----------------------------------------------------

project = 'LRS-Agents'
copyright = '2024-2025, LRS-Agents Contributors'
author = 'LRS-Agents Contributors'
release = '0.2.0'
version = '0.2.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings
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

# Add any paths that contain templates here, relative to this directory
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    'old',
    '**/old/**',
    '*.ipynb_checkpoints',
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets)
html_static_path = []

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
}

# The name for this set of Sphinx documents
html_title = f"{project} v{version}"

# A shorter title for the navigation bar
html_short_title = "LRS-Agents"

# -- Extension configuration -------------------------------------------------

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Autodoc settings --------------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'imported-members': False,
}

# This value selects what content will be inserted into the main body of an autoclass directive
autoclass_content = 'both'

# This value controls how to represent typehints
autodoc_typehints = 'description'

# Don't show type hints in signatures (they're in the description)
autodoc_typehints_description_target = 'documented'

# -- Autosummary settings ----------------------------------------------------
autosummary_generate = True
autosummary_imported_members = False

# -- Intersphinx settings ----------------------------------------------------
# Intersphinx mapping for linking to other project documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'langchain': ('https://api.python.langchain.com/en/latest/', None),
}

# -- MyST settings -----------------------------------------------------------
# MyST-Parser configuration for Markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

# -- Todo extension settings -------------------------------------------------
# If true, `todo` and `todoList` produce output, else they produce nothing
todo_include_todos = True

# -- Options for manual page output ------------------------------------------
# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'lrs-agents', 'LRS-Agents Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    ('index', 'LRS-Agents', 'LRS-Agents Documentation',
     author, 'LRS-Agents', 'Resilient AI agents via Active Inference.',
     'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------
# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
epub_identifier = 'https://lrs-agents.readthedocs.io'

# A unique identification for the text.
epub_uid = 'lrs-agents'

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

# -- Suppress warnings -------------------------------------------------------
# Don't be too strict about warnings
nitpicky = False

# Suppress specific warnings
suppress_warnings = [
    'ref.python',
    'ref.doc',
    'ref.ref',
    'autosummary',
]

# -- MathJax configuration ---------------------------------------------------
# Configure MathJax for rendering mathematical equations
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    },
}

# -- Additional configuration ------------------------------------------------

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx.
language = 'en'

# If true, figures, tables and code-blocks are automatically numbered if they have a caption
numfig = True

# Format for figure numbers
numfig_format = {
    'figure': 'Figure %s',
    'table': 'Table %s',
    'code-block': 'Listing %s',
    'section': 'Section %s',
}

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = ['lrs.']

# If true, keep warnings as "system message" paragraphs in the built documents.
keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
