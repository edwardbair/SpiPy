# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SpiPy'
copyright = '2024, Niklas Griessbaum'
author = 'Niklas Griessbaum'

# Get version from setuptools-scm
try:
    from importlib.metadata import version
    release = version('spires')
except Exception:
    release = 'unknown'

version = release  # Short version (e.g., '0.2.1')
# Full version including alpha/beta/rc tags

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))


extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx_automodapi.automodapi',
              'sphinx.ext.napoleon', # for math
              'sphinx.ext.mathjax', # for math
              'myst_parser',    # markdown parsing
              'nbsphinx',       # Notebook integration
              'sphinx_markdown_tables'
              ]


templates_path = ['_templates']
exclude_patterns = []

autodoc_default_flags = ['members', 'undoc-members', 'private-members', 'special-members', 'show-inheritance']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'

#html_static_path = ['_static']

autosummary_generate = True# ['spires.interpolate']
#autosummary_generate = ['autosummary/*.rst']

#numpydoc_show_class_members = False
#add_module_names = False

# Suppress warnings for missing cross-references to files outside doc tree
suppress_warnings = ['myst.xref_missing']



