# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information



import os
import sys
import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../dfdistill'))
sys.path.insert(0, os.path.abspath('../../dfdistill'))


project = 'DFDistill'
copyright = '2025, Nasyrov E., Okhotnikov N., Sapronov Y., Solodkin V.'
author = 'Nasyrov E., Okhotnikov N., Sapronov Y., Solodkin V.'

master_doc = 'index'
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.ifconfig',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.mathjax'
]

autodoc_mock_imports = ["numpy", "scipy", "sklearn", "torch"]

html_context = {
	"display_github": True,
	"github_user": "Intelligent-Systems-Phystech",
	"github_repo": "DFDistill",
	"github_version": "main",
	"conf_py_path": "./docs/"
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
