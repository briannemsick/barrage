# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
# -- Project information -----------------------------------------------------
project = "barrage"
copyright = "2019, Brian Nemsick"
author = "Brian Nemsick"
version = "0.3.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx_click.ext",
]
source_suffix = ".rst"
master_doc = "index"
language = None
pygments_style = None


# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_logo = "resources/barrage_logo_small.png"

autodoc_mock_imports = ["tensorflow"]
