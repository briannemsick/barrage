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
version = "0.5.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinxcontrib.apidoc",
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

autodoc_mock_imports = ["tensorflow"]
apidoc_module_dir = "../barrage"
apidoc_output_dir = "api"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_logo = "resources/barrage_logo_small.png"
