# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Curryer"
copyright = "2025 University of Colorado"
author = "Brandon Stone"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # Allows sphinx to make docs from markdown pages (rst still works too)
    "sphinx.ext.napoleon",  # Parses Numpy style docstrings
    "autoapi.extension",  # Automatic API generation
    "sphinx.ext.mathjax",  # Enable rendering of equations in docs pages
]

myst_enable_extensions = ["html_image", "dollarmath"]

autoapi_type = "python"
autoapi_dirs = ["../../curryer"]

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/curryer_logo.png"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#C86417",  # Orange
        "color-brand-content": "#C86417",
        "color-background-primary": "#e6e8e9",  # Light gray background
        "color-foreground-primary": "#0a0908",  # Almost black text
        "color-foreground-secondary": "#909ead",  # Gray for secondary text
    },
    "dark_css_variables": {
        "color-brand-primary": "#C86417",  # Orange (same for dark mode)
        "color-brand-content": "#C86417",
        "color-background-primary": "#0a0908",  # Dark background
        "color-foreground-primary": "#e6e8e9",  # Light text
        "color-foreground-secondary": "#909ead",  # Gray for secondary text
    },
}
