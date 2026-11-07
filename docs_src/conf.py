# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from kennard_stone import __author__, __version__

project = "kennard-stone"
copyright = "2021, yu9824"
author = __author__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # docstringからドキュメントを作成してくれる。
    "sphinx.ext.napoleon",  # google式・numpy式docstringを整形してくれる。
    "sphinx.ext.githubpages",  # github-pages用のファイルを生成してくれる。
    "myst_nb",  # Markdown と Jupyter Notebook を読み込む。
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Show type aliases by name instead of expanding them. Device would otherwise
# depend on whether torch is installed in the docs build.
autodoc_type_aliases = {
    "Device": "kennard_stone.utils._type_alias.Device",
    "MetricLike": "kennard_stone.utils._type_alias.MetricLike",
    "InitMethod": "kennard_stone.utils._type_alias.InitMethod",
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_theme_options = {
    "repository_url": "https://github.com/yu9824/kennard_stone",
    "use_repository_button": True,
    "show_toc_level": 2,
}

# MyST-NB: show the outputs saved in the notebooks without executing them.
nb_execution_mode = "off"

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]
