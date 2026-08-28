# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import time

import sphinx_bootstrap_theme
import systole

# -- Project information -----------------------------------------------------

project = "systole"
copyright = u"2020-{}, Micah Allen".format(time.strftime("%Y"))
author = "Micah Allen"
release = systole.__version__


image_scrapers = ("matplotlib",)

sphinx_gallery_conf = {
    "examples_dirs": "./examples/",
    "backreferences_dir": "api",
    "image_scrapers": image_scrapers,
}

bibtex_bibfiles = ['refs.bib']
bibtex_reference_style = "author_year"
bibtex_default_style = "unsrt"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "jupyter_sphinx",
    "sphinx_design",
    "myst_nb",
    "sphinx_gallery.load_style",
    "sphinxcontrib.bibtex",
    "sphinx_sitemap",
    "sphinxext.opengraph",
]

panels_add_bootstrap_css = False

# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = False

# raise an error if the documentation does not build and exit the process
# this should especially ensure that the notebooks run correctly
nb_execution_raise_on_error = True

# myst-nb defaults to a 30 second budget per notebook, which several of the
# tutorials exceed: they download example datasets and then run peak detection
# over twenty minutes of signal. The kernel was being killed mid-execution and
# surfaced as an opaque ZMQError.
nb_execution_timeout = 300

# Cache executed notebooks so that an unchanged notebook is not re-run on every
# build. The directory is restored between CI runs, see .github/workflows/docs.yml.
nb_execution_mode = "cache"
nb_execution_cache_path = os.environ.get(
    "SYSTOLE_NB_CACHE", os.path.join(os.path.dirname(__file__), "..", ".jupyter_cache")
)

# Include the example source for plots in API docs
plot_include_source = True
plot_formats = [("png", 90)]
plot_html_show_formats = False
plot_html_show_source_link = False

source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.

html_theme = "pydata_sphinx_theme"
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/embodied-computation-group/systole",
            icon="fa-brands fa-square-github",
        ),
        dict(
            name="Twitter",
            url="https://twitter.com/visceral_mind",
            icon="fa-brands fa-square-twitter",
        ),
        dict(
            name="Pypi",
            url="https://pypi.org/project/systole-core/",
            icon="fa-solid fa-box",
        ),
    ],
    "logo": {
        "text": "Systole",
    },
    # Puts an "Edit this page" link on every page, pointing at the file that
    # produced it. The icon links above reach the repository root only.
    "use_edit_page_button": True,
}

# Consumed by use_edit_page_button to build the per-page source URL.
html_context = {
    "github_user": "embodied-computation-group",
    "github_repo": "systole",
    "github_version": "master",
    "doc_path": "docs/source",
}

html_sidebars = {"**": []}

# -- Options for HTML output -------------------------------------------------

html_logo = "images/logo_small.svg"
html_favicon = "images/logo_small.svg"


# -- Canonical URLs, sitemap and page metadata -------------------------------

# Without html_title Sphinx falls back to "systole <version> documentation",
# which is what a search engine then records as the name of this site. It goes
# stale the moment a release ships, and the version already appears in the
# landing page title through _templates/layout.html, which reads it from the
# package.
html_title = "Systole"

# The documentation is published under the group domain. Naming the canonical
# base here makes Sphinx write a <link rel="canonical"> into every page, so a
# page reached by more than one address is indexed once rather than as
# competing copies.
html_baseurl = "https://www.the-ecg.org/systole/"

# sphinx-sitemap defaults to a "{lang}{version}{link}" layout meant for builds
# that publish several versions side by side. This one publishes a single
# version at the root of the path above, so those segments would fill the
# sitemap with URLs that do not exist.
sitemap_url_scheme = "{link}"

# The search page is an empty shell filled in by JavaScript and the index is a
# list of links, so neither is worth pointing a crawler at.
sitemap_excludes = ["search.html", "genindex.html"]

ogp_site_url = html_baseurl
ogp_site_name = "Systole"
ogp_type = "website"

# Pages that do not set their own description fall back to their opening prose,
# cut to this length.
ogp_description_length = 200

# sphinxext-opengraph draws a preview card per page with Matplotlib. It cannot
# read an SVG, and html_logo is one, so point it at the raster logo instead of
# letting it warn once per page and drop the image.
ogp_social_cards = {
    "enable": True,
    "image": "images/logo.png",
    "site_url": "the-ecg.org/systole",
}

# -- Intersphinx ------------------------------------------------

intersphinx_mapping = {
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("http://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
    "bokeh": ("http://docs.bokeh.org/en/latest/", None),
}
