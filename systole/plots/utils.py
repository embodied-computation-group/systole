# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import importlib

from packaging.version import parse


def get_plotting_function(plot_name, plot_module, backend="matplotlib"):
    """Return plotting function for correct backend.

    Inspired by Arviz'backend management.
    """

    _backend = {
        "mpl": "matplotlib",
        "bokeh": "bokeh",
        "matplotlib": "matplotlib",
    }

    backend = backend.lower()

    try:
        backend = _backend[backend]
    except KeyError as err:
        raise KeyError(
            "Backend {} is not implemented. Try backend in {}".format(
                backend, set(_backend.values())
            )
        ) from err

    if backend == "bokeh":
        try:
            import bokeh

            assert parse(bokeh.__version__) >= parse("1.4.0")

        except (ImportError, AssertionError) as err:
            raise ImportError(
                "'bokeh' backend needs Bokeh (1.4.0+) installed."
                " Please upgrade or install"
            ) from err

    # Perform import of plotting method
    module = importlib.import_module(
        "systole.plots.backends.{backend}.{plot_module}".format(
            backend=backend, plot_module=plot_module
        )
    )

    plotting_method = getattr(module, plot_name)

    return plotting_method
