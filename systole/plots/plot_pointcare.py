# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Optional, Tuple, Union

import numpy as np
from bokeh.plotting.figure import Figure
from matplotlib.axes import Axes

from systole.plots.utils import get_plotting_function


def plot_pointcare(
    rr: Union[np.ndarray, list],
    input_type: str = "peaks",
    figsize: Optional[Union[List[int], Tuple[int, int], int]] = None,
    backend: str = "matplotlib",
    ax: Optional["Axes"] = None,
    **kwargs
) -> Union[Figure, Axes]:
    """Pointcare plot.

    Parameters
    ----------
    rr : np.ndarray or list
        Boolean vector of peaks detection or RR intervals.
    input_type : str
        The type of input vector. Default is `"peaks"` (a boolean vector where
        `1` represents the occurrence of R waves or systolic peaks).
        Can also be `"rr_s"` or `"rr_ms"` for vectors of RR intervals, or
        interbeat intervals (IBI), expressed in seconds or milliseconds
        (respectively).
    figsize : list, tuple, int or None
        Figure size. Default is `(13, 5)`.
    backend: str
        Select plotting backend {"matplotlib", "bokeh"}. Defaults to
        "matplotlib".
    ax : :class:`matplotlib.axes.Axes` or None
        Where to draw the plot. Default is `None` (create a new figure).

    Returns
    -------
    plot : :class:`matplotlib.axes.Axes` or :class:`bokeh.plotting.figure.Figure`
        The matplotlib axes, or the boken figure containing the plot.

    See also
    --------
    plot_events, plot_ectopic, plot_shortlong, plot_subspaces, plot_frequency,
    plot_timedomain, plot_nonlinear

    Examples
    --------

    Visualizing pointcare plot from RR time series.

    Using Matplotlib backend

    .. jupyter-execute::

       from systole import import_rr
       from systole.plots import plot_pointcare
       # Import PPG recording as numpy array
       rr = import_rr().rr.to_numpy()
       plot_pointcare(rr, input_type="rr_ms")

    Using Bokeh backend

    .. jupyter-execute::

       from systole import import_rr
       from systole.plots import plot_pointcare
       # Import PPG recording as numpy array
       rr = import_rr().rr.to_numpy()
       plot_pointcare(rr, input_type="rr_ms", backend="bokeh")

    """
    if input_type == "rr_ms":
        rr = np.asarray(rr)
    elif input_type == "rr_s":
        rr = np.asarray(rr) * 1000
    elif input_type == "peaks":
        rr = np.diff(np.where(rr)[0])

    plot_pointcare_args = {
        "rr": rr,
        "figsize": figsize,
        "ax": ax,
    }

    plotting_function = get_plotting_function(
        "plot_pointcare", "plot_pointcare", backend
    )
    plot = plotting_function(**plot_pointcare_args)

    return plot
