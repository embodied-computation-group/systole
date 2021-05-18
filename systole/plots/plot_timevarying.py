# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Optional, Tuple, Union

import numpy as np
from matplotlib.axes import Axes

from bokeh.plotting.figure import Figure
from systole.plots.utils import get_plotting_function


def plot_timevarying(
    rr=Union[List, np.ndarray],
    input_type: str = "rr_s",
    ax: Optional[Axes] = None,
    figsize: Union[Tuple[float, float], int] = None,
    backend: str = "matplotlib",
) -> Union[Figure, Axes]:
    """Plot hrv subspace as described by Lipponen & Tarvainen (2019) [#]_.

    Parameters
    ----------
    rr : :py:class:`numpy.ndarray` or None
        Interval time-series (R-R, beat-to-beat...), in seconds or in
        miliseconds.
    input_type : str
        The type of input vector. Default is `"peaks"` (a boolean vector where
        `1` represents the occurrence of R waves or systolic peaks).
        Can also be `"rr_s"` or `"rr_ms"` for vectors of RR intervals, or
        interbeat intervals (IBI), expressed in seconds or milliseconds
        (respectively).
    ax : :class:`matplotlib.axes.Axes` or None
        Where to draw the plot. Default is *None* (create a new figure). Only
        applies when `backend="matplotlib"`.
    backend: str
        Select plotting backend {"matplotlib", "bokeh"}. Defaults to
        "matplotlib".
    figsize : tuple, int or None
        Figure size. Default is `(13, 5)` for matplotlib backend, and the
        height is `600` when using bokeh backend.

    Returns
    -------
    plot : :class:`matplotlib.axes.Axes` or :class:`bokeh.plotting.figure.Figure`
        The matplotlib axes, or the boken figure containing the plot.

    See also
    --------
    plot_events, plot_ectopic, plot_shortLong, plot_subspaces, plot_frequency,
    plot_timedomain, plot_nonlinear

    References
    ----------


    Examples
    --------


    """
    rr = np.asarray(rr)

    if figsize is None:
        if backend == "matplotlib":
            figsize = (13, 5)
        elif backend == "bokeh":
            figsize = 600

    plot_timevarying_args = {
        "rr": rr,
        "ax": ax,
        "figsize": figsize,
    }

    plotting_function = get_plotting_function(
        "plot_timevarying", "plot_timevarying", backend
    )
    plot = plotting_function(**plot_timevarying_args)

    return plot
