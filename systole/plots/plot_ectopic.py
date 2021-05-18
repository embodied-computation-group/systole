# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Optional, Tuple, Union, overload

import numpy as np
from matplotlib.axes import Axes

from bokeh.plotting.figure import Figure
from systole.correction import rr_artefacts
from systole.plots.utils import get_plotting_function
from systole.utils import to_rr


@overload
def plot_ectopic(
    rr: None,
    artefacts: Dict[str, np.ndarray],
    input_type: str = "rr_ms",
) -> Union[Figure, Axes]:
    ...


@overload
def plot_ectopic(
    rr: Union[List[float], np.ndarray],
    artefacts: None,
    input_type: str = "rr_ms",
) -> Union[Figure, Axes]:
    ...


@overload
def plot_ectopic(
    rr: Union[List[float], np.ndarray],
    artefacts: Dict[str, np.ndarray],
    input_type: str = "rr_ms",
) -> Union[Figure, Axes]:
    ...


def plot_ectopic(
    rr=None,
    artefacts=None,
    input_type: str = "rr_ms",
    ax: Optional[Axes] = None,
    backend: str = "matplotlib",
    figsize: Union[Tuple[float, float], int] = None,
) -> Union[Figure, Axes]:
    """Plot interactive ectobeats subspace.

    Parameters
    ----------
    rr : 1d array-like or None
        Interval time-series (R-R, beat-to-beat...), in miliseconds.
    artefacts : dict or None
        The artefacts detected using
        :py:func:`systole.detection.rr_artefacts()`.
    input_type : str
        The type of input vector. Default is `"rr_ms"` for vectors of
        RR intervals, or interbeat intervals (IBI), expressed in milliseconds.
        Can also be `"peaks"` (a boolean vector where `1` represents the
        occurrence of R waves or systolic peaks) or `"rr_s"` for IBI expressed
        in seconds.
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

    Notes
    -----
    If both *rr* or *artefacts* are provided, will recompute *artefacts*
    given the current rr time-series.

    Examples
    --------

    Visualizing ectopic subspace from RR time series.

    .. jupyter-execute::

       from systole import import_rr
       from systole.plots import plot_ectopic
       # Import PPG recording as numpy array
       rr = import_rr().rr.to_numpy()
       plot_ectopic(rr)

    Visualizing ectopic subspace from the `artefact` dictionary.

    .. jupyter-execute::

       from systole import import_rr
       from systole.plots import plot_ectopic
       from systole.detection import rr_artefacts
       # Import PPG recording as numpy array
       rr = import_rr().rr.to_numpy()
       # Use the rr_artefacts function to find ectopic beats
       artefacts = rr_artefacts(rr)
       plot_ectopic(artefacts=artefacts)
    """

    if figsize is None:
        if backend == "matplotlib":
            figsize = (13, 5)
        elif backend == "bokeh":
            figsize = 600

    if artefacts is None:
        if rr is None:
            raise ValueError("rr or artefacts should be provided")
        else:
            if input_type == "peaks":
                rr = to_rr(rr)
            elif input_type == "rr_s":
                rr *= 1000
            artefacts = rr_artefacts(rr)

    plot_ectopic_args = {
        "artefacts": artefacts,
        "ax": ax,
        "figsize": figsize,
    }

    plotting_function = get_plotting_function("plot_ectopic", "plot_ectopic", backend)
    plot = plotting_function(**plot_ectopic_args)

    return plot
