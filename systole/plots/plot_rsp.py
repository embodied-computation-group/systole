# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from bokeh.plotting.figure import Figure
from matplotlib.axes import Axes

from systole.plots.utils import get_plotting_function


def plot_rsp(
    signal: Union[pd.DataFrame, np.ndarray, List],
    sfreq: int = 1000,
    slider: bool = True,
    ax: Optional[Axes] = None,
    figsize: Optional[Union[int, List[int], Tuple[int, int]]] = None,
    backend: str = "matplotlib",
) -> Union[Axes, Figure]:
    """Visualization of Respiration signal.

    Parameters
    ----------
    signal : :py:class:`pandas.DataFrame` | :py:class:`numpy.ndarray` | list
        Dataframe of Respiration signal in the long format. If a data frame is
        provided, it should contain at least one ``'time'`` and one column for
        signal(`"respiration"`). If an array is provided, it will
        automatically create a DataFrame using the array as signal and
        ``sfreq`` as sampling frequency.
    sfreq : int
        Signal sampling frequency. Default is set to 1000 Hz.
    slider : bool
        If `True`, will add a slider to select the time window to plot
        (requires bokeh backend).
    ax : :class:`matplotlib.axes.Axes` | None
        Where to draw the plot. Default is *None* (create a new figure). Only
        applies when `backend="matplotlib"`.
    figsize : tuple, int or None
        Figure size. Default is `(13, 5)` for matplotlib backend, and the
        height is `300` when using bokeh backend.
    backend: str
        Select plotting backend {"matplotlib", "bokeh"}. Defaults to
        "matplotlib".

    Returns
    -------
    plot : :class:`matplotlib.axes.Axes` | :class:`bokeh.plotting.figure.Figure`
        The matplotlib axes, or the bokeh figure containing the plot.

    See also
    --------
    plot_events, plot_rr

    Examples
    --------

    Plotting raw Respiration recording.

    .. jupyter-execute::

       from systole import import_dataset1
       from systole.plots import plot_rsp

       # Import Respiration recording as pandas data frame
       rsp = import_dataset1(modalities=['Respiration'])

       # Only use the first 90 seconds for demonstration
       rsp = rsp[rsp.time.between(0, 90)]
       plot_rsp(rsp, sfreq=1000)

    Using Bokeh backend, with instantaneous respiration rate and artefacts.

    .. jupyter-execute::

       from bokeh.io import output_notebook
       from bokeh.plotting import show
       output_notebook()

       show(
           plot_rsp(rsp, backend="bokeh")
       )
    """

    if figsize is None:
        if backend == "matplotlib":
            figsize = (13, 5)
        elif backend == "bokeh":
            figsize = 300

    if isinstance(signal, pd.DataFrame):
        signal = signal.respiration
    else:
        signal = signal

    time = pd.to_datetime(np.arange(0, len(signal)), unit="ms", origin="unix")

    plot_rsp_args = {
        "time": time,
        "signal": signal,
        "ax": ax,
        "figsize": figsize,
        "slider": slider,
    }

    plotting_function = get_plotting_function("plot_rsp", "plot_rsp", backend)
    plot = plotting_function(**plot_rsp_args)

    return plot
