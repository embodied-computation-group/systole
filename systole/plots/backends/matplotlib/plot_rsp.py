# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from pandas.core.indexes.datetimes import DatetimeIndex


def plot_rsp(
    time: DatetimeIndex,
    signal: np.ndarray,
    ax: Optional[Union[List, Axes]] = None,
    slider: bool = True,
    figsize: int = 300,
) -> Axes:
    """Visualization of Respiration signal.

    Parameters
    ----------
    time : :py:class:`pandas.core.indexes.datetimes.DatetimeIndex`
        The time index.
    signal : :py:class:`numpy.ndarray`
        The physiological signal (1d numpy array).
    ax : :class:`matplotlib.axes.Axes` list or None
        Where to draw the plot. Default is *None* (create a new figure). Only
        applies when `backend="matplotlib"`.
    slider : bool
        If `True`, add a slider to zoom in/out in the signal (only working with
        bokeh backend).
    figsize : int
        Figure heights. Default is `300`.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes` | tuple
        The matplotlib axes containing the plot.
    """

    title = "Respiration recording"
    ylabel = "Respiration recording (mV)"

    #############
    # Upper panel
    #############
    if ax is None:
        _, signal_ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    else:
        signal_ax = ax

    # Signal
    signal_ax.plot(
        time, signal, label="Respiration signal", linewidth=1, color="#c44e52"
    )

    signal_ax.set_title(title)
    signal_ax.set_ylabel(ylabel)

    return signal_ax
