# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Optional, Tuple, Union

import numpy as np
from matplotlib.axes import Axes

from bokeh.plotting.figure import Figure
from systole.plots.utils import get_plotting_function


def plot_evoked(
    epochs: np.ndarray,
    tmin: float = -1,
    tmax: float = 10,
    sfreq: int = 10,
    color: str = "#4c72b0",
    label: Optional[str] = None,
    unit: str = "bpm",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 5),
    backend: str = "matplotlib",
    **kwargs
) -> Union[Figure, Axes]:
    """Plot events occurence across recording.

    Parameters
    ----------
    epochs : np.array
        A 2d (trial * time) numpy array containing the time series
        of the epoched signal.
    tmin, tmax : float
        Start and end time of the epochs in seconds, relative to the
        time-locked event. Defaults to -1 and 10, respectively.
    sfreq : int
        The sampling frequency.
    figsize : tuple
        Figure size. Default is `(13, 5)`.
    color : str
        The lines color.
    label : str
        The condition label.
    unit : str
        The heart rate unit in use. Can be `'rr'` (R-R intervals, in ms)
        or `'bpm'` (beats per minutes). Default is `'bpm'`.
    ax : :class:`matplotlib.axes.Axes` or None
        Where to draw the plot. Default is *None* (create a new figure).
    figsize : tuple
        The figure size.
    kwargs: key, value mappings
        Other keyword arguments are passed down to
        py:`func:seaborn.lineplot()`.

    Returns
    -------
    plot : :class:`matplotlib.axes.Axes` or :class:`bokeh.plotting.figure.Figure`
        The matplotlib axes, or the boken figure containing the plot.
    """
    if figsize is None:
        if backend == "matplotlib":
            figsize = (13, 5)
        elif backend == "bokeh":
            figsize = 300

    plot_evoked_args = {
        "epochs": epochs,
        "tmin": tmin,
        "tmax": tmax,
        "sfreq": sfreq,
        "color": color,
        "ax": ax,
        "figsize": figsize,
        "label": label,
    }

    plotting_function = get_plotting_function("plot_evoked", "plot_evoked", backend)
    plot = plotting_function(**plot_evoked_args)

    return plot
