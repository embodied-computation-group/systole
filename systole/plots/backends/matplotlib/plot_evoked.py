# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bokeh.plotting.figure import Figure
from matplotlib.axes import Axes


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
    **kwargs
) -> Figure:
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
        or `'bpm'` (beats per minutes). Default is `'rr'`.
    ax : :class:`matplotlib.axes.Axes` or None
        Where to draw the plot. Default is *None* (create a new figure).
    figsize : tuple
        The figure size.
    kwargs: key, value mappings
        Other keyword arguments are passed down to py:`func:seaborn.lineplot()`.

    Returns
    -------
    fig : :class:`bokeh.plotting.figure.Figure`
        The bokeh figure containing the plot.
    """
    time = np.arange(tmin, tmax, 1 / sfreq)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.axvline(x=0, linestyle="--", color="gray")
    ax.axhline(y=0, color="black")

    # Plot
    df = pd.DataFrame(epochs).melt()
    df.variable /= sfreq
    df.variable += tmin
    for i in range(len(epochs)):
        ax.plot(time, epochs[i], color=color, alpha=0.2, linestyle="--")

    sns.lineplot(
        data=df,
        x="variable",
        y="value",
        ci=68,
        label=label,
        color=color,
        ax=ax,
        **kwargs
    )

    ax.set_xlabel("Time (s)")
    ylabel = "R-R interval (ms)" if unit == "rr" else "Beats per minute (bpm)"
    ax.set_ylabel(ylabel)
    ax.minorticks_on()

    return ax
