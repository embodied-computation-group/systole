# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def plot_evoked(
    epochs: List[np.ndarray],
    time: np.ndarray,
    palette: Iterable,
    figsize: Tuple[float, float],
    labels: str,
    unit: str,
    ax=None,
    ci: Union[int, str] = "sd",
    **kwargs
) -> Axes:
    """Plot events occurence across recording.

    Parameters
    ----------
    epochs : np.array
        A 2d (trial * time) numpy array containing the time series
        of the epoched signal.
    time : float
        Start and end time of the epochs in seconds, relative to the
        time-locked event. Defaults to -1 and 10, respectively.
    palette : int
        The sampling frequency of the epoched data.
    figsize : str
        The lines color.
    labels : str
        The condition label.
    unit : str
        The heart rate unit. Can be `'rr'` (R-R intervals, in ms) or `'bpm'` (beats
        per minutes). Default is `'bpm'`.
    ax : tuple
        Figure size. Default is `(13, 5)`.
    ci : int | str
        The confidence interval around the point estimates. Passed down to
        py:`func:seaborn.lineplot()`.
    kwargs: key, value mappings
        Other keyword arguments are passed down to py:`func:seaborn.lineplot()`.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        The matplotlib axes containing the plot.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.axvline(x=0, linestyle="--", color="gray")
    ax.axhline(y=0, color="black", linewidth=1)

    # Loop across condition
    for ep, lab, col in zip(epochs, labels, palette):

        # Create a dataframe for seaborn
        df = pd.DataFrame(ep.T)
        df["Time"] = time
        df = df.melt(id_vars=["Time"])

        for i in range(ep.shape[0]):
            ax.plot(time, ep[i], color=col, alpha=0.2, linewidth=1)

        sns.lineplot(
            data=df, x="Time", y="value", label=lab, color=col, ax=ax, ci=ci, **kwargs
        )

    ax.set_xlabel("Time (s)")
    ylabel = "R-R interval (ms)" if unit == "rr" else "Beats per minute (bpm)"
    ax.set_ylabel(ylabel)
    ax.minorticks_on()

    return ax
