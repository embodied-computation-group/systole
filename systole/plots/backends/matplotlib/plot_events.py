# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


def plot_events(
    df: pd.DataFrame,
    figsize: Tuple[float, float] = (13, 5),
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot events occurence across recording.

    Parameters
    ----------
    df : pd.DataFrame
        The events data frame (tmin, trigger, tmax, label, color, [behavior]).
    figsize : tuple
        Figure size. Default is `(13, 5)`.
    ax : :class:`matplotlib.axes.Axes` or None
        Where to draw the plot. Default is *None* (create a new figure).

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        The matplotlib axes containing the plot.

    """
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 5))

    # Loop across events df
    for i, tmin, trigger, tmax, label, color in df.itertuples():

        # Plot time range
        ax.axvspan(xmin=tmin, xmax=tmax, color=color, alpha=0.2, label=label)

        # Plot trigger
        ax.axvline(x=trigger, color="gray", linestyle="--", linewidth=1)

    # Add y ticks with channels names
    ax.set_xlabel("Time")
    ax.legend()

    return ax
