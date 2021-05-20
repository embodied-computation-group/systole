# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_pointcare(
    rr: np.ndarray,
    figsize: Optional[Union[List[int], Tuple[int, int], int]] = None,
    ax: Optional[Axes] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Axes]:
    """Pointcare plot.

    Parameters
    ----------
    rr : np.ndarray
        RR intervals (miliseconds).
    figsize : list, tuple, int or None
        Figure size. Default is `(8, 8)`.
    ax : :class:`matplotlib.axes.Axes` or None
        Where to draw the plot. Default is `None` (create a new figure).

     Returns
     -------
     ax  : :class:`matplotlib.axes.Axes`
        The pointcare plot.

    """
    if figsize is None:
        figsize = (8, 8)
    elif isinstance(figsize, int):
        figsize = (figsize, figsize)
    else:
        if len(figsize) == 1:
            figsize = (figsize[0], figsize[0])

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if np.any(rr >= 3000) | np.any(rr <= 200):

        # Set outliers to reasonable values for plotting
        rr[np.where(rr > 3000)[0]] = 3000
        rr[np.where(rr < 200)[0]] = 200

    # Create x and y vectors
    rr_x, rr_y = rr[:-1], rr[1:]

    # Find outliers idx
    outliers = (rr_x == 3000) | (rr_x == 200) | (rr_y == 3000) | (rr_y == 200)
    range_min, range_max = rr.min() - 50, rr.max() + 50

    # Identity line
    ax.plot([range_min, range_max], [range_min, range_max], color="grey")

    # Scatter plot - valid intervals
    ax.scatter(
        rr_x[~outliers],
        rr_y[~outliers],
        s=5,
        color="#4c72b0",
        alpha=0.1,
        edgecolors="grey",
    )

    # Scatter plot - outliers
    ax.scatter(
        rr_x[outliers],
        rr_y[outliers],
        s=5,
        color="#a9373b",
        alpha=0.8,
        edgecolors="grey",
    )

    ax.set_xlabel("RR (n)")
    ax.set_ylabel("RR (n+1)")
    ax.set_title("Pointcare plot", fontweight="bold")

    return ax
