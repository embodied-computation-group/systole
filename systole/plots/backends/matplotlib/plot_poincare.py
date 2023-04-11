# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse

from systole.hrv import nonlinear_domain


def plot_poincare(
    rr: np.ndarray,
    figsize: Optional[Union[List[int], Tuple[int, int], int]] = None,
    ax: Optional[Axes] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Axes]:
    """poincare plot.

    Parameters
    ----------
    rr :
        RR intervals (miliseconds).
    figsize :
        Figure size. Default is `(8, 8)`.
    ax :
        Where to draw the plot. Default is `None` (create a new figure).

     Returns
     -------
     ax  :
        The poincare plot.

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
    ax.plot(
        [range_min, range_max], [range_min, range_max], color="grey", linestyle="--"
    )

    # Compute SD1 and SD2 metrics
    df = nonlinear_domain(rr)
    sd1 = df[df["Metric"] == "SD1"]["Values"].values[0]
    sd2 = df[df["Metric"] == "SD2"]["Values"].values[0]

    # Ellipse
    ellipse_ = Ellipse(
        (rr_x[~outliers].mean(), rr_y[~outliers].mean()),
        sd1 * 2,
        sd2 * 2,
        angle=-45,
        fc="grey",
        zorder=1,
        fill=False,
    )
    ax.add_artist(ellipse_)
    ellipse_ = Ellipse(
        (rr_x[~outliers].mean(), rr_y[~outliers].mean()),
        sd1 * 2,
        sd2 * 2,
        angle=-45,
        fc="#4c72b0",
        alpha=0.4,
        zorder=1,
    )
    ax.add_artist(ellipse_)

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

    # SD1 arrow
    ax.arrow(
        rr_x[~outliers].mean(),
        rr_y[~outliers].mean(),
        -sd1 * np.cos(np.deg2rad(45)),
        sd1 * np.sin(np.deg2rad(45)),
        head_width=10,
        head_length=10,
        fc="b",
        ec="b",
        zorder=4,
        linewidth=1.5,
    )

    # SD2 arrow
    ax.arrow(
        rr_x[~outliers].mean(),
        rr_y[~outliers].mean(),
        sd2 * np.cos(np.deg2rad(45)),
        sd2 * np.sin(np.deg2rad(45)),
        head_width=10,
        head_length=10,
        fc="b",
        ec="g",
        zorder=4,
        linewidth=1.5,
    )

    ax.set_xlabel("RR (n)")
    ax.set_ylabel("RR (n+1)")
    ax.set_title("Poincare plot", fontweight="bold")
    ax.minorticks_on()

    return ax
