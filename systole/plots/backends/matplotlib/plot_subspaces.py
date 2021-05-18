# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from systole.detection import rr_artefacts


def plot_subspaces(
    rr: Union[List, np.ndarray],
    c1: float = 0.17,
    c2: float = 0.13,
    xlim: float = 10.0,
    ylim: float = 5.0,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> Axes:
    """Plot hrv subspace as described by Lipponen & Tarvainen (2019).

    Parameters
    ----------
    rr : :py:class:`numpy.ndarray` or list
        Array of RR intervals or subspace1. If subspace1 is provided, subspace2
        and 3 must also be provided.
    c1 : float
        Fixed variable controling the slope of the threshold lines. Default is
        0.13.
    c2 : float
        Fixed variable controling the intersect of the threshold lines. Default
        is 0.17.
    xlim : float
        Absolute range of the x axis. Default is 10.
    ylim : float
        Absolute range of the y axis. Default is 5.
    ax : :class:`matplotlib.axes.Axes` or None
        Where to draw the plot. Default is *None* (create a new figure).
    figsize : tuple
        Figure size. Default set to `(10, 5)`

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        The matplotlib axes containing the plot.

    References
    ----------
    [1] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for
        heart rate variability time series artefact correction using novel beat
        classification. Journal of Medical Engineering & Technology, 43(3),
        173–181. https://doi.org/10.1080/03091902.2019.1640306

    Examples
    --------

    Visualizing artefacts from RR time series.

    .. plot::

       from systole import import_rr
       from systole.plotting import plot_subspaces
       # Import PPG recording as numpy array
       rr = import_rr().rr.to_numpy()
       plot_subspaces(rr)

    """
    if not isinstance(rr, (np.ndarray, np.generic)):
        rr = np.asarray(rr)
    artefacts = rr_artefacts(rr)

    # Rescale to show outlier in scatterplot
    if xlim is not None:
        artefacts["subspace1"][artefacts["subspace1"] < -xlim] = -xlim
        artefacts["subspace1"][artefacts["subspace1"] > xlim] = xlim
    if ylim is not None:
        artefacts["subspace2"][artefacts["subspace2"] < -ylim] = -ylim
        artefacts["subspace2"][artefacts["subspace2"] > ylim] = ylim

        artefacts["subspace3"][artefacts["subspace3"] < -ylim * 2] = -ylim * 2
        artefacts["subspace3"][artefacts["subspace3"] > ylim * 2] = ylim * 2

    # Filter for normal beats
    normalBeats = (
        (~artefacts["ectopic"])
        & (~artefacts["short"])
        & (~artefacts["long"])
        & (~artefacts["missed"])
        & (~artefacts["extra"])
    )

    #############
    # First panel
    #############

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=figsize)

    # Plot normal beats
    ax[0].scatter(
        artefacts["subspace1"][normalBeats],
        artefacts["subspace2"][normalBeats],
        color="gray",
        edgecolors="k",
        s=15,
        alpha=0.2,
        zorder=10,
        label="Normal",
    )

    # Plot outliers
    ax[0].scatter(
        artefacts["subspace1"][artefacts["ectopic"]],
        artefacts["subspace2"][artefacts["ectopic"]],
        color="r",
        edgecolors="k",
        zorder=10,
        label="Ectopic",
    )
    ax[0].scatter(
        artefacts["subspace1"][artefacts["short"]],
        artefacts["subspace2"][artefacts["short"]],
        color="b",
        edgecolors="k",
        zorder=10,
        marker="s",
        label="Short",
    )
    ax[0].scatter(
        artefacts["subspace1"][artefacts["long"]],
        artefacts["subspace2"][artefacts["long"]],
        color="g",
        edgecolors="k",
        zorder=10,
        marker="s",
        label="Long",
    )
    ax[0].scatter(
        artefacts["subspace1"][artefacts["missed"]],
        artefacts["subspace2"][artefacts["missed"]],
        color="g",
        edgecolors="k",
        zorder=10,
        label="Missed",
    )
    ax[0].scatter(
        artefacts["subspace1"][artefacts["extra"]],
        artefacts["subspace2"][artefacts["extra"]],
        color="b",
        edgecolors="k",
        zorder=10,
        label="Extra",
    )

    # Upper area
    def f1(x):
        return -c1 * x + c2

    ax[0].plot([-1, -10], [f1(-1), f1(-10)], "k", linewidth=1, linestyle="--")
    ax[0].plot([-1, -1], [f1(-1), 10], "k", linewidth=1, linestyle="--")
    x = [-10, -10, -1, -1]
    y = [f1(-10), 10, 10, f1(-1)]
    ax[0].fill(x, y, color="gray", alpha=0.3)

    # Lower area
    def f2(x):
        return -c1 * x - c2

    ax[0].plot([1, 10], [f2(1), f2(10)], "k", linewidth=1, linestyle="--")
    ax[0].plot([1, 1], [f2(1), -10], "k", linewidth=1, linestyle="--")
    x = [1, 1, 10, 10]
    y = [f2(1), -10, -10, f2(10)]
    ax[0].fill(x, y, color="gray", alpha=0.3)

    ax[0].set_xlabel("Subspace $S_{11}$")
    ax[0].set_ylabel("Subspace $S_{12}$")
    ax[0].set_ylim(-ylim, ylim)
    ax[0].set_xlim(-xlim, xlim)
    ax[0].set_title("Subspace 1 \n (ectopic beats detection)")
    ax[0].legend()

    ##############
    # Second panel
    ##############

    # Plot normal beats
    ax[1].scatter(
        artefacts["subspace1"][normalBeats],
        artefacts["subspace3"][normalBeats],
        color="gray",
        edgecolors="k",
        alpha=0.2,
        zorder=10,
        s=15,
        label="Normal",
    )

    # Plot outliers
    ax[1].scatter(
        artefacts["subspace1"][artefacts["ectopic"]],
        artefacts["subspace3"][artefacts["ectopic"]],
        color="r",
        edgecolors="k",
        zorder=10,
        label="Ectopic",
    )
    ax[1].scatter(
        artefacts["subspace1"][artefacts["short"]],
        artefacts["subspace3"][artefacts["short"]],
        color="b",
        edgecolors="k",
        zorder=10,
        marker="s",
        label="Short",
    )
    ax[1].scatter(
        artefacts["subspace1"][artefacts["long"]],
        artefacts["subspace3"][artefacts["long"]],
        color="g",
        edgecolors="k",
        zorder=10,
        marker="s",
        label="Long",
    )
    ax[1].scatter(
        artefacts["subspace1"][artefacts["missed"]],
        artefacts["subspace3"][artefacts["missed"]],
        color="g",
        edgecolors="k",
        zorder=10,
        label="Missed",
    )
    ax[1].scatter(
        artefacts["subspace1"][artefacts["extra"]],
        artefacts["subspace3"][artefacts["extra"]],
        color="b",
        edgecolors="k",
        zorder=10,
        label="Extra",
    )
    # Upper area
    ax[1].plot([-1, -10], [1, 1], "k", linewidth=1, linestyle="--")
    ax[1].plot([-1, -1], [1, 10], "k", linewidth=1, linestyle="--")
    x = [-10, -10, -1, -1]
    y = [1, 10, 10, 1]
    ax[1].fill(x, y, color="gray", alpha=0.3)

    # Lower area
    ax[1].plot([1, 10], [-1, -1], "k", linewidth=1, linestyle="--")
    ax[1].plot([1, 1], [-1, -10], "k", linewidth=1, linestyle="--")
    x = [1, 1, 10, 10]
    y = [-1, -10, -10, -1]
    ax[1].fill(x, y, color="gray", alpha=0.3)

    ax[1].set_xlabel("Subspace $S_{21}$")
    ax[1].set_ylabel("Subspace $S_{22}$")
    ax[1].set_ylim(-ylim * 2, ylim * 2)
    ax[1].set_xlim(-xlim, xlim)
    ax[1].set_title("Subspace 2 \n (long and short beats detection)")
    ax[1].legend()

    plt.tight_layout()

    return ax
