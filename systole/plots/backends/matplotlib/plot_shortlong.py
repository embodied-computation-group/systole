# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_shortlong(
    artefacts=Dict[str, np.ndarray],
    figsize: int = 600,
    ax: Optional[Axes] = None,
    **kwargs
) -> Axes:
    """Plot ectopic subspace.

    Parameters
    ----------
    artefacts : dict or None
        The artefacts detected using
        :py:func:`systole.detection.rr_artefacts()`.
    figsize : int
        Figure heights. Default is `600`.
    ax : :class:`matplotlib.axes.Axes` or None
        Where to draw the plot. Default is `None` (create a new figure).

    Returns
    -------
    plot : :class:`bokeh.plotting.figure.Figure`
        The boken figure containing the plot.

    See also
    --------
    plot_events, plot_ectopic, plot_shortlong, plot_subspaces, plot_frequency,
    plot_timedomain, plot_nonlinear

    Notes
    -----
    If both ``rr`` or ``artefacts`` are provided, will recompute ``artefacts``
    given the current rr time-series.
    """
    xlim, ylim = 10, 5

    outliers = (
        artefacts["ectopic"]
        | artefacts["short"]
        | artefacts["long"]
        | artefacts["extra"]
        | artefacts["missed"]
    )

    # All values fit in the x and y lims
    for this_art in [artefacts["subspace1"]]:
        this_art[this_art > xlim] = xlim
        this_art[this_art < -xlim] = -xlim
    for this_art in [artefacts["subspace2"]]:
        this_art[this_art > ylim] = ylim
        this_art[this_art < -ylim] = -ylim

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Plot normal beats
    ax.scatter(
        artefacts["subspace1"][~outliers],
        artefacts["subspace3"][~outliers],
        color="gray",
        edgecolors="k",
        alpha=0.2,
        zorder=10,
        s=15,
        label="Normal",
    )

    # Plot outliers
    ax.scatter(
        artefacts["subspace1"][artefacts["ectopic"]],
        artefacts["subspace3"][artefacts["ectopic"]],
        color="r",
        edgecolors="k",
        zorder=10,
        label="Ectopic",
    )
    ax.scatter(
        artefacts["subspace1"][artefacts["short"]],
        artefacts["subspace3"][artefacts["short"]],
        color="b",
        edgecolors="k",
        zorder=10,
        marker="s",
        label="Short",
    )
    ax.scatter(
        artefacts["subspace1"][artefacts["long"]],
        artefacts["subspace3"][artefacts["long"]],
        color="g",
        edgecolors="k",
        zorder=10,
        marker="s",
        label="Long",
    )
    ax.scatter(
        artefacts["subspace1"][artefacts["missed"]],
        artefacts["subspace3"][artefacts["missed"]],
        color="g",
        edgecolors="k",
        zorder=10,
        label="Missed",
    )
    ax.scatter(
        artefacts["subspace1"][artefacts["extra"]],
        artefacts["subspace3"][artefacts["extra"]],
        color="b",
        edgecolors="k",
        zorder=10,
        label="Extra",
    )
    # Upper area
    ax.plot([-1, -10], [1, 1], "k", linewidth=1, linestyle="--")
    ax.plot([-1, -1], [1, 10], "k", linewidth=1, linestyle="--")
    x = [-10, -10, -1, -1]
    y = [1, 10, 10, 1]
    ax.fill(x, y, color="gray", alpha=0.3)

    # Lower area
    ax.plot([1, 10], [-1, -1], "k", linewidth=1, linestyle="--")
    ax.plot([1, 1], [-1, -10], "k", linewidth=1, linestyle="--")
    x = [1, 1, 10, 10]
    y = [-1, -10, -10, -1]
    ax.fill(x, y, color="gray", alpha=0.3)

    ax.set_xlabel("Subspace $S_{21}$")
    ax.set_ylabel("Subspace $S_{22}$")
    ax.set_ylim(-ylim * 2, ylim * 2)
    ax.set_xlim(-xlim, xlim)
    ax.set_title("Subspace 2 \n (long and short beats detection)")
    ax.legend()

    return ax
