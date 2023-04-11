# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict

import numpy as np
from bokeh.plotting._figure import figure


def plot_ectopic(
    artefacts=Dict[str, np.ndarray], figsize: int = 600, ax=None
) -> figure:
    """Plot interactive ectopic subspace.

    Parameters
    ----------
    artefacts : dict or None
        The artefacts detected using
        :py:func:`systole.detection.rr_artefacts()`.
    figsize : int
        Figure heights. Default is `600`.
    ax : None
        Only apply when using Matplotlib backend.

    Returns
    -------
    ectopic_plot : :class:`bokeh.plotting.figure.Figure`
        The boken figure containing the plot.

    """
    c1, c2, xlim, ylim = 0.13, 0.17, 10, 5

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

    ectopic_plot = figure(
        title="Ectopic beats",
        height=figsize,
        width=figsize,
        x_axis_label="Subspace 1",
        y_axis_label="Subspace 2",
        output_backend="webgl",
        x_range=[-xlim, xlim],
        y_range=[-ylim, ylim],
    )

    # Upper area
    def f1(x):
        return -c1 * x + c2

    ectopic_plot.patch(
        [-10, -10, -1, -1], [f1(-5), 5, 5, f1(-1)], alpha=0.2, color="grey"
    )

    # Lower area
    def f2(x):
        return -c1 * x - c2

    ectopic_plot.patch([1, 1, 10, 10], [f2(1), -5, -5, f2(5)], alpha=0.2, color="grey")

    # Plot normal intervals
    ectopic_plot.circle(
        artefacts["subspace1"][~outliers],
        artefacts["subspace2"][~outliers],
        color="gray",
        size=8,
        alpha=0.2,
        legend_label="Standard IBI",
    )

    # Plot ectopic beats
    if artefacts["ectopic"].any():
        ectopic_plot.triangle(
            artefacts["subspace1"][artefacts["ectopic"]],
            artefacts["subspace2"][artefacts["ectopic"]],
            size=8,
            alpha=0.8,
            legend_label="Ectopic beats",
            color="#6c0073",
        )

    # Plot missed beats
    if artefacts["missed"].any():
        ectopic_plot.square(
            artefacts["subspace1"][artefacts["missed"]],
            artefacts["subspace2"][artefacts["missed"]],
            size=8,
            alpha=0.8,
            legend_label="Missed beats",
            color="#2f5f91",
        )

    # Plot long beats
    if artefacts["long"].any():
        ectopic_plot.circle(
            artefacts["subspace1"][artefacts["long"]],
            artefacts["subspace2"][artefacts["long"]],
            size=8,
            alpha=0.8,
            legend_label="Long beats",
            color="#9ac1d4",
        )

    # Plot extra beats
    if artefacts["extra"].any():
        ectopic_plot.square(
            artefacts["subspace1"][artefacts["extra"]],
            artefacts["subspace2"][artefacts["extra"]],
            size=8,
            alpha=0.8,
            legend_label="Extra beats",
            color="#9d2b39",
        )

    # Plot short beats
    if artefacts["short"].any():
        ectopic_plot.circle(
            artefacts["subspace1"][artefacts["short"]],
            artefacts["subspace2"][artefacts["short"]],
            size=8,
            alpha=0.8,
            legend_label="Short beats",
            color="#c56c5e",
        )

    return ectopic_plot
