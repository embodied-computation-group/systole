# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import TYPE_CHECKING, Dict

import numpy as np

from bokeh.plotting import figure

if TYPE_CHECKING:
    from bokeh.plotting.figure import Figure


def plot_shortLong(
    artefacts=Dict[str, np.ndarray], figsize: int = 600, **kwargs
) -> "Figure":
    """Plot interactive short/long subspace.

    Parameters
    ----------
    artefacts : dict or None
        The artefacts detected using
        :py:func:`systole.detection.rr_artefacts()`.
    figsize : int
        Figure heights. Default is `600`.

    Returns
    -------
    plot : :class:`bokeh.plotting.figure.Figure`
        The boken figure containing the plot.

    See also
    --------
    plot_events, plot_ectopic, plot_shortLong, plot_subspaces, plot_frequency,
    plot_timedomain, plot_nonlinear

    Notes
    -----
    If both ``rr`` or ``artefacts`` are provided, will recompute ``artefacts``
    given the current rr time-series.
    """
    xlim, ylim = 10, 10

    outliers = (
        artefacts["ectopic"]
        | artefacts["short"]
        | artefacts["long"]
        | artefacts["extra"]
        | artefacts["missed"]
    )

    # All values fit in the x and y lims
    for this_art in [artefacts["subspace1"], artefacts["subspace3"]]:
        this_art[this_art > xlim] = xlim
        this_art[this_art < -xlim] = -xlim
        this_art[this_art > ylim] = ylim
        this_art[this_art < -ylim] = -ylim

    shorLong_plot = figure(
        title="Short and long intervals",
        plot_height=figsize,
        plot_width=figsize,
        x_axis_label="Subspace 1",
        y_axis_label="Subspace 3",
        output_backend="webgl",
        x_range=[-xlim, xlim],
        y_range=[-ylim, ylim],
    )

    # Upper area
    shorLong_plot.patch([-10, -10, -1, -1], [1, 10, 10, 1], alpha=0.2, color="grey")

    # Lower area
    shorLong_plot.patch([1, 1, 10, 10], [-1, -10, -10, -1], alpha=0.2, color="grey")

    # Plot normal intervals
    shorLong_plot.circle(
        artefacts["subspace1"][~outliers],
        artefacts["subspace3"][~outliers],
        color="gray",
        size=8,
        alpha=0.2,
        legend_label="Standard IBI",
    )

    # Plot ectopic beats
    shorLong_plot.triangle(
        artefacts["subspace1"][artefacts["ectopic"]],
        artefacts["subspace3"][artefacts["ectopic"]],
        size=8,
        alpha=0.8,
        legend_label="Ectopic beats",
        color="#6c0073",
    )

    # Plot missed beats
    shorLong_plot.square(
        artefacts["subspace1"][artefacts["missed"]],
        artefacts["subspace3"][artefacts["missed"]],
        size=8,
        alpha=0.8,
        legend_label="Missed beats",
        color="#2f5f91",
    )

    # Plot long beats
    shorLong_plot.circle(
        artefacts["subspace1"][artefacts["long"]],
        artefacts["subspace3"][artefacts["long"]],
        size=8,
        alpha=0.8,
        legend_label="Long beats",
        color="#9ac1d4",
    )

    # Plot extra beats
    shorLong_plot.square(
        artefacts["subspace1"][artefacts["extra"]],
        artefacts["subspace3"][artefacts["extra"]],
        size=8,
        alpha=0.8,
        legend_label="Extra beats",
        color="#9d2b39",
    )

    # Plot short beats
    shorLong_plot.circle(
        artefacts["subspace1"][artefacts["short"]],
        artefacts["subspace3"][artefacts["short"]],
        size=8,
        alpha=0.8,
        legend_label="Short beats",
        color="#c56c5e",
    )

    return shorLong_plot
