# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Optional

import numpy as np
import pandas as pd
from bokeh.models import Band, ColumnDataSource, Span
from bokeh.plotting import figure
from bokeh.plotting.figure import Figure


def plot_evoked(
    epochs: np.ndarray,
    tmin: float = -1,
    tmax: float = 10,
    sfreq_out: int = 10,
    color: str = "#4c72b0",
    label: Optional[str] = None,
    unit: str = "bpm",
    ax=None,
    figsize: int = 200,
    **kwarg,
) -> Figure:
    """Plot continuous or discontinuous RR intervals time series.

    Parameters
    ----------
    epochs : np.array
        A 2d (trial * time) numpy array containing the time series
        of the epoched signal.
    tmin, tmax : float
        Start and end time of the epochs in seconds, relative to the
        time-locked event. Defaults to -1 and 10, respectively.
    sfreq_out : int
        The sampling frequency of the epoched data.
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
    kwargs: None
        Other keyword arguments are passed down to py:`func:seaborn.lineplot()`.
    figsize : int
        The height of the figure. Default is `200`.

    Returns
    -------
    fig : :class:`bokeh.plotting.figure.Figure`
        The bokeh figure containing the plot.

    """

    ylabel = "R-R interval (ms)" if unit == "rr" else "Beats per minute (bpm)"

    # Create time vector
    time = pd.to_datetime(np.arange(tmin, tmax, 1 / sfreq_out), unit="s", origin="unix")

    p1 = figure(
        title="Instantaneous heart rate",
        sizing_mode="stretch_width",
        plot_height=figsize,
        x_axis_label="Time",
        x_axis_type="datetime",
        y_axis_label=ylabel,
        output_backend="webgl",
    )

    # Vertical and horizontal lines
    vline = Span(
        location=0,
        dimension="height",
        line_color="grey",
        line_width=2,
        line_dash="dashed",
    )
    hline = Span(location=0, dimension="width", line_color="black", line_width=2)
    p1.renderers.extend([vline, hline])

    # Plot
    df = pd.DataFrame(epochs).melt()
    df.variable /= sfreq_out
    df.variable += tmin
    for i in range(len(epochs)):
        p1.line(
            x=time,
            y=epochs[i],
            alpha=0.2,
            line_color=color,
        )

    df_source = pd.DataFrame(
        {
            "time": time,
            "average": epochs.mean(0),
            "lower": epochs.mean(0) - epochs.std(0),
            "upper": epochs.mean(0) + epochs.std(0),
        }
    )

    source = ColumnDataSource(df_source)

    # Show confidence interval
    band = Band(
        base="time",
        lower="lower",
        upper="upper",
        source=source,
        level="underlay",
        fill_alpha=0.4,
        line_width=1,
        fill_color=color,
    )
    p1.add_layout(band)

    # Show average
    p1.line(
        x="time",
        y="average",
        source=source,
        line_width=4,
        legend_label=label,
        line_color=color,
    )

    return p1
