# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Optional

import pandas as pd
from bokeh.models import BoxAnnotation, Span
from bokeh.plotting import ColumnDataSource, figure
from bokeh.plotting.figure import Figure


def plot_events(
    df: pd.DataFrame,
    figsize: int = 400,
    ax: Optional[Figure] = None,
) -> Figure:
    """Plot events occurence across recording.

    Parameters
    ----------
    df : pd.DataFrame
        The events data frame (tmin, trigger, tmax, label, color, [behavior]).
    figsize : tuple
        Figure size. Default is `(13, 5)`.
    ax : :class:`bokeh.plotting.figure.Figure` or None
        Where to draw the plot. Default is `None` (create a new figure).

    Returns
    -------
    event_plot : :class:`bokeh.plotting.figure.Figure`
        The bokeh figure containing the plot.

    """

    if ax is None:
        TOOLTIPS = [
            ("(x,y)", "($tmin, $tmax)"),
        ]

        event_plot = figure(
            title="Events",
            sizing_mode="stretch_width",
            plot_height=figsize,
            x_axis_label="Time",
            x_axis_type="datetime",
            y_range=(0, df.label.nunique() + 1),
            tooltips=TOOLTIPS,
        )
        # Plot time course of events
        event_source = ColumnDataSource(data=df)

        event_plot.circle(
            x="trigger",
            y=1,
            size=10,
            line_color="color",
            fill_color="white",
            line_width=3,
            source=event_source,
        )
    else:
        event_plot = ax

    # Loop across events df
    for i, tmin, trigger, tmax, label, color in df.itertuples():

        # Plot time range
        event_range = BoxAnnotation(
            left=tmin, right=tmax, fill_alpha=0.2, fill_color=color
        )
        event_plot.add_layout(event_range)

        # Plot trigger
        event_trigger = Span(
            location=trigger,
            dimension="height",
            line_color="gray",
            line_dash="dashed",
            line_width=1,
        )
        event_plot.add_layout(event_trigger)

    return event_plot
