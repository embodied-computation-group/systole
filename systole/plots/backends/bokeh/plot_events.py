# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Optional

import pandas as pd
from bokeh.models import BoxAnnotation, Span
from bokeh.plotting import ColumnDataSource, figure


def plot_events(
    df: pd.DataFrame,
    figsize: int = 400,
    ax: Optional[figure] = None,
    behavior: Optional[List[pd.DataFrame]] = None,
) -> figure:
    """Plot events to get a visual display of the paradigm (Bokeh).

    Parameters
    ----------
    df :
        The events data frame (tmin, trigger, tmax, label, color, [behavior]).
    figsize :
        Figure size. Default is `(13, 5)`.
    ax :
        Where to draw the plot. Default is `None` (create a new figure).
    behavior :
        (Optional) Additional information about trials that will appear when hovering
        on the area (`bokeh` version only).

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
            height=figsize,
            x_axis_label="Time",
            x_axis_type="datetime",
            y_range=(0, 1),
            tooltips=TOOLTIPS,
        )
        # Plot time course of events
        event_source = ColumnDataSource(data=df)

        event_plot.circle(
            x="trigger",
            y=0.5,
            size=10,
            line_color="color",
            fill_color="white",
            legend_field="label",
            line_width=3,
            source=event_source,
        )
        # Hide y axis if no other time series is provided
        event_plot.yaxis.visible = False

    else:
        event_plot = ax

    # Loop across events df
    for _, tmin, trigger, tmax, _, color in df.itertuples():
        # Plot time range
        event_range = BoxAnnotation(
            left=tmin, right=tmax, fill_alpha=0.2, fill_color=color
        )
        event_range.level = "underlay"
        event_plot.add_layout(event_range)

        # Plot trigger
        event_trigger = Span(
            location=trigger,
            dimension="height",
            line_color="gray",
            line_dash="dashed",
            line_width=1,
        )
        event_trigger.level = "underlay"
        event_plot.add_layout(event_trigger)

    return event_plot
