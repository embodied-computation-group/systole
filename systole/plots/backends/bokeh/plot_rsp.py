# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Optional, Union

import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
from bokeh.plotting import figure
from bokeh.plotting.figure import Figure
from matplotlib.axes import Axes
from pandas.core.indexes.datetimes import DatetimeIndex

def plot_rsp(
    time: DatetimeIndex,
    signal: np.ndarray,
    ax: Optional[Union[List, Axes]] = None,
    slider: bool = True,
    figsize: int = 300
) -> Figure:
    """Visualization of Respiration signal.

    Parameters
    ----------
    time : :py:class:`pandas.core.indexes.datetimes.DatetimeIndex`
        The time index.
    signal : :py:class:`numpy.ndarray`
        The physiological signal (1d numpy array).
    ax : :class:`matplotlib.axes.Axes` list or None
        Where to draw the plot. Default is *None* (create a new figure). Only
        applies when `backend="matplotlib"`.
    slider : bool
        If `True`, add a slider to zoom in/out in the signal (only working with
        bokeh backend).
    figsize : int
        Figure heights. Default is `300`.

    Returns
    -------
    raw : :class:`bokeh.plotting.figure.Figure`
        The bokeh figure containing the plot.
    """

    source = ColumnDataSource(
        data={"time": time[::10], "signal": signal[::10]}
    )

    title = "Respiration recording"
    ylabel = "Respiration (mV)"
    signal_label = "RSP signal"

    # Raw plot
    ##########

    raw = figure(
        title=title,
        x_axis_type="datetime",
        sizing_mode="stretch_width",
        plot_height=figsize,
        x_axis_label="Time",
        y_axis_label=ylabel,
        output_backend="webgl",
        x_range=(time[0], time[-1]),
    )

    raw.line(
        "time",
        "signal",
        source=source,
        legend_label=signal_label,
        line_color="#a9373b",
    )

    raw.legend.title = "Raw signal"

    cols = (raw,)

    if slider is True:
        select = figure(
            title="Select the time window",
            y_range=raw.y_range,
            y_axis_type=None,
            plot_height=int(figsize * 0.5),
            x_axis_type="datetime",
            tools="",
            toolbar_location=None,
            background_fill_color="#efefef",
        )

        range_tool = RangeTool(x_range=raw.x_range)
        range_tool.overlay.fill_color = "navy"
        range_tool.overlay.fill_alpha = 0.2

        select.line("time", "signal", source=source)
        select.ygrid.grid_line_color = None
        select.add_tools(range_tool)
        select.toolbar.active_multi = range_tool

        cols += (select,)  # type: ignore

    if len(cols) > 1:
        return column(*cols, sizing_mode="stretch_width")
    else:
        return cols[0]
