# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import pandas as pd

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
from bokeh.plotting import figure
from bokeh.plotting.figure import Figure
from systole.plots import plot_rr


def plot_raw(
    time: np.ndarray,
    signal: np.ndarray,
    peaks: np.ndarray,
    modality: str = "ppg",
    show_heart_rate: bool = True,
    slider: bool = True,
    figsize: int = 300,
    ax=None,
    **kwargs
) -> Figure:
    """Visualization of PPG signal and systolic peaks detection.

    Parameters
    ----------
    time : :py:class:`numpy.ndarray`
    signal : :py:class:`numpy.ndarray`
    peaks : :py:class:`numpy.ndarray`
    modality : str
    show_heart_rate : bool
    slider : bool
    figsize : int
        Figure heights. Default is `300`.
    **kwargs : keyword arguments
        Additional arguments will be passed to
        `:py:func:systole.detection.oxi_peaks()` or
        `:py:func:systole.detection.ecg_peaks()`, depending on the type
        of data.

    Returns
    -------
    fig : :class:`bokeh.plotting.figure.Figure`
        The bokeh figure containing the plot.
    """

    time = pd.to_datetime(time, unit="s", origin="unix")
    source = ColumnDataSource(
        data={"time": time[::10], "signal": signal[::10], "peaks": peaks[::10]}
    )

    if modality == "ppg":
        title = "PPG recording"
        ylabel = "PPG level (a.u.)"
        peaks_label = "Systolic peaks"
    elif modality == "ecg":
        title = "ECG recording"
        ylabel = "ECG (mV)"
        peaks_label = "R wave"

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

    # Instantaneous Heart Rate - Lines
    raw.line(
        "time",
        "signal",
        source=source,
        legend_label="PPG signal",
        line_color="#a9373b",
    )

    raw.circle(
        x=time[peaks],
        y=signal[peaks],
        size=10,
        legend_label=peaks_label,
        fill_color="lightgrey",
        line_color="grey",
    )
    raw.legend.title = "Raw signal"

    cols = (raw,)

    # Instantaneous heart rate
    ##########################
    if show_heart_rate is True:
        instantaneous_hr = plot_rr(
            peaks, input_type="peaks", backend="bokeh", figsize=figsize
        )
        instantaneous_hr.x_range = raw.x_range

        cols += (instantaneous_hr,)  # type: ignore

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
