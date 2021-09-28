# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Optional, Union

import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
from bokeh.plotting import figure
from bokeh.plotting.figure import Figure
from matplotlib.axes import Axes
from pandas.core.indexes.datetimes import DatetimeIndex

from systole.plots import plot_rr


def plot_raw(
    time: DatetimeIndex,
    signal: np.ndarray,
    peaks: np.ndarray,
    modality: str = "ppg",
    show_heart_rate: bool = True,
    show_artefacts: bool = False,
    ax: Optional[Union[List, Axes]] = None,
    slider: bool = True,
    figsize: int = 300,
    **kwargs
) -> Figure:
    """Visualization of PPG or ECG signal with systolic peaks/R wave detection.

    The instantaneous heart rate can be derived in a second row.

    Parameters
    ----------
    time : :py:class:`pandas.core.indexes.datetimes.DatetimeIndex`
        The time index.
    signal : :py:class:`numpy.ndarray`
        The physiological signal (1d numpy array).
    peaks : :py:class:`numpy.ndarray`
        The peaks or R wave detection (1d boolean array).
    modality : str
        The recording modality. Can be `"ppg"` or `"ecg"`.
    show_heart_rate : bool
        If `True`, create a second row and plot the instantanesou heart rate
        derived from the physiological signal
        (calls :py:func:`systole.plots.plot_rr` internally). Defaults to `False`.
    show_artefacts : bool
        If `True`, the function will call
        py:func:`systole.detection.rr_artefacts` to detect outliers intervalin the time
        serie and outline them using different colors.
    ax : :class:`matplotlib.axes.Axes` list or None
        Where to draw the plot. Default is *None* (create a new figure). Only
        applies when `backend="matplotlib"`. If `show_heart_rate is True`, a list
        of axes can be provided to plot the signal and instantaneous heart rate
        separately.
    slider : bool
        If `True`, add a slider to zoom in/out in the signal (only working with
        bokeh backend).
    figsize : int
        Figure heights. Default is `300`.
    **kwargs : keyword arguments
        Additional arguments will be passed to
        `:py:func:systole.detection.ppg_peaks()` or
        `:py:func:systole.detection.ecg_peaks()`, depending on the type
        of data.

    Returns
    -------
    raw : :class:`bokeh.plotting.figure.Figure`
        The bokeh figure containing the plot.
    """

    source = ColumnDataSource(
        data={"time": time[::10], "signal": signal[::10], "peaks": peaks[::10]}
    )

    if modality == "ppg":
        title = "PPG recording"
        ylabel = "PPG level (a.u.)"
        peaks_label = "Systolic peaks"
        signal_label = "PPG signal"
    elif modality == "ecg":
        title = "ECG recording"
        ylabel = "ECG (mV)"
        peaks_label = "R wave"
        signal_label = "ECG signal"

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

    raw.circle(
        x=time[peaks],
        y=signal[peaks],
        size=5,
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
            peaks,
            input_type="peaks",
            backend="bokeh",
            figsize=figsize,
            slider=False,
            line=True,
            show_artefacts=show_artefacts,
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
