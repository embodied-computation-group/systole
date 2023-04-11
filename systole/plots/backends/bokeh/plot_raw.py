# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Optional, Tuple

import numpy as np
from bokeh.layouts import column
from bokeh.models import BoxAnnotation, ColumnDataSource, RangeTool
from bokeh.plotting import figure
from pandas.core.indexes.datetimes import DatetimeIndex

from systole.plots import plot_rr
from systole.utils import ecg_strings, ppg_strings, resp_strings


def plot_raw(
    time: DatetimeIndex,
    signal: np.ndarray,
    peaks: np.ndarray,
    modality: str = "ppg",
    show_heart_rate: bool = True,
    show_artefacts: bool = False,
    bad_segments: Optional[List[Tuple[int, int]]] = None,
    decim: int = 10,
    slider: bool = True,
    figsize: int = 300,
    events_params: Optional[Dict] = None,
    **kwargs
) -> figure:
    """Visualization of PPG or ECG signal with systolic peaks/R wave detection.

    The instantaneous heart rate can be derived in a second row.

    Parameters
    ----------
    time :
        The time index.
    signal :
        The physiological signal (1d numpy array).
    peaks :
        The peaks or R wave detection (1d boolean array).
    modality :
        The recording modality. Can be `"ppg"`, `"ecg"` or `"resp"`.
    show_heart_rate :
        If `True`, create a second row and plot the instantanesou heart rate
        derived from the physiological signal
        (calls :py:func:`systole.plots.plot_rr` internally). Defaults to `False`.
    show_artefacts :
        If `True`, the function will call
       :py:func:`systole.detection.rr_artefacts` to detect outliers intervalin the time
        serie and outline them using different colors.
    bad_segments :
        Mark some portion of the recording as bad. Grey areas are displayed on the top
        of the signal to help visualization (this is not correcting or transforming the
        post-processed signals). Should be a list of tuples shuch as (start_idx,
        end_idx) for each segment.
    decim :
        Factor by which to subsample the raw signal. Selects every Nth sample (where N
        is the value passed to decim). Default set to `10` (considering that the imput
        signal has a sampling frequency of 1000 Hz) to save memory.
    slider :
        If `True`, add a slider to zoom in/out in the signal (only working with
        bokeh backend).
    figsize :
        Figure heights. Default is `300`.
    events_params :
        (Optional) Additional parameters that will be passed to
       :py:func:`systole.plots.plot_events` and plot the events timing in the backgound.
    kwargs:
        Other keyword arguments passed to the function but unused by the Bokeh backend.

    Returns
    -------
    raw :
        The bokeh figure containing the plot.

    """

    source = ColumnDataSource(
        data={"time": time[::decim], "signal": signal[::decim], "peaks": peaks[::decim]}
    )

    if modality in ppg_strings:
        title = "PPG recording"
        ylabel = "PPG level (a.u.)"
        peaks_label = "Systolic peaks"
        signal_label = "PPG signal"
    elif modality in ecg_strings:
        title = "ECG recording"
        ylabel = "ECG (mV)"
        peaks_label = "R wave"
        signal_label = "ECG signal"
    elif modality in resp_strings:
        title = "Respiration"
        ylabel = "Respiratory signal"
        peaks_label = "End of inspiration"
        signal_label = "Respiratory signal"

    ############
    # Raw plot #
    ############

    raw = figure(
        title=title,
        x_axis_type="datetime",
        sizing_mode="stretch_width",
        height=figsize,
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

    # Highlight bad segments if provided
    if bad_segments is not None:
        for bads in bad_segments:
            # Plot time range
            event_range = BoxAnnotation(
                left=time[bads[0]],
                right=time[bads[1]],
                fill_alpha=0.2,
                fill_color="grey",
            )
            event_range.level = "underlay"
            raw.add_layout(event_range)

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
            events_params=events_params,
        )
        instantaneous_hr.x_range = raw.x_range

        cols += (instantaneous_hr,)  # type: ignore

    if slider is True:
        select = figure(
            title="Select the time window",
            y_range=raw.y_range,
            y_axis_type=None,
            height=int(figsize * 0.5),
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

        cols += (select,)  # type: ignore

    if len(cols) > 1:
        return column(*cols, sizing_mode="stretch_width")
    else:
        return cols[0]
