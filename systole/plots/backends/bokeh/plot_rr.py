# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, Optional

import numpy as np
import pandas as pd
from bokeh.models import BoxAnnotation, Circle, Line
from bokeh.models.tools import HoverTool
from bokeh.plotting import ColumnDataSource, figure
from bokeh.plotting.figure import Figure

from systole.utils import heart_rate


def plot_rr(
    rr: np.ndarray,
    unit: str = "rr",
    kind: str = "cubic",
    line: bool = True,
    points: bool = True,
    artefacts: Optional[Dict[str, np.ndarray]] = None,
    input_type: str = "peaks",
    show_limits: bool = True,
    ax=None,
    figsize: int = 200,
) -> Figure:
    """Plot continuous or discontinuous RR intervals time series.

    Parameters
    ----------
    rr : np.ndarray
        1d numpy array of RR intervals (in seconds or miliseconds) or peaks
        vector (boolean array).
    unit : str
        The heart rate unit in use. Can be `'rr'` (R-R intervals, in ms)
        or `'bpm'` (beats per minutes). Default is `'rr'`.
    kind : str
        The method to use (parameter of `scipy.interpolate.interp1d`). The
        possible relevant methods for instantaneous heart rate are `'cubic'`
        (defalut), `'linear'`, `'previous'` and `'next'`.
    line : bool
        If `True`, plot the interpolated instantaneous heart rate.
    points : bool
        If `True`, plot each peaks (R wave or systolic peaks) as separated
        points.
    artefacts : dict
        Dictionnary storing the parameters of RR artefacts rejection.
    input_type : str
        The type of input vector. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"`,
        or `"rr_s"`. Default to `"peaks"`.
    show_limits : bool
        Use shaded areas to represent the range of physiologically impossible R-R
        intervals. Defaults to `True`.
    ax : None
        Only relevant when using `backend="matplotlib"`.
    figsize : int
        The height of the figure. Default is `200`.

    Returns
    -------
    fig : :class:`bokeh.plotting.figure.Figure`
        The bokeh figure containing the plot.

    """

    ylabel = "R-R interval (ms)" if unit == "rr" else "Beats per minute (bpm)"

    p1 = figure(
        title="Instantaneous heart rate",
        sizing_mode="stretch_width",
        plot_height=figsize,
        x_axis_label="Time",
        x_axis_type="datetime",
        y_axis_label=ylabel,
        output_backend="webgl",
        tools="pan,wheel_zoom,box_zoom,box_select,reset,save",
    )

    if line is True:

        # Extract instantaneous heart rate
        hr, time = heart_rate(rr, unit=unit, kind=kind, input_type=input_type)

        # Convert to datetime format
        time = pd.to_datetime(time, unit="s", origin="unix")

        # Downsample to 10Hz for plotting
        time = time[::100]
        hr = hr[::100]

        line_source = ColumnDataSource(data=dict(time=time, hr=hr, bpm=60000 / hr))

        # Instantaneous Heart Rate - Lines
        linePlot = Line(
            x="time",
            y="hr",
            line_color="#4c72b0",
        )
        p1.add_glyph(
            line_source,
            linePlot,
        )

    if points is True:

        # Instantaneous Heart Rate - Peaks
        if input_type == "rr_ms":
            ibi = np.array(rr)
            rr_idx = pd.to_datetime(np.cumsum(ibi), unit="ms", origin="unix")
        elif input_type == "rr_s":
            ibi = np.array(rr) * 1000
            rr_idx = pd.to_datetime(np.cumsum(ibi), unit="ms", origin="unix")
        elif input_type == "peaks":
            ibi = np.diff(np.where(rr)[0])
            rr_idx = pd.to_datetime(np.where(rr)[0][1:], unit="ms", origin="unix")
        elif input_type == "peaks_idx":
            ibi = np.diff(rr)
            rr_idx = pd.to_datetime(rr[1:], unit="ms", origin="unix")

        if artefacts is None:
            outliers = np.zeros(len(ibi), dtype=bool)
        else:
            outliers = (
                artefacts["ectopic"]
                | artefacts["short"]
                | artefacts["long"]
                | artefacts["extra"]
                | artefacts["missed"]
            )

        points_source = ColumnDataSource(
            data=dict(
                time=rr_idx,
                rr=ibi,
                bpm=60000 / ibi,
                nbeat=np.arange(1, len(rr_idx) + 1),
                outliers=outliers,
            )
        )

        # Normal RR intervals
        circlePlot = Circle(
            x="time",
            y=unit,
            size=6,
            fill_color="lightgrey",
            line_color="grey",
        )
        circlePlot_selected = Circle(
            x="time",
            y=unit,
            size=15,
            fill_color="firebrick",
            line_color="grey",
        )
        g1 = p1.add_glyph(
            points_source,
            circlePlot,
            hover_glyph=circlePlot_selected,
        )

        hover = HoverTool(
            renderers=[g1],
            tooltips=[
                ("time", "@time{:%M:%S.%3Ns}"),
                ("R-R interval", "@rr{%0.2f} ms"),
                ("BPM", "@bpm{%0.2f} BPM"),
                ("Heartbeat number", "@nbeat"),
            ],
            formatters={"@time": "datetime", "@rr": "printf", "@bpm": "printf"},
            mode="mouse",
        )

        if artefacts is not None:

            # Short RR intervals
            p1.circle(
                x=rr_idx[artefacts["short"]],
                y=ibi[artefacts["short"]],
                size=10,
                legend_label="Short intervals",
                fill_color="#c56c5e",
                line_color="black",
            )

            # Long RR intervals
            p1.circle(
                x=rr_idx[artefacts["long"]],
                y=ibi[artefacts["long"]],
                size=10,
                legend_label="Long intervals",
                fill_color="#9ac1d4",
                line_color="black",
            )

            # Missed RR intervals
            p1.square(
                x=rr_idx[artefacts["missed"]],
                y=ibi[artefacts["missed"]],
                size=10,
                legend_label="Missed intervals",
                fill_color="#2f5f91",
                line_color="black",
            )

            # Extra RR intervals
            p1.square(
                x=rr_idx[artefacts["extra"]],
                y=ibi[artefacts["extra"]],
                size=10,
                legend_label="Extra intervals",
                fill_color="#9d2b39",
                line_color="black",
            )

            # Ectopic beats
            p1.triangle(
                x=rr_idx[artefacts["ectopic"]],
                y=ibi[artefacts["ectopic"]],
                size=10,
                legend_label="Ectopic beats",
                fill_color="#6c0073",
                line_color="black",
            )

        # Add hover tool
        p1.add_tools(hover)

    # Show physiologically impossible ranges
    if show_limits is True:
        high, low = (3000, 200) if unit == "rr" else (20, 300)
        upper_bound = BoxAnnotation(bottom=high, fill_alpha=0.1, fill_color="red")
        p1.add_layout(upper_bound)
        lower_bound = BoxAnnotation(top=low, fill_alpha=0.1, fill_color="red")
        p1.add_layout(lower_bound)

    return p1
