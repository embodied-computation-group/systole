# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.plotting.figure import Figure

from systole.utils import heart_rate


def plot_rr(
    rr: np.ndarray,
    unit: str = "rr",
    kind: str = "cubic",
    line: bool = True,
    points: bool = True,
    input_type: str = "peaks",
    figsize: int = 200,
    **kwarg,
) -> Figure:
    """Plot continuous or discontinuous RR intervals time series.

    Parameters
    ----------
    rr : np.ndarray
        1d numpy array of RR intervals (miliseconds).
    unit : str
        The heart rate unit in use. Can be `'rr'` (R-R intervals, in ms)
        or `'bpm'` (beats per minutes). Default is `'rr'`.
    kind : str
        The method to use (parameter of `scipy.interpolate.interp1d`). The
        possible relevant methods for instantaneous heart rate are `'cubic'`
        (defalut), `'linear'`, `'previous'` and `'next'`.
    sfreq : int
        The sampling frequency of the interpolated line.
    line : bool
        If `True`, plot the interpolated instantaneous heart rate.
    points : bool
        If `True`, plot each peaks (R wave or systolic peaks) as separated
        points.
    input_type : str
        The type of input vector. Default is `"peaks"` (a boolean vector where
        `1` represents the occurrence of R waves or systolic peaks).
        Can also be `"rr_s"` or `"rr_ms"` for vectors of RR intervals, or
        interbeat intervals (IBI), expressed in seconds or milliseconds
        (respectively).
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
    )

    if line is True:

        # Extract instantaneous heart rate
        hr, time = heart_rate(rr, unit=unit, kind=kind, input_type=input_type)

        # Convert to datetime format
        time = pd.to_datetime(time, unit="s", origin="unix")

        # Downsample to 5Hz for plotting
        time = time[::200]
        hr = hr[::200]

        # Instantaneous Heart Rate - Lines
        p1.line(
            x=time,
            y=hr,
            legend_label="Instantaneous heart rate",
            line_color="#4c72b0",
        )

    if points is True:

        # Instantaneous Heart Rate - Peaks
        if input_type == "rr_ms":
            ibi = np.array(rr)
            peaks_idx = pd.to_datetime(np.cumsum(ibi), unit="ms", origin="unix")
        elif input_type == "rr_s":
            ibi = np.array(rr) * 1000
            peaks_idx = pd.to_datetime(np.cumsum(ibi) * 1000, unit="ms", origin="unix")
        elif input_type == "peaks":
            ibi = np.diff(np.where(rr)[0])
            peaks_idx = pd.to_datetime(np.where(rr)[0][1:], unit="ms", origin="unix")

        if unit == "bpm":
            ibi = 60000 / ibi

        p1.circle(
            x=peaks_idx,
            y=ibi,
            legend_label="R-R intervals",
            fill_color="lightgrey",
            line_color="grey",
        )

    p1.legend.title = "Instantaneous heart rate"

    return p1
