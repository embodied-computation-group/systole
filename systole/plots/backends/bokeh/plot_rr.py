# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, Optional

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
    artefacts: Optional[Dict[str, np.ndarray]] = None,
    input_type: str = "peaks",
    figsize: int = 200,
    **kwarg,
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
    sfreq : int
        The sampling frequency of the interpolated line.
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

        # Normal RR intervals
        p1.circle(
            x=peaks_idx[~outliers],
            y=ibi[~outliers],
            legend_label="R-R intervals",
            fill_color="lightgrey",
            line_color="grey",
        )

        if artefacts is not None:

            # Short RR intervals
            p1.circle(
                x=peaks_idx[artefacts["short"]],
                y=ibi[artefacts["short"]],
                size=10,
                legend_label="Short intervals",
                fill_color="#c56c5e",
                line_color="black",
            )

            # Long RR intervals
            p1.circle(
                x=peaks_idx[artefacts["long"]],
                y=ibi[artefacts["long"]],
                size=10,
                legend_label="Long intervals",
                fill_color="#9ac1d4",
                line_color="black",
            )

            # Missed RR intervals
            p1.square(
                x=peaks_idx[artefacts["missed"]],
                y=ibi[artefacts["missed"]],
                size=10,
                legend_label="Missed intervals",
                fill_color="#2f5f91",
                line_color="black",
            )

            # Extra RR intervals
            p1.square(
                x=peaks_idx[artefacts["extra"]],
                y=ibi[artefacts["extra"]],
                size=10,
                legend_label="Extra intervals",
                fill_color="#9d2b39",
                line_color="black",
            )

            # Ectopic beats
            p1.triangle(
                x=peaks_idx[artefacts["ectopic"]],
                y=ibi[artefacts["ectopic"]],
                size=10,
                legend_label="Ectopic beats",
                fill_color="#6c0073",
                line_color="black",
            )

    return p1
