# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from systole.plots import plot_events
from systole.utils import heart_rate


def plot_rr(
    rr: np.ndarray,
    unit: str = "rr",
    kind: str = "cubic",
    line: bool = True,
    points: bool = True,
    artefacts: Optional[Dict[str, np.ndarray]] = None,
    bad_segments: Optional[List[Tuple[int, int]]] = None,
    input_type: str = "peaks",
    ax: Optional[Axes] = None,
    show_limits: bool = True,
    slider=None,
    figsize: Tuple[float, float] = (13, 5),
    events_params: Optional[Dict] = None,
) -> Axes:
    """Plot continuous or discontinuous RR intervals time series.

    Parameters
    ----------
    rr :
        1d numpy array of RR intervals (in seconds or miliseconds) or peaks vector
        (boolean array).
    unit :
        The heart rate unit in use. Can be `'rr'` (R-R intervals, in ms)or `'bpm'`
        (beats per minutes). Default is `'rr'`.
    kind :
        The method to use (parameter of `scipy.interpolate.interp1d`). The possible
        relevant methods for instantaneous heart rate are `'cubic'` (defalut),
        `'linear'`, `'previous'` and `'next'`.
    line :
        If `True`, plot the interpolated instantaneous heart rate.
    points :
        If `True`, plot each peaks (R wave or systolic peaks) as separated points.
    artefacts :
        Dictionary storing the parameters of RR artefacts rejection.
    bad_segments :
        Mark some portion of the recording as bad. Grey areas are displayed on the top
        of the signal to help visualization (this is not correcting or transforming the
        post-processed signals). Should be a list of tuples shuch as (start_idx,
        end_idx) for each segment.
    input_type :
        The type of input vector. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"`, or
        `"rr_s"`. Default to `"peaks"`.
    ax :
        Where to draw the plot. Default is *None* (create a new figure).
    show_limits :
        Use shaded areas to represent the range of physiologically impossible R-R
        intervals. Defaults to `True`.
    slider :
        If `True`, add a slider to zoom in/out in the signal (only working with
        bokeh backend).
    figsize :
        Figure size. Default is `(13, 5)`.
    events_params :
        (Optional) Additional parameters that will be passed to
       :py:func:`systole.plots.plot_events` and plot the events timing in the backgound.

    Returns
    -------
    ax :
        The matplotlib axes containing the plot.

    """

    ylabel = "R-R interval (ms)" if unit == "rr" else "Beats per minute (bpm)"

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Plot the events in the background if required
    if events_params is not None:
        plot_events(**events_params, ax=ax)

    if line is True:
        # Extract instantaneous heart rate
        hr, time = heart_rate(rr, unit=unit, kind=kind, input_type=input_type)

        # Convert to datetime format
        time = pd.to_datetime(time, unit="s", origin="unix")

        # Instantaneous Heart Rate - Lines
        ax.plot(
            time,
            hr,
            label="Instantaneous heart rate",
            linewidth=0.75,
            color="#4c72b0",
            alpha=0.4,
            zorder=1,
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

        ax.scatter(
            x=peaks_idx[~outliers],
            y=ibi[~outliers],
            marker="o",
            label="R-R intervals",
            s=20,
            color="lightgray",
            edgecolors="gray",
            zorder=2,
        )

    if artefacts is not None:
        # Short RR intervals
        if artefacts["short"].any():
            ax.scatter(
                x=peaks_idx[artefacts["short"]],
                y=ibi[artefacts["short"]],
                s=20,
                label="Short intervals",
                color="#c56c5e",
                edgecolors="black",
            )

        # Long RR intervals
        if artefacts["long"].any():
            ax.scatter(
                x=peaks_idx[artefacts["long"]],
                y=ibi[artefacts["long"]],
                s=20,
                label="Long intervals",
                color="#9ac1d4",
                edgecolors="black",
            )

        # Missed RR intervals
        if artefacts["missed"].any():
            ax.scatter(
                x=peaks_idx[artefacts["missed"]],
                y=ibi[artefacts["missed"]],
                s=20,
                marker="s",
                label="Missed intervals",
                color="#2f5f91",
                edgecolors="black",
            )

        # Extra RR intervals
        if artefacts["extra"].any():
            ax.scatter(
                x=peaks_idx[artefacts["extra"]],
                y=ibi[artefacts["extra"]],
                s=20,
                marker="s",
                label="Extra intervals",
                color="#9d2b39",
                edgecolors="black",
            )

        # Ectopic beats
        if artefacts["ectopic"].any():
            ax.scatter(
                x=peaks_idx[artefacts["ectopic"]],
                y=ibi[artefacts["ectopic"]],
                s=20,
                marker="^",
                label="Ectopic beats",
                color="#6c0073",
                edgecolors="black",
            )

    # Show physiologically impossible ranges
    if show_limits is True:
        high, low = (3000, 200) if unit == "rr" else (300, 20)
        if points is True:
            if (ibi > high).any() | (ibi < low).any():
                ylim_low, ylim_high = ax.get_ylim()
                ax.axhspan(ymin=ylim_low, ymax=low, color="r", alpha=0.1)
                ax.axhspan(ymin=high, ymax=ylim_high, color="r", alpha=0.1)
                ax.set_ylim(ylim_low, ylim_high)
        else:
            if (hr > high).any() | (hr < low).any():
                ylim_low, ylim_high = ax.get_ylim()
                ax.axhspan(ymin=ylim_low, ymax=low, color="r", alpha=0.1)
                ax.axhspan(ymin=high, ymax=ylim_high, color="r", alpha=0.1)
                ax.set_ylim(ylim_low, ylim_high)

    # Show bad segments if any
    if bad_segments is not None:
        if line is False:
            # Create the time vector
            hr, time = heart_rate(rr, unit=unit, kind=kind, input_type=input_type)
            time = pd.to_datetime(time, unit="s", origin="unix")

        for bads in bad_segments:
            ax.axvspan(xmin=time[bads[0]], xmax=time[bads[1]], color="grey", alpha=0.2)

    ax.set_title("Instantaneous heart rate")
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")

    return ax
