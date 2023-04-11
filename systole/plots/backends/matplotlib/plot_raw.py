# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
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
    ax: Optional[Union[List, Axes]] = None,
    slider: bool = True,
    figsize: int = 300,
    events_params: Optional[Dict] = None,
) -> Axes:
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
        The recording modality. Can be one of the modality strings defined in
        `systole.utils.ppg_string`, `systole.utils.ecg_string` or
        `systole.utils.resp_string`.
    show_heart_rate :
        If `True`, create a second row and plot the instantanesou heart rate
        derived from the physiological signal
        (calls :py:func:`systole.plots.plot_rr` internally). Defaults to
        `False`.
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
    ax :
        Where to draw the plot. Default is *None* (create a new figure). Only
        applies when `backend="matplotlib"`. If `show_heart_rate is True`, a
        list of axes can be provided to plot the signal and instantaneous heart
        rate separately.
    slider :
        If `True`, add a slider to zoom in/out in the signal (only working with
        bokeh backend).
    figsize :
        Figure heights. Default is `300`.
    events_params :
        (Optional) Additional parameters that will be passed to
       :py:func:`systole.plots.plot_events` and plot the events timing in the backgound.

    Returns
    -------
    ax :
        The matplotlib axes containing the plot.

    """

    if modality in ppg_strings:
        title = "PPG recording"
        ylabel = "PPG level (a.u.)"
        peaks_label = "Systolic peaks"
    elif modality in ecg_strings:
        title = "ECG recording"
        ylabel = "ECG (mV)"
        peaks_label = "R wave"
    elif modality in resp_strings:
        title = "Respiration"
        ylabel = "Respiratory signal"
        peaks_label = "End of inspiration"
    else:
        raise ValueError(
            "Invalid modality parameter. See systole.utils.ecg_strings, "
            "systole.utils.ppg_strings or systole.utils.resp_strings "
            "for valid parameters."
        )

    #############
    # Upper panel
    #############
    if ax is None:
        if show_heart_rate is True:
            _, axs = plt.subplots(ncols=1, nrows=2, figsize=figsize, sharex=True)
            signal_ax, hr_ax = axs
        else:
            _, signal_ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)

    elif isinstance(ax, list):
        signal_ax, hr_ax = ax
    else:
        signal_ax = ax

    # Signal
    signal_ax.plot(
        time[::decim],
        signal[::decim],
        label=ylabel,
        linewidth=0.5,
        color="#c44e52",
        alpha=0.5,
        zorder=-1,
    )

    # Peaks
    signal_ax.scatter(
        x=time[peaks],
        y=signal[peaks],
        marker="o",
        label=peaks_label,
        s=20,
        color="grey",
        edgecolors="DarkSlateGrey",
    )
    signal_ax.set_title(title)
    signal_ax.set_ylabel(ylabel)

    if bad_segments is not None:
        for bads in bad_segments:
            signal_ax.axvspan(
                xmin=time[bads[0]], xmax=time[bads[1]], color="grey", alpha=0.2
            )

    #############
    # Lower panel
    #############
    if show_heart_rate is True:
        # Instantaneous Heart Rate - Peaks
        plot_rr(
            peaks,
            input_type="peaks",
            backend="matplotlib",
            figsize=figsize,
            show_artefacts=show_artefacts,
            bad_segments=bad_segments,
            ax=hr_ax,
            events_params=events_params,
        )

        return signal_ax, hr_ax

    else:
        return signal_ax
