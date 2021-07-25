# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from pandas.core.indexes.datetimes import DatetimeIndex

from systole.plots import plot_rr


def plot_raw(
    time: DatetimeIndex,
    signal: np.ndarray,
    peaks: np.ndarray,
    modality: str = "ppg",
    show_heart_rate: bool = True,
    ax: Optional[Union[List, Axes]] = None,
    slider: bool = True,
    figsize: int = 300,
    **kwargs
) -> Axes:
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
        (calls :py:func:`systole.plots.plot_rr` internally). Defaults to
        `False`.
    ax : :class:`matplotlib.axes.Axes` list or None
        Where to draw the plot. Default is *None* (create a new figure). Only
        applies when `backend="matplotlib"`. If `show_heart_rate is True`, a
        list of axes can be provided to plot the signal and instantaneous heart
        rate separately.
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
    fig : :class:`matplotlib.axes.Axes` or tuple
        The matplotlib axes containing the plot.
    """

    if modality == "ppg":
        title = "PPG recording"
        ylabel = "PPG level (a.u.)"
        peaks_label = "Systolic peaks"
    elif modality == "ecg":
        title = "ECG recording"
        ylabel = "ECG (mV)"
        peaks_label = "R wave"

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
    signal_ax.plot(time, signal, label="PPG signal", linewidth=1, color="#c44e52")

    # Peaks
    signal_ax.scatter(
        x=time[peaks],
        y=signal[peaks],
        marker="o",
        label=peaks_label,
        s=30,
        color="white",
        edgecolors="DarkSlateGrey",
    )
    if modality == "ppg":
        signal_ax.set_title(title)
        signal_ax.set_ylabel(ylabel)
    elif modality == "ecg":
        signal_ax.set_title(title)
        signal_ax.set_ylabel(ylabel)
    signal_ax.grid(True)

    #############
    # Lower panel
    #############

    if show_heart_rate is True:

        # Instantaneous Heart Rate - Peaks
        plot_rr(
            peaks, input_type="peaks", backend="matplotlib", figsize=figsize, ax=hr_ax
        )

        plt.tight_layout()

        return signal_ax, hr_ax

    else:
        return signal_ax
