# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes


def plot_raw(
    time: np.ndarray,
    signal: np.ndarray,
    peaks: np.ndarray,
    hr: np.ndarray,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (13, 5),
    modality: str = "ppg",
    **kwargs
) -> Axes:
    """Visualization of PPG signal and systolic peaks detection.

    Parameters
    ----------
    tim : :py:class:`numpy.ndarray`
    signal : :py:class:`numpy.ndarray`
    peaks : :py:class:`numpy.ndarray`
    hr : :py:class:`numpy.ndarray`
    ax : :class:`matplotlib.axes.Axes` or None
        Where to draw the plot. Default is *None* (create a new figure).
    figsize : tuple
        Figure size. Default is `(13, 5)`.
    **kwargs : keyword arguments
        Additional arguments will be passed to
        `:py:func:systole.detection.oxi_peaks()` or
        `:py:func:systole.detection.ecg_peaks()`, depending on the type
        of data.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        The matplotlib axes containing the plot.

    See also
    --------
    plot_events, plot_subspaces, plot_events, plot_psd, plot_oximeter

    Examples
    --------
    Plotting PPG recording.

    .. plot::

       >>> from systole import import_ppg
       >>> from systole.plotting import plot_raw
       >>> # Import PPG recording as pandas data frame
       >>> ppg = import_ppg()
       >>> # Only use the first 60 seconds for demonstration
       >>> ppg = ppg[ppg.time<60]
       >>> plot_raw(ppg)

    Plotting ECG recording.

    .. plot::

       >>> from systole import import_dataset1
       >>> from systole.plotting import plot_raw
       >>> # Import PPG recording as pandas data frame
       >>> ecg = import_dataset1(modalities=['ECG'])
       >>> # Only use the first 60 seconds for demonstration
       >>> ecg = ecg[ecg.time<60]
       >>> plot_raw(ecg, type='ecg', sfreq=1000, ecg_method='pan-tompkins')
    """

    #############
    # Upper panel
    #############
    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=2, figsize=figsize, sharex=True)

    # Signal
    ax[0].plot(time, signal, label="PPG signal", linewidth=1, color="#c44e52")

    # Peaks
    ax[0].scatter(
        x=time[peaks],
        y=signal[peaks],
        marker="o",
        label="Peaks",
        s=30,
        color="white",
        edgecolors="DarkSlateGrey",
    )
    if type == "ppg":
        ax[0].set_title("PPG recording")
        ax[0].set_ylabel("PPG level (a.u.)")
    elif type == "ecg":
        ax[0].set_title("ECG recording")
        ax[0].set_ylabel("ECG (mV)")
    ax[0].grid(True)

    #############
    # Lower panel
    #############

    # Instantaneous Heart Rate - Lines
    ax[1].plot(time, hr, label="R-R intervals", linewidth=1, color="#4c72b0")

    # Instantaneous Heart Rate - Peaks
    ax[1].scatter(
        x=time[peaks],
        y=hr[peaks],
        marker="o",
        label="R-R intervals",
        s=20,
        color="white",
        edgecolors="DarkSlateGrey",
    )
    ax[1].set_title("Instantaneous heart rate")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("R-R interval (ms)")
    ax[1].grid(True)

    plt.tight_layout()
    sns.despine()

    return ax
