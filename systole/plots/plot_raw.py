# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from bokeh.plotting.figure import Figure
from systole.detection import ecg_peaks, oxi_peaks
from systole.plots.utils import get_plotting_function


def plot_raw(
    signal: Union[pd.DataFrame, np.ndarray, List],
    sfreq: int = 75,
    modality: str = "ppg",
    ecg_method: str = "hamilton",
    show_heart_rate: bool = False,
    slider: bool = True,
    ax: Optional[Axes] = None,
    figsize: Optional[Union[int, List[int], Tuple[int, int]]] = None,
    backend: str = "matplotlib",
    **kwargs
) -> Union[Axes, Figure]:
    """Visualization of PPG or ECG signal with systolic peaks/R wave detection.

    The instantaneous heart rate can be derived in a second row.

    Parameters
    ----------
    signal : :py:class:`pandas.DataFrame`, :py:class:`numpy.ndarray` or list
        Dataframe of PPG or ECG signal in the long format. If a data frame is
        provided, it should contain at least one ``'time'`` and one colum for
        signal(either ``'ppg'`` or ``'ecg'``). If an array is provided, it will
        automatically create a DataFrame using the array as signal and
        ``sfreq`` as sampling frequency.
    sfreq : int
        Signal sampling frequency. Default is set to 75 Hz.
    modality : str
        The type of signal provided. Can be ``'ppg'`` (pulse oximeter) or
        ``'ecg'`` (electrocardiography). The peak detection algorithm used
        depend on the type of signal provided.
    ecg_method : str
        Peak detection algorithm used by the
        :py:func:`systole.detection.ecg_peaks` function. Can be one of the
        following: `'hamilton'`, `'christov'`, `'engelse-zeelenberg'`,
        `'pan-tompkins'`, `'wavelet-transform'`, `'moving-average'`. The
        default is `'hamilton'`.
    show_heart_rate : bool
        If `True`, show the instnataneous heart rate below the raw signal.
        Defaults to `False`.
    slider : bool
        If `True`, will add a slider to select the time window to plot
        (requires bokeh backend).
    ax : :class:`matplotlib.axes.Axes` or None
        Where to draw the plot. Default is *None* (create a new figure). Only
        applies when `backend="matplotlib"`.
    figsize : tuple, int or None
        Figure size. Default is `(13, 5)` for matplotlib backend, and the
        height is `300` when using bokeh backend.
    backend: str
        Select plotting backend {"matplotlib", "bokeh"}. Defaults to
        "matplotlib".
    **kwargs : keyword arguments
        Additional arguments will be passed to
        `:py:func:systole.detection.oxi_peaks()` or
        `:py:func:systole.detection.ecg_peaks()`, depending on the type
        of data.

    Returns
    -------
    plot : :class:`matplotlib.axes.Axes` or :class:`bokeh.plotting.figure.Figure`
        The matplotlib axes, or the boken figure containing the plot.

    See also
    --------
    plot_events, plot_rr

    Examples
    --------
    Plotting PPG recording.

    .. plot::

       >>> from systole import import_ppg
       >>> from systole.plots import plot_raw
       >>> # Import PPG recording as pandas data frame
       >>> ppg = import_ppg()
       >>> # Only use the first 60 seconds for demonstration
       >>> ppg = ppg[ppg.time<60]
       >>> plot_raw(ppg)

    Plotting ECG recording.

    .. plot::

       >>> from systole import import_dataset1
       >>> from systole.plots import plot_raw
       >>> # Import PPG recording as pandas data frame
       >>> ecg = import_dataset1(modalities=['ECG'])
       >>> # Only use the first 60 seconds for demonstration
       >>> ecg = ecg[ecg.time<60]
       >>> plot_raw(ecg, type='ecg', sfreq=1000, ecg_method='pan-tompkins')
    """
    if figsize is None:
        if backend == "matplotlib":
            figsize = (13, 5)
        elif backend == "bokeh":
            figsize = 300

    if isinstance(signal, pd.DataFrame):
        # Find peaks - Remove learning phase
        if modality == "ppg":
            signal, peaks = oxi_peaks(signal.ppg, noise_removal=False, **kwargs)
        elif modality == "ecg":
            signal, peaks = ecg_peaks(
                signal.ecg, method=ecg_method, find_local=True, **kwargs
            )
    else:
        if modality == "ppg":
            signal, peaks = oxi_peaks(
                signal, noise_removal=False, sfreq=sfreq, **kwargs
            )
        elif modality == "ecg":
            signal, peaks = ecg_peaks(
                signal, method=ecg_method, sfreq=sfreq, find_local=True, **kwargs
            )

    time = pd.to_datetime(np.arange(0, len(signal)), unit="ms", origin="unix")

    plot_raw_args = {
        "time": time,
        "signal": signal,
        "peaks": peaks,
        "modality": modality,
        "show_heart_rate": show_heart_rate,
        "ax": ax,
        "figsize": figsize,
        "slider": slider,
    }

    plotting_function = get_plotting_function("plot_raw", "plot_raw", backend)
    plot = plotting_function(**plot_raw_args)

    return plot
