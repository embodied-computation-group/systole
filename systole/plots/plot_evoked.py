# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Optional, Tuple, Union, overload

import numpy as np
from bokeh.plotting.figure import Figure
from matplotlib.axes import Axes

from systole.detection import ecg_peaks, oxi_peaks
from systole.plots.utils import get_plotting_function
from systole.utils import heart_rate, to_epochs


@overload
def plot_evoked(
    epochs: None,
    signal: np.ndarray,
    triggers: Union[np.ndarray, List],
    rr: None,
    outliers: Optional[np.ndarray],
) -> Union[Figure, Axes]:
    ...


@overload
def plot_evoked(
    epochs: None,
    signal: None,
    triggers: Union[np.ndarray, List],
    rr: np.ndarray,
    outliers: Optional[np.ndarray],
) -> Union[Figure, Axes]:
    ...


@overload
def plot_evoked(
    epochs: np.ndarray,
    signal: None,
    triggers: None,
    rr: None,
    outliers: Optional[np.ndarray],
) -> Union[Figure, Axes]:
    ...


def plot_evoked(
    epochs=None,
    signal=None,
    triggers=None,
    rr=None,
    outliers=None,
    modality: str = "ppg",
    tmin: float = -1,
    tmax: float = 10,
    sfreq: int = 75,
    sfreq_out: int = 10,
    color: str = "#4c72b0",
    label: Optional[str] = None,
    unit: str = "bpm",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 5),
    backend: str = "matplotlib",
    **kwargs
) -> Union[Figure, Axes]:
    """Plot events occurence across recording.

    Parameters
    ----------
    epochs : np.array
        A 2d (trial * time) numpy array containing the time series
        of the epoched signal.
    signal : np.array
        A 1d numpy array containing the physiological signal (can be PPG or
        ECG). The modality of the signal is parametrized using the `modality`
        parameter.
    triggers : np.array
        A 1d numpy array that has the same length than `signal` and contain
        the triggers. A list of triggers array can also be provided for each
        condition.
    rr : np.array
        Interval time-series (R-R, beat-to-beat...), in miliseconds.
    outliers : np.array
        A boolean array indexing trial outliers with `True`. The outliers will
        be plotted using a different color but will not be part of the average
        computation.
    tmin, tmax : float
        Start and end time of the epochs in seconds, relative to the
        time-locked event. Defaults to -1 and 10, respectively.
    sfreq : int
        The sampling frequency of the input signal and triggers.
    sfreq_out : int
        The sampling frequency of the evoked time series.
    figsize : tuple
        Figure size. Default is `(13, 5)`.
    color : str
        The lines color.
    label : str
        The condition label.
    unit : str
        The heart rate unit in use. Can be `'rr'` (R-R intervals, in ms)
        or `'bpm'` (beats per minutes). Default is `'bpm'`.
    ax : :class:`matplotlib.axes.Axes` or None
        Where to draw the plot. Default is *None* (create a new figure).
    figsize : tuple
        The figure size.
    kwargs: key, value mappings
        Other keyword arguments are passed down to
        py:`func:seaborn.lineplot()`.

    Returns
    -------
    plot : :class:`matplotlib.axes.Axes` or :class:`bokeh.plotting.figure.Figure`
        The matplotlib axes, or the boken figure containing the plot.
    """
    if figsize is None:
        if backend == "matplotlib":
            figsize = (13, 5)
        elif backend == "bokeh":
            figsize = 300

    ##########
    # Epoching
    ##########
    if epochs is None:

        if isinstance(signal, np.ndarray) & isinstance(triggers, np.ndarray):

            # Extract instantaneous heart rate
            if modality == "ppg":
                _, peaks = oxi_peaks(signal, sfreq=sfreq)
            elif modality == "ecg":
                _, peaks = ecg_peaks(signal, sfreq=sfreq)

            rr, _ = heart_rate(peaks, kind="cubic", unit="bpm")

        # Epoch intantaneous heart rate
        # and downsample to 10 Hz to save memory
        epochs = to_epochs(rr, triggers, tmin=tmin, tmax=tmax)[
            :, :: int(1000 / sfreq_out)
        ]

    plot_evoked_args = {
        "epochs": epochs,
        "tmin": tmin,
        "tmax": tmax,
        "sfreq": sfreq,
        "color": color,
        "ax": ax,
        "figsize": figsize,
        "label": label,
    }

    plotting_function = get_plotting_function("plot_evoked", "plot_evoked", backend)
    plot = plotting_function(**plot_evoked_args)

    return plot
