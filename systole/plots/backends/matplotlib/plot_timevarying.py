# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from mne.time_frequency import tfr_array_multitaper
from systole.utils import heart_rate


def plot_timevarying(
    rr: Union[List, np.ndarray],
    unit: str = "rr",
    kind: str = "cubic",
    sfreq: int = 1000,
    line: bool = True,
    points: bool = True,
    input_type: str = "peaks",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (13, 5),
) -> Axes:
    """Plot continuous or discontinuous RR intervals time series.

    Parameters
    ----------
    rr : np.ndarray or list
        Boolean vector of peaks detection or RR intervals.
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
        points
    input_type : str
        The type of input vector. Default is `"peaks"` (a boolean vector where
        `1` represents the occurrence of R waves or systolic peaks).
        Can also be `"rr_s"` or `"rr_ms"` for vectors of RR intervals, or
        interbeat intervals (IBI), expressed in seconds or milliseconds
        (respectively).
    ax : :class:`matplotlib.axes.Axes` or None
        Where to draw the plot. Default is *None* (create a new figure).
    figsize : tuple
        Figure size. Default is `(13, 5)`.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        The matplotlib axes containing the plot.

    See also
    --------
    plot_events, plot_subspaces, plot_events, plot_psd, plot_oximeter, plot_raw

    Examples
    --------

    .. plot::

        >>> from systole import import_rr
        >>> from systole.plotting import plot_rr
        >>> rr = import_rr().rr.values
        >>> plot_rr(rr=rr, input_type="rr_ms", unit="bpm",)
    """

    rr = np.array(rr)

    if (points is False) & (line is False):
        raise ValueError("point and line arguments cannot be False at the same time")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Extract instantaneous heart rate
    hr, time = heart_rate(rr, sfreq=1000, unit="rr", kind="cubic", input_type="rr_s")

    # Remove Nans
    hr = hr[1000:]

    # Downsample
    hr = hr

    freqs = np.arange(0.04, 0.4, 0.005)
    sfreq = 1000
    tfr = tfr_array_multitaper(
        hr[np.newaxis, np.newaxis, :], output="power", sfreq=sfreq, freqs=freqs
    )[0, 0]

    ax.imshow(
        tfr, aspect="auto", origin="lower", extent=[0, len(hr) / sfreq, 0.04, 0.4]
    )

    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
