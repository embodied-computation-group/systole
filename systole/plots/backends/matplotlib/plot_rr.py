# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from systole.utils import heart_rate


def plot_rr(
    rr: np.ndarray,
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
    figsize : tuple
        Figure size. Default is `(13, 5)`.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        The matplotlib axes containing the plot.

    """

    ylabel = "R-R interval (ms)" if unit == "rr" else "Beats per minute (bpm)"

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if line is True:

        # Extract instantaneous heart rate
        hr, time = heart_rate(
            rr, sfreq=sfreq, unit=unit, kind=kind, input_type=input_type
        )

        # Convert to datetime format
        time = pd.to_datetime(time, unit="s", origin="unix")

        # Instantaneous Heart Rate - Lines
        ax.plot(time, hr, label="Instantaneous heart rat", linewidth=1, color="#4c72b0")

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

        ax.scatter(
            x=peaks_idx,
            y=ibi,
            marker="o",
            label="R-R intervals",
            s=20,
            color="white",
            edgecolors="DarkSlateGrey",
        )
    ax.set_title("Instantaneous heart rate")
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    return ax
