# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def rolling_average_resp(
    signal: np.ndarray,
    sfreq: int,
    win: float = 0.025,
    kind: str = "peaks-onsets",
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """A simple peaks and/or onsets detection algorithm for respiratory signal inspired
    by [1]_.

    Parameters
    ----------
    signal :
        The respiratory signal. Peaks are considered to represent end of inspiration,
        trough represent end of expiration.
    sfreq :
        The sampling frequency.
    win :
        Window size (in seconds). Default is set to 25ms, following recommandation
        from [1]_.
    kind :
        What kind of detection to perform. Peak detection (`"peaks"`), trough detection
        (`"onsets"`) or both (`"peaks-onsets"`, default).

    Returns
    -------
    peaks_idx | trough_idx | (peaks_idx, trough_idx) :
        Indexes of peaks and / or onsets in the respiratory signal.

    References
    ----------
    .. [1] Torben Noto, Guangyu Zhou, Stephan Schuele, Jessica Templer, Christina
       Zelano,Automated analysis of breathing waveforms using BreathMetrics: a
       respiratory signal processing toolbox, Chemical Senses, Volume 43, Issue 8,
       October 2018, Pages 583-597, https://doi.org/10.1093/chemse/bjy045

    """
    # Soothing using rolling mean
    signal = (
        pd.DataFrame({"signal": signal})
        .rolling(int(sfreq * win), center=True)
        .mean()
        .fillna(method="bfill")
        .fillna(method="ffill")
        .signal.to_numpy()
    )

    # Normalize (z-score) the respiration signal
    signal = (signal - signal.mean()) / signal.std()  # type: ignore

    # Peak enhancement
    signal = signal**3

    # Find peaks and trough in preprocessed signal
    if "peaks" in kind:
        peaks_idx = find_peaks(signal, height=0, distance=int(2 * sfreq))[0]

    if "onsets" in kind:
        onsets_idx = find_peaks(-signal, height=0, distance=int(2 * sfreq))[0]

    if kind == "peaks":
        return peaks_idx
    elif kind == "trough":
        return onsets_idx
    else:
        return peaks_idx, onsets_idx
