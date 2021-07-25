# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import pywt
from scipy.signal import butter, lfilter

from systole.detectors.pan_tompkins import panPeakDetect


def wavelet_transform(signal: np.ndarray, sfreq: int) -> np.ndarray:
    """R peaks detection using wavelet transform.

    Parameters
    ----------
    signal : np.ndarray
        The unfiltered ECG signal.
    sfreq : int
        The sampling frequency.

    Returns
    -------
    peaks : np.ndarray
        The indexs of the ECG peaks.

    Notes
    -----
    Uses the Pan and Tompkins thresholding.

    References
    ----------
    This function is directly adapted from py-ecg-detectors
    (https://github.com/berndporr/py-ecg-detectors). This version of the code has been
    optimized using Numba for better performances.

    [1].. Stationary Wavelet Transform based on Vignesh Kalidas and Lakshman Tamil.
        Real-time QRS detector using Stationary Wavelet Transform  for Automated ECG
        Analysis. In: 2017 IEEE 17th International Conference on Bioinformatics and
        Bioengineering (BIBE).
    """
    swt_level = 3
    padding = -1
    for i in range(1000):
        if (len(signal) + i) % 2 ** swt_level == 0:
            padding = i
            break

    if padding > 0:
        signal = np.pad(signal, (0, padding), "edge")
    elif padding == -1:
        print("Padding greater than 1000 required\n")

    swt_ecg = pywt.swt(signal, "db3", level=swt_level)
    swt_ecg = np.array(swt_ecg)[0, 1, :]

    squared = swt_ecg * swt_ecg

    f1 = 0.01 / sfreq
    f2 = 10 / sfreq

    b, a = butter(3, [f1 * 2, f2 * 2], btype="bandpass")
    filtered_squared = lfilter(b, a, squared)

    peaks = panPeakDetect(filtered_squared, sfreq)

    return peaks
