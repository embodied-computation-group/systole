# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List

import numpy as np
from numba import jit
from scipy.signal import butter, lfilter

from systole.detectors.pan_tompkins import MWA_cumulative


def moving_average(signal: np.ndarray, sfreq: int) -> np.ndarray:
    """R peaks detection using two moving average.

    Parameters
    ----------
    signal :
        The unfiltered ECG signal.
    sfreq :
        The sampling frequency.

    Returns
    -------
    peaks :
        The indexs of the ECG peaks.

    References
    ----------
    This function is directly adapted from py-ecg-detectors
    (https://github.com/berndporr/py-ecg-detectors). This version of the code has been
    optimized using Numba for better performances.

    [1].. Elgendi, Mohamed & Jonkman, Mirjam & De Boer, Friso. (2010). Frequency Bands
        Effects on QRS Detection. The 3rd International Conference on Bio-inspired
        Systems and Signal Processing (BIOSIGNALS2010). 428-431.
    """

    f1 = 8 / sfreq
    f2 = 20 / sfreq

    b, a = butter(2, [f1 * 2, f2 * 2], btype="bandpass")

    filtered_ecg = lfilter(b, a, signal)

    window1 = int(0.12 * sfreq)
    mwa_qrs = MWA_cumulative(np.abs(filtered_ecg), window1)

    window2 = int(0.6 * sfreq)
    mwa_beat = MWA_cumulative(np.abs(filtered_ecg), window2)

    peaks = numba_one(signal, mwa_qrs, mwa_beat, sfreq, filtered_ecg)

    return peaks


@jit(nopython=True)
def numba_one(
    signal: np.ndarray, mwa_qrs, mwa_beat, sfreq: int, filtered_ecg: np.ndarray
) -> np.ndarray:
    blocks = np.zeros(len(signal))
    block_height = np.max(filtered_ecg)

    for i in range(len(mwa_qrs)):
        if mwa_qrs[i] > mwa_beat[i]:
            blocks[i] = block_height
        else:
            blocks[i] = 0

    QRS: List = []

    for i in range(1, len(blocks)):
        if blocks[i - 1] == 0 and blocks[i] == block_height:
            start = i

        elif blocks[i - 1] == block_height and blocks[i] == 0:
            end = i - 1

            if end - start > int(0.08 * sfreq):
                detection = np.argmax(filtered_ecg[start : end + 1]) + start
                if QRS:
                    if detection - QRS[-1] > int(0.3 * sfreq):
                        QRS.append(detection)
                else:
                    QRS.append(detection)

    return np.array(QRS)
