# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List

import numpy as np
from numba import jit
from scipy.signal import butter, lfilter


def pan_tompkins(
    signal: np.ndarray, sfreq: int, moving_average: str = "cumulative"
) -> np.ndarray:
    """
    Parameters
    ----------
    signal :
        The unfiltered ECG signal.
    sfreq :
        The sampling frequency.
    moving_average :
        The moving average function to use.

    Returns
    -------
    peaks_idx :
        Indexes of R peaks in the input signal.

    References
    ----------
    This function is directly adapted from py-ecg-detectors
    (https://github.com/berndporr/py-ecg-detectors). This version of the code has been
    optimized using Numba for better performances.

    [1].. Jiapu Pan and Willis J. Tompkins. A Real-Time QRS Detection Algorithm.
    In: IEEE Transactions on Biomedical Engineering BME-32.3 (1985), pp. 230–236.
    """
    signal = np.asarray(signal, dtype=float)

    f1 = 5 / sfreq
    f2 = 15 / sfreq

    b, a = butter(1, [f1 * 2, f2 * 2], btype="bandpass")

    filtered_ecg = lfilter(b, a, signal)

    diff = np.diff(filtered_ecg)

    squared = diff * diff

    N = int(0.12 * sfreq)
    ma = {"cumulative": MWA_cumulative}  # Dict of moving average methods
    mwa = ma[moving_average](squared, N)
    mwa[: int(0.2 * sfreq)] = 0

    peaks = panPeakDetect(mwa, sfreq)

    return np.array(peaks, dtype=int)


@jit(nopython=True)
def MWA_cumulative(input_array: np.ndarray, window_size: int) -> np.ndarray:
    """Cumulative moving average method"""

    ret = np.cumsum(input_array)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    ret[: window_size - 1] /= np.arange(1, window_size)
    ret[window_size - 1 :] = ret[window_size - 1 :] / window_size

    return ret


@jit(nopython=True)
def panPeakDetect(detection: np.ndarray, sfreq: int) -> List:
    """Pan-Tompkins detection algorithm.

    Parameters
    ----------
    detection :
        Vector of detected peaks.
    sfreq :
        The sampling frequency.

    Returns
    -------
    signal_peaks :
        The indexs of the ECG peaks.

    """

    min_distance = int(0.25 * sfreq)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    peaks = []

    for i in range(len(detection)):
        if i > 0 and i < len(detection) - 1:
            if detection[i - 1] < detection[i] and detection[i + 1] < detection[i]:
                peak = i
                peaks.append(i)

                if (
                    detection[peak] > threshold_I1
                    and (peak - signal_peaks[-1]) > 0.3 * sfreq
                ):
                    signal_peaks.append(peak)
                    indexes.append(index)
                    SPKI = 0.125 * detection[signal_peaks[-1]] + 0.875 * SPKI
                    if RR_missed != 0:
                        if signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                            missed_section_peaks = peaks[indexes[-2] + 1 : indexes[-1]]
                            missed_section_peaks2 = []
                            for missed_peak in missed_section_peaks:
                                if (
                                    missed_peak - signal_peaks[-2] > min_distance
                                    and signal_peaks[-1] - missed_peak > min_distance
                                    and detection[missed_peak] > threshold_I2
                                ):
                                    missed_section_peaks2.append(missed_peak)

                            if len(missed_section_peaks2) > 0:
                                missed_peak = missed_section_peaks2[
                                    np.argmax(
                                        detection[np.array(missed_section_peaks2)]
                                    )
                                ]
                                missed_peaks.append(missed_peak)
                                signal_peaks.append(signal_peaks[-1])
                                signal_peaks[-2] = missed_peak

                else:
                    noise_peaks.append(peak)
                    NPKI = 0.125 * detection[noise_peaks[-1]] + 0.875 * NPKI

                threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
                threshold_I2 = 0.5 * threshold_I1

                if len(signal_peaks) > 8:
                    RR = np.array(signal_peaks[-9:])
                    RR = RR[1:] - RR[:-1]
                    RR_ave = int(np.mean(RR))
                    RR_missed = int(1.66 * RR_ave)

                index = index + 1

    signal_peaks.pop(0)

    return signal_peaks
