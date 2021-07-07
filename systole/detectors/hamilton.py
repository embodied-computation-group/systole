# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
from numba import jit
from scipy.signal import butter, lfilter


def hamilton(signal: np.ndarray, sfreq: int) -> np.ndarray:
    """R peaks detection using Hamilton's method.

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

    References
    ----------
    This function is directly adapted from py-ecg-detectors
    (https://github.com/berndporr/py-ecg-detectors). This version of the code has been
    optimized using Numba for better performances.

    [1].. P.S. Hamilton, Open Source ECG Analysis Software Documentation, E.P.Limited,
        2002.
    """
    frequencies = numba_first(signal, sfreq)

    b, a = butter(1, frequencies, btype="bandpass")

    filtered_ecg = lfilter(b, a, signal)

    diff, a, b = numba_second(filtered_ecg, sfreq)

    ma = lfilter(b, a, diff)

    peaks = numba_third(ma, b, sfreq)

    return peaks


@jit(nopython=True)
def numba_first(signal, sfreq):
    signal = np.asarray(signal)
    f1 = 8 / sfreq
    f2 = 16 / sfreq
    return [f1 * 2, f2 * 2]


@jit(nopython=True)
def numba_second(filtered_ecg, sfreq):
    diff = np.abs(np.diff(filtered_ecg))
    b = np.ones(int(0.08 * sfreq))
    b = b / int(0.08 * sfreq)
    a = [1]
    return diff, a, b


@jit(nopython=True)
def numba_third(ma, b, sfreq):
    ma[0 : len(b) * 2] = 0

    n_pks = []
    n_pks_ave = 0.0
    s_pks = []
    s_pks_ave = 0.0
    QRS = [0]
    RR = []
    RR_ave = 0.0

    th = 0.0

    i = 0
    idx = []
    peaks = []

    for i in range(len(ma)):

        if i > 0 and i < len(ma) - 1:
            if ma[i - 1] < ma[i] and ma[i + 1] < ma[i]:
                peak = i
                peaks.append(i)

                if ma[peak] > th and (peak - QRS[-1]) > 0.3 * sfreq:
                    QRS.append(peak)
                    idx.append(i)
                    s_pks.append(ma[peak])
                    if len(n_pks) > 8:
                        s_pks.pop(0)
                    s_pks_ave = np.mean(np.asarray(s_pks))

                    if RR_ave != 0.0:
                        if QRS[-1] - QRS[-2] > 1.5 * RR_ave:
                            missed_peaks = peaks[idx[-2] + 1 : idx[-1]]
                            for missed_peak in missed_peaks:
                                if (
                                    missed_peak - peaks[idx[-2]] > int(0.360 * sfreq)
                                    and ma[missed_peak] > 0.5 * th
                                ):
                                    QRS.append(missed_peak)
                                    QRS.sort()
                                    break

                    if len(QRS) > 2:
                        RR.append(QRS[-1] - QRS[-2])
                        if len(RR) > 8:
                            RR.pop(0)
                        RR_ave = int(np.mean(np.asarray(RR)))

                else:
                    n_pks.append(ma[peak])
                    if len(n_pks) > 8:
                        n_pks.pop(0)
                    n_pks_ave = np.mean(np.asarray(n_pks))

                th = n_pks_ave + 0.45 * (s_pks_ave - n_pks_ave)

                i += 1

    QRS.pop(0)

    return QRS
