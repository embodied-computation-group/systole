# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
from numba import jit
from scipy.signal import butter, lfilter


def engelse_zeelenberg(signal, sfreq):
    """R peaks detection using Engelse and Zeelenberg's method.

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

    [1].. Engelse, W.A.H., Zeelenberg, C. A single scan algorithm for QRS detection and
        feature extraction, IEEE Comp. in Cardiology, vol. 6, pp. 37-42, 1979 with
        modifications A. Lourenco, H. Silva, P. Leite, R. Lourenco and A. Fred,
        “Real Time Electrocardiogram Segmentation for Finger Based ECG Biometrics”,
        BIOSIGNALS 2012, pp. 49-54, 2012.
    """

    f1 = 48 / sfreq
    f2 = 52 / sfreq
    b, a = butter(4, [f1 * 2, f2 * 2], btype="bandstop")
    filtered_ecg = lfilter(b, a, signal)

    diff = numba_one(filtered_ecg)

    ci = [1, 4, 6, 4, 1]
    low_pass = lfilter(ci, 1, diff)

    peaks = numba_two(sfreq, low_pass, signal)

    return peaks


@jit(nopython=True)
def numba_one(filtered_ecg):
    diff = np.zeros(len(filtered_ecg))
    for i in range(4, len(diff)):
        diff[i] = filtered_ecg[i] - filtered_ecg[i - 4]
    return diff


@jit(nopython=True)
def numba_two(sfreq, low_pass, signal):

    low_pass[: int(0.2 * sfreq)] = 0

    ms200 = int(0.2 * sfreq)
    ms1200 = int(1.2 * sfreq)
    ms160 = int(0.16 * sfreq)
    neg_threshold = int(0.01 * sfreq)

    M = 0
    M_list = []
    neg_m = []
    MM = []
    M_slope = np.linspace(1.0, 0.6, ms1200 - ms200)

    QRS = []
    r_peaks = []

    counter = 0

    thi_list = []
    thi = False
    thf_list = []
    thf = False
    newM5 = False

    for i in range(len(low_pass)):

        # M
        if i < 5 * sfreq:
            M = 0.6 * np.max(low_pass[: i + 1])
            MM.append(M)
            if len(MM) > 5:
                MM.pop(0)

        elif QRS and i < QRS[-1] + ms200:

            newM5 = 0.6 * np.max(low_pass[QRS[-1] : i])

            if newM5 > 1.5 * MM[-1]:
                newM5 = 1.1 * MM[-1]

        elif newM5 and QRS and i == QRS[-1] + ms200:
            MM.append(newM5)
            if len(MM) > 5:
                MM.pop(0)
            M = np.mean(np.asarray(MM))

        elif QRS and i > QRS[-1] + ms200 and i < QRS[-1] + ms1200:

            M = np.mean(np.asarray(MM)) * M_slope[i - (QRS[-1] + ms200)]

        elif QRS and i > QRS[-1] + ms1200:
            M = 0.6 * np.mean(np.asarray(MM))

        M_list.append(M)
        neg_m.append(-M)

        if not QRS and low_pass[i] > M:
            QRS.append(i)
            thi_list.append(i)
            thi = True

        elif QRS and i > QRS[-1] + ms200 and low_pass[i] > M:
            QRS.append(i)
            thi_list.append(i)
            thi = True

        if thi and i < thi_list[-1] + ms160:
            if low_pass[i] < -M and low_pass[i - 1] > -M:
                # thf_list.append(i)
                thf = True

            if thf and low_pass[i] < -M:
                thf_list.append(i)
                counter += 1

            elif low_pass[i] > -M and thf:
                counter = 0
                thi = False
                thf = False

        elif thi and i > thi_list[-1] + ms160:
            counter = 0
            thi = False
            thf = False

        if counter > neg_threshold:
            unfiltered_section = signal[thi_list[-1] - int(0.01 * sfreq) : i]
            r_peaks.append(
                np.argmax(unfiltered_section) + thi_list[-1] - int(0.01 * sfreq)
            )
            counter = 0
            thi = False
            thf = False

    # removing the 1st detection as it 1st needs the QRS complex amplitude for the threshold
    r_peaks.pop(0)

    return r_peaks
