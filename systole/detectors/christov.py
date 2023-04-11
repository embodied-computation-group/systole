# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Tuple, Union

import numpy as np
from numba import jit
from scipy.signal import lfilter


def christov(signal: np.ndarray, sfreq: int) -> np.ndarray:
    """R peaks detection using Christov's method.

    Parameters
    ----------
    signal :
        The unfiltered ECG signal.
    sfreq :
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

    [1].. Ivaylo I. Christov, Real time electrocardiogram QRS detection using
        combined adaptive threshold, BioMedical Engineering OnLine 2004, vol. 3:28,
        2004.
    """
    b, total_taps = numba_one(sfreq)

    MA1 = lfilter(b, [1], signal)

    b, total_taps = numba_two(sfreq, total_taps)

    MA2 = lfilter(b, [1], MA1)

    b, Y, total_taps = numba_three(sfreq, total_taps, MA2)

    MA3 = lfilter(b, [1], Y)

    peaks = numba_four(MA3, sfreq, total_taps)

    return peaks


@jit(nopython=True)
def numba_one(sfreq: int) -> Tuple:
    total_taps = 0

    b = np.ones(int(0.02 * sfreq))
    b = b / int(0.02 * sfreq)
    total_taps += len(b)

    return b, total_taps


@jit(nopython=True)
def numba_two(sfreq: int, total_taps) -> Tuple:
    b = np.ones(int(0.028 * sfreq))
    b = b / int(0.028 * sfreq)
    total_taps += len(b)

    return b, total_taps


@jit(nopython=True)
def numba_three(sfreq: int, total_taps, MA2) -> Tuple:
    Y = []
    for i in range(1, len(MA2) - 1):
        diff = abs(MA2[i + 1] - MA2[i - 1])
        Y.append(diff)

    b = np.ones(int(0.040 * sfreq))
    b = b / int(0.040 * sfreq)
    total_taps += len(b)

    return b, Y, total_taps


@jit(nopython=True)
def numba_four(MA3, sfreq: int, total_taps) -> np.ndarray:
    MA3[0:total_taps] = 0

    ms50 = int(0.05 * sfreq)
    ms200 = int(0.2 * sfreq)
    ms1200 = int(1.2 * sfreq)
    ms350 = int(0.35 * sfreq)

    M = 0
    newM5 = 0.0
    M_list = []
    MM: List[Union[int, float]] = []
    M_slope = np.linspace(1.0, 0.6, ms1200 - ms200)
    F = 0
    F_list = []
    R = 0
    RR = []
    Rm = 0
    R_list = []

    MFR = 0
    MFR_list = []

    QRS: List = []

    for i in range(len(MA3)):
        # M
        if i < 5 * sfreq:
            M = 0.6 * np.max(MA3[: i + 1])
            MM.append(M)
            if len(MM) > 5:
                MM.pop(0)

        elif QRS and i < QRS[-1] + ms200:
            newM5 = 0.6 * np.max(MA3[QRS[-1] : i])
            if newM5 > 1.5 * MM[-1]:
                newM5 = 1.1 * MM[-1]

        elif QRS and i == QRS[-1] + ms200:
            if newM5 == 0:
                newM5 = MM[-1]
            MM.append(newM5)
            if len(MM) > 5:
                MM.pop(0)
            M = np.mean(np.asarray(MM))

        elif QRS and i > QRS[-1] + ms200 and i < QRS[-1] + ms1200:
            M = np.mean(np.asarray(MM)) * M_slope[i - (QRS[-1] + ms200)]

        elif QRS and i > QRS[-1] + ms1200:
            M = 0.6 * np.mean(np.asarray(MM))

        # F
        if i > ms350:
            F_section = MA3[i - ms350 : i]
            max_latest = np.max(F_section[-ms50:])
            max_earliest = np.max(F_section[:ms50])
            F = F + ((max_latest - max_earliest) / 150.0)

        # R
        if QRS and i < QRS[-1] + int((2.0 / 3.0 * Rm)):
            R = 0

        elif QRS and i > QRS[-1] + int((2.0 / 3.0 * Rm)) and i < QRS[-1] + Rm:
            dec = (M - np.mean(np.asarray(MM))) / 1.4
            R = 0 + dec

        MFR = M + F + R
        M_list.append(M)
        F_list.append(F)
        R_list.append(R)
        MFR_list.append(MFR)

        if not QRS and MA3[i] > MFR:
            QRS.append(i)

        elif QRS and i > QRS[-1] + ms200 and MA3[i] > MFR:
            QRS.append(i)
            if len(QRS) > 2:
                RR.append(QRS[-1] - QRS[-2])
                if len(RR) > 5:
                    RR.pop(0)
                Rm = int(np.mean(np.asarray(RR)))

    QRS.pop(0)

    return np.asarray(QRS)
