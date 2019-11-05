# QRS detection
import numpy as np
from scipy.signal import find_peaks


def oxi_peaks(x, sfreq=75, win=1):
    """Peak detection on Oximetric data.

    Parameters
    ----------
    x : list or Numpy array
        The oxi signal.
    sfreq = int
        The sampling frequency. Default is set to 75 Hz.
    win : int
        Window size (in seconds) used to compute the threshold.

    Retruns
    -------
    peaks : Numpy array
        Numpy array containing R peak timing, in sfreq.

    Notes
    -----
    Signal squaring and detection of peaks using threshold set by the moving
    averagte + stadard deviation.
    """
    if isinstance(x, list):
        x = np.asarray(x)

    # Square signal
    x = x ** 2

    # Compute moving mean + standard deviation
    win = int(win * sfreq)
    rm = []
    for i in range(len(x)):

        if i < win/2:
            rm.append(np.mean(x[:win]) + np.std((x[:win])))

        elif (i >= win/2) & (i < len(x - win)):
            rm.append(np.mean(x[i-int(win/2):i+int(win/2)]) +
                      np.std(x[i-int(win/2):i+int(win/2)]))
        else:
            rm.append(np.mean(x[-win:]) + np.std(x[-win:]))

    # Substract moving mean + standard deviation
    x = x - rm

    # Find positive peaks
    peaks = find_peaks(x, height=0)[0]

    return peaks
