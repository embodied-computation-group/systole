# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy import interpolate


def oxi_peaks(x, sfreq=75, win=1, new_sfreq=200, resample=False):
    """Detecting peaks on PPG signal.

    Parameters
    ----------
    x : list or Numpy array
        The oxi signal.
    sfreq = int
        The sampling frequency. Default is set to 75 Hz.
    win : int
        Window size (in seconds) used to compute the threshold.
    new_sfreq : int
        If `resample=True`, the new sampling frequency.
    resample : boolean
        If `True`, will resample the signal at `new_sfreq`.

    Retruns
    -------
    peaks : array
        Numpy array containing R peak timing, in sfreq.

    Notes
    -----
    This algorithm use a simple rolling average to detect peaks. The signal is
    first resampled and a rolling average is applyed to correct high frequency
    noise and clipping. The signal is then squared and detection of peaks is
    performed using threshold set by the moving averagte + stadard deviation.

    References
    ----------
    Some of the processing steps are adapted from the HeartPy toolbox:
    https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/index.html

    [1] : van Gent, P., Farah, H., van Nes, N. and van Arem, B., 2019.
    Analysing Noisy Driver Physiology Real-Time Using Off-the-Shelf Sensors:
    Heart Rate Analysis Software from the Taking the Fast Lane Project. Journal
    of Open Research Software, 7(1), p.32. DOI: http://doi.org/10.5334/jors.241
    """
    if isinstance(x, list):
        x = np.asarray(x)

    # Interpolate
    if resample is True:
        f = interpolate.interp1d(np.arange(0, len(x)/sfreq, 1/sfreq),
                                 x,
                                 fill_value="extrapolate")
        time = np.arange(0, len(x)/sfreq, 1/new_sfreq)
        x = f(time)
    else:
        new_sfreq = sfreq

    # Moving average (high frequency noise + clipping)
    x = pd.DataFrame({'signal': x}).rolling(int(sfreq/10)).mean().signal.values

    # Square signal
    x = x ** 2

    # Compute moving average + standard deviation
    signal = pd.DataFrame({'signal': x})

    mean_signal = signal.rolling(int(sfreq*0.75)).mean().signal.values
    std_signal = signal.rolling(int(sfreq*0.75)).std().signal.values

    # Substract moving mean + standard deviation
    x -= (mean_signal + std_signal)

    # Find positive peaks
    peaks_idx = find_peaks(x, height=0)[0]

    # Create boolean vector as Output
    peaks = np.zeros(len(x))
    peaks[peaks_idx] = 1

    if len(peaks) != len(x):
        raise ValueError('Inconsistent output lenght')

    return peaks
