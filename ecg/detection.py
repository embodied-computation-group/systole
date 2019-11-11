# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
from scipy.signal import find_peaks
from scipy import interpolate
from ecg.utils import moving_function


def oxi_peaks(x, sfreq=75, win=1, new_sfreq=200, resample=True):
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
    f = interpolate.interp1d(np.arange(0, len(x)/sfreq, 1/sfreq),
                             x,
                             fill_value="extrapolate")
    time = np.arange(0, len(x)/sfreq, 1/new_sfreq)
    x = f(time)

    # Moving average (high frequency noise + clipping)
    x = moving_function(x, win=0.2, sfreq=new_sfreq, function=np.mean)

    # Square signal
    x = x ** 2

    # Compute moving average + standard deviation
    mean_signal = moving_function(x, win=0.75, sfreq=new_sfreq,
                                  function=np.mean)
    std_signal = moving_function(x, win=0.75, sfreq=new_sfreq,
                                 function=np.std)

    # Substract moving mean + standard deviation
    x -= (mean_signal + std_signal)

    # Find positive peaks
    peaks = find_peaks(x, height=0)[0]

    return peaks
