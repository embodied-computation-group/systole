from scipy.signal import butter, lfilter
import numpy as np
from scipy import signal


def welch(x, sfreq, low=1, window=None):
    """Extract the PSD from R-R intervals.

    Parameters
    ----------
    x : list | numpy array
        Length of R-R intervals (in miliseconds).

    Returns
    -------
    psd : Numpy array
        The PSD.
    psd : Numpy array
        The PSD.

    Notes
    -----
    Because R peaks are unevenly spaced, the signal is first interpolated and
    resampled to 5 Hz.

    References
    ----------
    [1] https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html

    [2] https://en.wikipedia.org/wiki/Welch%27s_method

    [3] https://raphaelvallat.com/bandpower.html
    """
    x = np.asarray(x)

    # Define window length
    if window is not None:
        nperseg = window * sfreq
    else:
        nperseg = (2 / low) * sfreq

    # Compute Power Spectral Density
    freq, psd = signal.welch(x=x, fs=sfreq, nperseg=nperseg)

    return freq, psd
