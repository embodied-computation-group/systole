# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import pandas as pd
from scipy.stats import iqr
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


def oxi_peaks(x, sfreq=75, win=1, new_sfreq=1000, clipping=True,
              noise_removal=True, peak_enhancement=True):
    """A simple peak finder for PPG signal.

    Parameters
    ----------
    x : list or 1d array-like
        The oxi signal.
    sfreq : int
        The sampling frequency. Default is set to 75 Hz.
    win : int
        Window size (in seconds) used to compute the threshold.
    new_sfreq : int
        If resample is *True*, the new sampling frequency.
    resample : bool
        If *True (defaults), will resample the signal at *new_sfreq*. Default
        value is 1000 Hz.

    Returns
    -------
    peaks : 1d array-like
        Numpy array containing R peak timing, in sfreq.
    resampled_signal : 1d array-like
        Signal resampled to the `new_sfreq` frequency.

    Notes
    -----
    This algorithm use a simple rolling average to detect peaks. The signal is
    first resampled and a rolling average is applyed to correct high frequency
    noise and clipping. The signal is then squared and detection of peaks is
    performed using threshold set by the moving averagte + stadard deviation.

    .. warning :: This function will resample the signal to 1000 Hz.

    References
    ----------
    Some of the processing steps were adapted from the HeartPy toolbox [1]:
    https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/index.html

    [1] : van Gent, P., Farah, H., van Nes, N. and van Arem, B., 2019.
    Analysing Noisy Driver Physiology Real-Time Using Off-the-Shelf Sensors:
    Heart Rate Analysis Software from the Taking the Fast Lane Project. Journal
    of Open Research Software, 7(1), p.32. DOI: http://doi.org/10.5334/jors.241
    """
    if isinstance(x, list):
        x = np.asarray(x)

    # Interpolate
    f = interp1d(np.arange(0, len(x)/sfreq, 1/sfreq),
                             x,
                             fill_value="extrapolate")
    time = np.arange(0, len(x)/sfreq, 1/new_sfreq)
    x = f(time)

    # Copy resampled signal for output
    resampled_signal = np.copy(x)

    # Remove clipping artefacts with cubic interpolation
    if clipping is True:
        x = interpolate_clipping(x)

    if noise_removal is True:
        # Moving average (high frequency noise + clipping)
        rollingNoise = int(new_sfreq*.1)  # 0.1 second window
        x = pd.DataFrame(
            {'signal': x}).rolling(rollingNoise,
                                   center=True).mean().signal.values
    if peak_enhancement is True:
        # Square signal (peak enhancement)
        x = x ** 2

    # Compute moving average and standard deviation
    signal = pd.DataFrame({'signal': x})
    mean_signal = signal.rolling(int(new_sfreq*0.75),
                                 center=True).mean().signal.values
    std_signal = signal.rolling(int(new_sfreq*0.75),
                                center=True).std().signal.values

    # Substract moving average + standard deviation
    x -= (mean_signal + std_signal)

    # Find positive peaks
    peaks_idx = find_peaks(x, height=0)[0]

    # Create boolean vector
    peaks = np.zeros(len(x), dtype=bool)
    peaks[peaks_idx] = 1

    if len(peaks) != len(x):
        raise ValueError('Inconsistent output lenght')

    return resampled_signal, peaks


def hrv_subspaces(x, alpha=5.2, window=45):
    """Plot hrv subspace as described by Lipponen & Tarvainen (2019).

    Parameters
    ----------
    x : 1d array-like
        Array of RR intervals.
    alpha : float
        Scaling factor used to normalize the RR intervals first deviation.
    window : int
        Size of the window used to compute the interquartil range and normalize
        the dRR serie.

    Returns
    -------
    subspace1 : 1d array-like
        The first dimension. First derivative of R-R interval time serie.
    subspace2 : 1d array-like
        The second dimension (1st plot).
    subspace3 : 1d array-like
        The third dimension (2nd plot).

    References
    ----------
    [1] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for
        heart rate variability time series artefact correction using novel
        beat classification. Journal of Medical Engineering & Technology,
        43(3), 173–181. https://doi.org/10.1080/03091902.2019.1640306
    """
    # Subspace 1 - dRRs time serie
    s11 = np.append(0, np.diff(x))

    th = []
    for i in range(len(s11)):
        mi, ma = i-45, i+45
        if mi < 0:
            mi = 0
        if ma > len(s11):
            ma = len(s11)
        th.append(alpha*iqr(np.abs(s11[mi:ma]))/2)
    s11 = s11/th

    # Subspace 2
    diff = np.array([np.append(s11[1], s11[:-1]), np.append(s11[1:], s11[-1])])
    ma = np.max(diff, 0)
    mi = np.min(diff, 0)
    s12 = []
    for i in range(len(s11)):
        if s11[i] <= 0:
            s12.append(mi[i])
        elif s11[i] > 0:
            s12.append(ma[i])

    # Subspace 3
    diff = np.array([s11[1:-1], s11[2:]])
    ma = np.max(diff, 0)
    mi = np.min(diff, 0)
    s22 = []
    for i in range(len(s11)-2):
        if s11[i] < 0:
            s22.append(ma[i])
        elif s11[i] >= 0:
            s22.append(mi[i])

    return np.asarray(s11), np.asarray(s12), np.append(np.asarray(s22), [0, 0])


def interpolate_clipping(signal, threshold=255):
    """Interoplate clipping segment.

    Parameters
    ----------
    signal : 1d array-like
        Noisy signal.
    threshold : int
        Threshold of clipping artefact.

    Returns
    -------
    clean_signal : 1d array-like
        Interpolated signal.

    Notes
    -----
    Correct signal segment reaching recording threshold (default is 255)
    using a cubic spline interpolation. Adapted from [#]_.

    References
    ----------
    .. [#] https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/
    """
    if isinstance(signal, list):
        signal = np.array(signal)

    # Security check for clipping at signal edge
    if signal[0]==threshold:
        signal[0] = threshold-1
    if signal[-1]==threshold:
        signal[-1] = threshold-1

    time = np.arange(0, len(signal))

    # Interpolate
    f = interp1d(time[np.where(signal != 255)[0]],
                 signal[np.where(signal != 255)[0]],
                 kind='cubic')

    # Use the peaks vector as time input
    clean_signal = f(time)

    return clean_signal


def rr_outliers(rr, c1=0.13, c2=0.17):
    """Find outliers in RR time series using subspaces decomposition.

    Parameters
    ----------
    rr : 1d array-like
        Array of RR intervals.
    c1 : float
        Fixed variable controling the slope of the threshold lines. Default is
        0.13.
    c2 : float
        Fixed variable controling the intersect of the threshold lines. Default
        is 0.17.

    Returns
    -------
    ectobeats : 1d array-like
        Boolean array indexing probable ectobeats.
    outliers : 1d array-like
        Boolean array indexing abberant shorts/long RR intervals.

    Notes
    -----
    This function will use the method proposed by Lipponen & Tarvainen [1]_ to
    find probable ectobeats and abberant long/shorts RR intervals.

    References
    ----------
    [1] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for
        heart rate variability time series artefact correction using novel
        beat classification. Journal of Medical Engineering & Technology,
        43(3), 173–181. https://doi.org/10.1080/03091902.2019.1640306
    """

    subspace1, subspace2, subspace3 = hrv_subspaces(rr)

    # Find ectobeats
    cond1 = (subspace1 > 1) & (subspace2 < (-c1 * subspace1-c2))
    cond2 = (subspace1 < -1) & (subspace2 > (-c1 * subspace1+c2))
    ectobeats = cond1 | cond2

    # Find long or shorts
    cond1 = (subspace1 > 1) & (subspace3 < -1)
    cond2 = (subspace1 < -1) & (subspace3 > 1)
    outliers = cond1 | cond2

    return ectobeats, outliers
