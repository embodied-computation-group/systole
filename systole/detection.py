# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import pandas as pd
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
        rollingNoise = int(new_sfreq*.05)  # 0.5 second window
        x = pd.DataFrame(
            {'signal': x}).rolling(rollingNoise,
                                   center=True).mean().signal.to_numpy()
    if peak_enhancement is True:
        # Square signal (peak enhancement)
        x = x ** 2

    # Compute moving average and standard deviation
    signal = pd.DataFrame({'signal': x})
    mean_signal = signal.rolling(int(new_sfreq*0.75),
                                 center=True).mean().signal.to_numpy()
    std_signal = signal.rolling(int(new_sfreq*0.75),
                                center=True).std().signal.to_numpy()

    # Substract moving average + standard deviation
    x -= (mean_signal + std_signal)

    # Find positive peaks
    peaks_idx = find_peaks(x, height=0, distance=int(new_sfreq*0.2))[0]

    # Create boolean vector
    peaks = np.zeros(len(x), dtype=bool)
    peaks[peaks_idx] = 1

    return resampled_signal, peaks


def rr_artefacts(rr, c1=0.13, c2=0.17, alpha=5.2):
    """Artefacts detection from RR time series using the subspaces approach
    proposed by Lipponen & Tarvainen (2019).

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
    alpha : float
        Scaling factor used to normalize the RR intervals first deviation.

    Returns
    -------
    artefacts : dictionnary
        Dictionnary storing the parameters of RR artefacts rejection. All the
        vectors outputed have the same length as the provided RR time serie:

        * subspace1 : 1d array-like
            The first dimension. First derivative of R-R interval time serie.
        * subspace2 : 1d array-like
            The second dimension (1st plot).
        * subspace3 : 1d array-like
            The third dimension (2nd plot).
        * mRR : 1d array-like
            The mRR time serie.
        * ectopic : 1d array-like
            Boolean array indexing probable ectopic beats.
        * long : 1d array-like
            Boolean array indexing long RR intervals.
        * short : 1d array-like
            Boolean array indexing short RR intervals.
        * missed : 1d array-like
            Boolean array indexing missed RR intervals.
        * extra : 1d array-like
            Boolean array indexing extra RR intervals.
        * threshold1 : 1d array-like
            Threshold 1.
        * threshold2 : 1d array-like
            Threshold 2.

    Notes
    -----
    This function will use the method proposed by Lipponen & Tarvainen (2019)
    to detect ectopic beats, long, shorts, missed and extra RR intervals.

    References
    ----------
    .. [1] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for
        heart rate variability time series artefact correction using novel
        beat classification. Journal of Medical Engineering & Technology,
        43(3), 173–181. https://doi.org/10.1080/03091902.2019.1640306
    """
    if isinstance(rr, list):
        rr = np.array(rr)

    ###########
    # Detection
    ###########

    # Subspace 1 (dRRs time serie)
    dRR = np.diff(rr, prepend=0)
    dRR[0] = dRR[1:].mean()  # Set first item to a realistic value

    dRR_df = pd.DataFrame({'signal': np.abs(dRR)})
    q1 = dRR_df.rolling(
        91, center=True, min_periods=1).quantile(.25).signal.to_numpy()
    q3 = dRR_df.rolling(
        91, center=True, min_periods=1).quantile(.75).signal.to_numpy()

    th1 = alpha * ((q3 - q1) / 2)
    dRR = dRR / th1
    s11 = dRR

    # mRRs time serie
    medRR = pd.DataFrame({'signal': rr}).rolling(
                    11, center=True, min_periods=1).median().signal.to_numpy()
    mRR = rr - medRR
    mRR[mRR < 0] = 2 * mRR[mRR < 0]

    mRR_df = pd.DataFrame({'signal': np.abs(mRR)})
    q1 = mRR_df.rolling(
        91, center=True, min_periods=1).quantile(.25).signal.to_numpy()
    q3 = mRR_df.rolling(
        91, center=True, min_periods=1).quantile(.75).signal.to_numpy()

    th2 = alpha * ((q3 - q1) / 2)
    mRR /= th2

    # Subspace 2
    ma = np.hstack(
        [0, [np.max([dRR[i-1], dRR[i+1]]) for i in range(1, len(dRR)-1)], 0])
    mi = np.hstack(
        [0, [np.min([dRR[i-1], dRR[i+1]]) for i in range(1, len(dRR)-1)], 0])
    s12 = ma
    s12[dRR < 0] = mi[dRR < 0]

    # Subspace 3
    ma = np.hstack(
        [[np.max([dRR[i+1], dRR[i+2]]) for i in range(0, len(dRR)-2)], 0, 0])
    mi = np.hstack(
        [[np.min([dRR[i+1], dRR[i+2]]) for i in range(0, len(dRR)-2)], 0, 0])
    s22 = ma
    s22[dRR >= 0] = mi[dRR >= 0]

    ##########
    # Decision
    ##########

    # Find ectobeats
    cond1 = (s11 > 1) & (s12 < (-c1 * s11-c2))
    cond2 = (s11 < -1) & (s12 > (-c1 * s11+c2))
    ectopic = cond1 | cond2
    # No ectopic detection and correction at time serie edges
    ectopic[-2:] = False
    ectopic[:2] = False

    # Find long or shorts
    longBeats = \
        ((s11 > 1) & (s22 < -1)) | ((np.abs(mRR) > 3) & (rr > np.median(rr)))
    shortBeats = \
        ((s11 < -1) & (s22 > 1)) | ((np.abs(mRR) > 3) & (rr <= np.median(rr)))

    # Test if next interval is also outlier
    for cond in [longBeats, shortBeats]:
        for i in range(len(cond)-2):
            if cond[i] is True:
                if np.abs(s11[i+1]) < np.abs(s11[i+2]):
                    cond[i+1] = True

    # Ectopic beats are not considered as short or long
    shortBeats[ectopic] = False
    longBeats[ectopic] = False

    # Missed vector
    missed = np.abs((rr/2) - medRR) < th2
    missed = missed & longBeats
    longBeats[missed] = False  # Missed beats are not considered as long

    # Etra vector
    extra = np.abs(rr + np.append(rr[1:], 0) - medRR) < th2
    extra = extra & shortBeats
    shortBeats[extra] = False  # Extra beats are not considered as short

    # No short or long intervals at time serie edges
    shortBeats[0], shortBeats[-1] = False, False
    longBeats[0], longBeats[-1] = False, False

    artefacts = {'subspace1': s11, 'subspace2': s12, 'subspace3': s22,
                 'mRR': mRR, 'ectopic': ectopic, 'long': longBeats,
                 'short': shortBeats, 'missed': missed, 'extra': extra,
                 'threshold1': th1, 'threshold2': th2}

    return artefacts


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

    .. Warning:: If clipping artefact is found at the edge of the signal, this
        function will decrement the first/last value to allow interpolation,
        which can lead to incorrect estimation.

    References
    ----------
    .. [#] https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/
    """
    if isinstance(signal, list):
        signal = np.array(signal)

    # Security check for clipping at signal edge
    if signal[0] == threshold:
        signal[0] = threshold-1
    if signal[-1] == threshold:
        signal[-1] = threshold-1

    time = np.arange(0, len(signal))

    # Interpolate
    f = interp1d(time[np.where(signal != 255)[0]],
                 signal[np.where(signal != 255)[0]],
                 kind='cubic')

    # Use the peaks vector as time input
    clean_signal = f(time)

    return clean_signal
