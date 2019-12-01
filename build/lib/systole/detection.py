# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy import interpolate
from adtk.detector import QuantileAD, GeneralizedESDTestAD
from adtk.data import validate_series


def oxi_peaks(x, sfreq=75, win=1, new_sfreq=1000):
    """A simple peak finder for PPG signal.

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
        If `True` (defaults), will resample the signal at `new_sfreq`. Default
        value is 750 Hz.

    Retruns
    -------
    peaks : boolean array
        Numpy array containing R peak timing, in sfreq.
    resampled_signal : array
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
    f = interpolate.interp1d(np.arange(0, len(x)/sfreq, 1/sfreq),
                             x,
                             fill_value="extrapolate")
    time = np.arange(0, len(x)/sfreq, 1/new_sfreq)
    x = f(time)

    # Copy resampled signal for output
    resampled_signal = np.copy(x)

    # Moving average (high frequency noise + clipping)
    rollingNoise = int(new_sfreq/10)  # 0.1 second window
    x = pd.DataFrame({'signal': x}).rolling(rollingNoise,
                                            center=True).mean().signal.values

    # Square signal (peak enhancement)
    x = x ** 2

    # Compute moving average and standard deviation
    signal = pd.DataFrame({'signal': x})
    mean_signal = signal.rolling(int(new_sfreq*0.75),
                                 center=True).mean().signal.values
    std_signal = signal.rolling(int(new_sfreq*0.75),
                                center=True).std().signal.values

    # Substract moving mean + standard deviation
    x -= (mean_signal + std_signal)

    # Find positive peaks
    peaks_idx = find_peaks(x, height=0)[0]

    # Create boolean vector
    peaks = np.zeros(len(x))
    peaks[peaks_idx] = 1

    if len(peaks) != len(x):
        raise ValueError('Inconsistent output lenght')

    return resampled_signal, peaks


def artifact_removal(peaks):
    """Artfact and outliers detection and removal.

    Parameters
    ----------
    peak : boolean array
        The peaks indexes to inspect. The sampling frequency must be 1000 Hz.

    Return
    ------
    clean_peaks : boolean array
        The cleaned peak indexes.

    Notes
    -----
    This function use the QuantileAD and the GeneralizedESDTestAD to detect
    outliers.
    """
    # Store into a Panda DataFrame
    rr = np.diff(np.where(peaks))[0]
    time = pd.to_datetime(np.cumsum(rr), unit='ms')
    df = pd.DataFrame({'rr': rr}, index=time)
    df = validate_series(df.rr)

    ##############################
    # Find high frequency outliers
    ##############################
    quantile_ad = QuantileAD(low=0.01)
    anomalies = quantile_ad.fit_detect(df)

    # Remove peaks
    rm_peaks = np.where(peaks)[0][1:][anomalies.values]
    peaks[rm_peaks] = 0
    new_rr = np.diff(np.where(peaks))[0]

    # Create a new DataFrame
    time = pd.to_datetime(np.cumsum(new_rr), unit='ms')
    df = pd.DataFrame({'rr': new_rr}, index=time)
    df = validate_series(df.rr)

    #############################
    # Find low frequency outliers
    #############################
    quantile_ad = GeneralizedESDTestAD()
    anomalies = quantile_ad.fit_detect(df)

    # Add R peaks using peak_replacement
    clean, per = [], []  # Store each outlier removal separately
    for outlier in np.where(anomalies.values)[0]:
        new_peaks, npeaks = peak_replacement(peaks, outlier)
        clean.append(new_peaks)
        per.append(npeaks)

    # Merge all vectors together
    clean_peaks = peaks.copy()
    for c in clean:
        clean_peaks = np.logical_or(clean_peaks, c)

    return clean_peaks.astype(int), sum(per)/sum(peaks)


def peak_replacement(peaks, outlier):
    """Given the index of a detected peak, will add the requiered number of R
    peaks to minimize signal standard deviation.

    Parameters
    ----------
    rr: boolean array
        Indexes of R peaks.
    outlier: int
        Index of detected outlier

    Return
    ------
    new_rr: boolean array
        New indexes vector with added spikes.
    npeaks: int
        Number of peaks added [1-5]
    """
    actual = np.where(peaks)[0][1:][outlier]
    previous = np.where(peaks)[0][1:][outlier-1]

    # Replace n peaks by minimising signal standard deviation
    min_std = np.inf
    for i in range(1, 5):

        this_peaks = peaks.copy()

        new_rr = int((actual - previous)/(i+1))

        for ii in range(i):

            # Add peak in vector
            new_pos = previous+(new_rr*(ii+1))
            this_peaks[new_pos] = 1

        # Compute signal std to measure effect of peak replacement
        this_std = np.std(np.diff(np.where(this_peaks)[0]))
        if min_std > this_std:
            min_std = this_std
            final_peaks = this_peaks.copy()
            npeaks = i

    return final_peaks, npeaks
