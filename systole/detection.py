# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from systole.detectors import (
    christov,
    engelse_zeelenberg,
    hamilton,
    moving_average,
    pan_tompkins,
)
from systole.utils import input_conversion, to_neighbour


def ppg_peaks(
    x: Union[List, np.ndarray],
    sfreq: int = 75,
    win: float = 0.75,
    new_sfreq: int = 1000,
    clipping: bool = True,
    noise_removal: bool = True,
    peak_enhancement: bool = True,
    distance: float = 0.3,
    clean_extra: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """A simple systolic peak finder for PPG signals.

    This method uses a rolling average + standard deviation
    approach to update a detection threshold. All the peaks found
    above this threshold are potential systolic peaks.

    Parameters
    ----------
    x : np.ndarray or list
        The pulse oximeter time series.
    sfreq : int
        The sampling frequency. Default is set to 75 Hz.
    win : int
        Window size (in seconds) used to compute the threshold (i.e.
        rolling mean + standard deviation).
    new_sfreq : int
        If resample is *True*, the new sampling frequency.
    resample : boolean
        If `True` (default), will resample the signal at *new_sfreq*. Default
        value is 1000 Hz.
    peak_enhancement : boolean
        If `True` (default), the ppg signal is squared before peaks detection.
    distance : float
        The minimum interval between two peaks (seconds).
    clean_extra : bool
        If `True`, use `:py:func:systole.detection.rr_artefacts()` to find and
        remove extra peaks. Default is `False`.

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
    noise and clipping, using method detailled in [1]_. The signal is then
    squared and detection of peaks is performed using threshold corresponding
    to the moving averagte + stadard deviation.

    .. warning :: This function will resample the signal to 1000 Hz by default.

    Examples
    --------
    >>> from systole import import_ppg
    >>> from systole.detection import ppg_peaks
    >>> df = import_ppg()  # Import PPG recording
    >>> signal, peaks = ppg_peaks(df.ppg.to_numpy())
    >>> print(f'{sum(peaks)} peaks detected.')
    378 peaks detected.

    References
    ----------
    .. [1] van Gent, P., Farah, H., van Nes, N. and van Arem, B., 2019.
    Analysing Noisy Driver Physiology Real-Time Using Off-the-Shelf Sensors:
    Heart Rate Analysis Software from the Taking the Fast Lane Project. Journal
    of Open Research Software, 7(1), p.32. DOI: http://doi.org/10.5334/jors.241

    """
    x = np.asarray(x)

    # Interpolate
    time = np.arange(0, len(x) / sfreq, 1 / sfreq)
    new_time = np.arange(0, len(x) / sfreq, 1 / new_sfreq)
    x = np.interp(new_time, time, x)

    # Copy resampled signal for output
    resampled_signal = np.copy(x)

    # Remove clipping artefacts with cubic interpolation
    if clipping is True:
        x = interpolate_clipping(x)

    if noise_removal is True:
        # Moving average (high frequency noise + clipping)
        rollingNoise = max(int(new_sfreq * 0.05), 1)  # 0.05 second window
        x = (
            pd.DataFrame({"signal": x})
            .rolling(rollingNoise, center=True)
            .mean()
            .signal.to_numpy()
        )
    if peak_enhancement is True:
        # Square signal (peak enhancement)
        x = (np.asarray(x) ** 2) * np.sign(x)

    # Compute moving average and standard deviation
    signal = pd.DataFrame({"signal": x})
    mean_signal = (
        signal.rolling(int(new_sfreq * win), center=True).mean().signal.to_numpy()
    )
    std_signal = (
        signal.rolling(int(new_sfreq * win), center=True).std().signal.to_numpy()
    )

    # Substract moving average + standard deviation
    x -= mean_signal + std_signal

    # Find positive peaks
    peaks_idx = find_peaks(x, height=0, distance=int(new_sfreq * distance))[0]

    # Create boolean vector
    peaks = np.zeros(len(x), dtype=bool)
    peaks[peaks_idx] = 1

    # Remove extra peaks
    if clean_extra:

        # Search artefacts
        rr = np.diff(np.where(peaks)[0])  # Convert to RR time series
        artefacts = rr_artefacts(rr)

        # Clean peak vector
        peaks[peaks_idx[1:][artefacts["extra"]]] = 0

    return resampled_signal, peaks


def ecg_peaks(
    x: Union[List, np.ndarray],
    sfreq: int = 1000,
    new_sfreq: int = 1000,
    method: str = "pan-tompkins",
    find_local: bool = True,
    win_size: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """A simple wrapper for many popular R peaks detectors algorithms.

    This function calls methods from `py-ecg-detectors` [1]_.

    Parameters
    ----------
    x : np.ndarray or list
        The oxi signal.
    sfreq : int
        The sampling frequency. Default is set to 75 Hz.
    method : str
        The method used. Can be one of the following: `'hamilton'`,
        `'christov'`, `'engelse-zeelenberg'`, `'pan-tompkins'`,
        `'wavelet-transform'`, `'moving-average'`.
    find_local : bool
        If *True*, will use peaks indexs to search for local peaks given the
        window size (win_size).
    win_size : int
        Size of the time window used by :py:func:`systole.utils.to_neighbour()`
        expressed in seconds. Defaut set to `0.1`.

    Returns
    -------
    peaks : np.ndarray
        Numpy array containing peaks index.
    resampled_signal : np.ndarray
        Signal resampled to the `new_sfreq` frequency.

    Notes
    -----
    This function will call the py-ecg-detectors package to perform R wave
    detection.

    .. warning :: This function will resample the signal to 1000 Hz.

    Examples
    --------
    >>> from systole import import_dataset
    >>> from systole.detection import ecg_peaks
    >>> signal_df = import_dataset()[:20*2000]
    >>> signal, peaks = ecg_peaks(signal_df.ecg.to_numpy(), method='hamilton',
    >>>                           sfreq=2000, find_local=True)
    >>> print(f'{sum(peaks)} peaks detected.')
    24 peaks detected.

    References
    ----------
    .. [1] Howell, L., Porr, B. Popular ECG R peak detectors written in
    python. DOI: 10.5281/zenodo.3353396

    """
    x = np.asarray(x)

    # Interpolate
    time = np.arange(0, len(x) / sfreq, 1 / sfreq)
    new_time = np.arange(0, len(x) / sfreq, 1 / new_sfreq)
    x = np.interp(new_time, time, x)

    # Copy resampled signal for output
    resampled_signal = np.copy(x)

    if method == "hamilton":
        peaks_idx = hamilton(resampled_signal, sfreq=new_sfreq)
    elif method == "christov":
        peaks_idx = christov(resampled_signal, sfreq=new_sfreq)
    elif method == "engelse-zeelenberg":
        peaks_idx = engelse_zeelenberg(resampled_signal, sfreq=new_sfreq)
    elif method == "pan-tompkins":
        peaks_idx = pan_tompkins(resampled_signal, sfreq=new_sfreq)
    elif method == "moving-average":
        peaks_idx = moving_average(resampled_signal, sfreq=new_sfreq)
    else:
        raise ValueError(
            "Invalid method provided, should be: hamilton, "
            "christov, engelse-zeelenberg, pan-tompkins, wavelet-transform, "
            "moving-average"
        )
    peaks = np.zeros(len(resampled_signal), dtype=bool)
    peaks[peaks_idx] = True

    if find_local is True:
        peaks = to_neighbour(resampled_signal, peaks, size=int(win_size * new_sfreq))

    return resampled_signal, peaks


def rr_artefacts(
    rr: Union[List, np.ndarray],
    c1: float = 0.13,
    c2: float = 0.17,
    alpha: float = 5.2,
    input_type: str = "rr_ms",
) -> Dict[str, np.ndarray]:
    """Artefacts detection from RR time series using the subspaces approach
    proposed by Lipponen & Tarvainen (2019).

    Parameters
    ----------
    rr : np.ndarray or list
        1d numpy array of RR intervals (in seconds or miliseconds) or peaks
        vector (boolean array).
    c1 : float
        Fixed variable controling the slope of the threshold lines. Default is
        `0.13`.
    c2 : float
        Fixed variable controling the intersect of the threshold lines. Default
        is `0.17`.
    alpha : float
        Scaling factor used to normalize the RR intervals first deviation.
    input_type : str
        The type of input vector. Defaults to `"rr_ms"` for vectors of RR
        intervals, or  interbeat intervals (IBI), expressed in milliseconds.
        Can also be a boolean vector where `1` represents the occurrence of
        R waves or systolic peakspeaks vector `"rr_s"` or IBI expressed in
        seconds.

    Returns
    -------
    artefacts : dict
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
    This function will use the method proposed by [1]_ to detect ectopic beats,
    long, shorts, missed and extra RR intervals.

    Examples
    --------
    >>> from systole import simulate_rr
    >>> from systole.detection import rr_artefacts
    >>> rr = simulate_rr()  # Simulate RR time series
    >>> artefacts = rr_artefacts(rr)
    >>> print(artefacts.keys())
    dict_keys(['subspace1', 'subspace2', 'subspace3', 'mRR', 'ectopic', 'long',
    'short', 'missed', 'extra', 'threshold1', 'threshold2'])

    References
    ----------
    .. [1] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for
        heart rate variability time series artefact correction using novel
        beat classification. Journal of Medical Engineering & Technology,
        43(3), 173–181. https://doi.org/10.1080/03091902.2019.1640306
    """
    rr = np.asarray(rr)

    if input_type != "rr_ms":
        rr = input_conversion(rr, input_type, output_type="rr_ms")

    ###########
    # Detection
    ###########

    # Subspace 1 (dRRs time serie)
    dRR = np.diff(rr, prepend=0)
    dRR[0] = dRR[1:].mean()  # Set first item to a realistic value

    dRR_df = pd.DataFrame({"signal": np.abs(dRR)})
    q1 = dRR_df.rolling(91, center=True, min_periods=1).quantile(0.25).signal.to_numpy()
    q3 = dRR_df.rolling(91, center=True, min_periods=1).quantile(0.75).signal.to_numpy()

    th1 = alpha * ((q3 - q1) / 2)
    dRR = dRR / th1
    s11 = dRR

    # mRRs time serie
    medRR = (
        pd.DataFrame({"signal": rr})
        .rolling(11, center=True, min_periods=1)
        .median()
        .signal.to_numpy()
    )
    mRR = rr - medRR
    mRR[mRR < 0] = 2 * mRR[mRR < 0]

    mRR_df = pd.DataFrame({"signal": np.abs(mRR)})
    q1 = mRR_df.rolling(91, center=True, min_periods=1).quantile(0.25).signal.to_numpy()
    q3 = mRR_df.rolling(91, center=True, min_periods=1).quantile(0.75).signal.to_numpy()

    th2 = alpha * ((q3 - q1) / 2)
    mRR /= th2

    # Subspace 2
    ma = np.hstack(
        [0, [np.max([dRR[i - 1], dRR[i + 1]]) for i in range(1, len(dRR) - 1)], 0]
    )
    mi = np.hstack(
        [0, [np.min([dRR[i - 1], dRR[i + 1]]) for i in range(1, len(dRR) - 1)], 0]
    )
    s12 = ma
    s12[dRR < 0] = mi[dRR < 0]

    # Subspace 3
    ma = np.hstack(
        [[np.max([dRR[i + 1], dRR[i + 2]]) for i in range(0, len(dRR) - 2)], 0, 0]
    )
    mi = np.hstack(
        [[np.min([dRR[i + 1], dRR[i + 2]]) for i in range(0, len(dRR) - 2)], 0, 0]
    )
    s22 = ma
    s22[dRR >= 0] = mi[dRR >= 0]

    ##########
    # Decision
    ##########

    # Find ectobeats
    cond1 = (s11 > 1) & (s12 < (-c1 * s11 - c2))
    cond2 = (s11 < -1) & (s12 > (-c1 * s11 + c2))
    ectopic = cond1 | cond2
    # No ectopic detection and correction at time serie edges
    ectopic[-2:] = False
    ectopic[:2] = False

    # Find long or shorts
    longBeats = ((s11 > 1) & (s22 < -1)) | ((np.abs(mRR) > 3) & (rr > np.median(rr)))
    shortBeats = ((s11 < -1) & (s22 > 1)) | ((np.abs(mRR) > 3) & (rr <= np.median(rr)))

    # Test if next interval is also outlier
    for cond in [longBeats, shortBeats]:
        for i in range(len(cond) - 2):
            if cond[i] is True:
                if np.abs(s11[i + 1]) < np.abs(s11[i + 2]):
                    cond[i + 1] = True

    # Ectopic beats are not considered as short or long
    shortBeats[ectopic] = False
    longBeats[ectopic] = False

    # Missed vector
    missed = np.abs((rr / 2) - medRR) < th2
    missed = missed & longBeats
    longBeats[missed] = False  # Missed beats are not considered as long

    # Etra vector
    extra = np.abs(rr + np.append(rr[1:], 0) - medRR) < th2
    extra = extra & shortBeats
    shortBeats[extra] = False  # Extra beats are not considered as short

    # No short or long intervals at time serie edges
    shortBeats[0], shortBeats[-1] = False, False
    longBeats[0], longBeats[-1] = False, False

    artefacts = {
        "subspace1": s11,
        "subspace2": s12,
        "subspace3": s22,
        "mRR": mRR,
        "ectopic": ectopic,
        "long": longBeats,
        "short": shortBeats,
        "missed": missed,
        "extra": extra,
        "threshold1": th1,
        "threshold2": th2,
    }

    return artefacts


def interpolate_clipping(
    signal: Union[List, np.ndarray], threshold: int = 255
) -> np.ndarray:
    """Interoplate clipping artefacts.

    This function removes all data points equalling the provided threshold
    and re-creates the missing segments using cubic spline interpolation.

    Parameters
    ----------
    signal : np.ndarray or list
        Noisy signal.
    threshold : int
        Threshold of clipping artefact.

    Returns
    -------
    clean_signal : np.ndarray
        Interpolated signal.

    Examples
    --------
    .. plot::

       >>> import matplotlib.pyplot as plt
       >>> from systole import import_ppg
       >>> from systole.detection import interpolate_clipping
       >>> df = import_ppg()
       >>> clean_signal = interpolate_clipping(df.ppg.to_numpy())
       >>> plt.plot(df.time, clean_signal, color='#F15854')
       >>> plt.plot(df.time, df.ppg, color='#5DA5DA')
       >>> plt.axhline(y=255, linestyle='--', color='k')
       >>> plt.xlabel('Time (s)')
       >>> plt.ylabel('PPG level (a.u)')

    Notes
    -----
    Correct signal segment reaching recording threshold (default is 255)
    using a cubic spline interpolation. Adapted from [1]_.

    .. Warning:: If clipping artefact is found at the edge of the signal, this
        function will decrement the first/last value to allow interpolation,
        which can result in incorrect estimation.

    References
    ----------
    .. [1] https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/
    """
    if isinstance(signal, list):
        signal = np.array(signal)

    # Security check for clipping at signal edge
    if signal[0] == threshold:
        signal[0] = threshold - 1
    if signal[-1] == threshold:
        signal[-1] = threshold - 1

    time = np.arange(0, len(signal))

    # Interpolate
    f = interp1d(
        time[np.where(signal != 255)[0]],
        signal[np.where(signal != 255)[0]],
        kind="cubic",
    )

    # Use the peaks vector as time input
    clean_signal = f(time)

    return clean_signal
