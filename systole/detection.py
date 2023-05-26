# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sleepecg import detect_heartbeats

from systole.detectors import (
    christov,
    engelse_zeelenberg,
    hamilton,
    moving_average,
    msptd,
    pan_tompkins,
    rolling_average_ppg,
    rolling_average_resp,
)
from systole.utils import find_clipping, input_conversion, nan_cleaning, to_neighbour


def ppg_peaks(
    signal: Union[List, np.ndarray, pd.Series],
    sfreq: int,
    new_sfreq: int = 1000,
    method: str = "rolling_average",
    clipping: bool = True,
    clipping_thresholds: Union[Tuple, List, str] = "auto",
    clean_nan: bool = False,
    verbose: bool = False,
    detector_kws: Dict = {},
) -> Tuple[np.ndarray, np.ndarray]:
    """Systolic peak detection for PPG signals.

    Two methods are available:
    - an adaptation of the rolling average + standard deviation approach described in
    [1]_.
    - The Multi-scale peak and trough detection algorithm (MSPTD) [2]_.

    Before peaks detection, nans are interpolated (optional, `Fale` by default) the
    signal is resampled to the new sampling frequency (1000 Hz by default) and clipping
    artefacts are corrected using cubic spline interpolation (optional, `True` by
    default).

    .. note :: This function will resample the signal to 1000 Hz by default.

    Parameters
    ----------
    signal :
        The raw signal recorded from the pulse oximeter time series.
    sfreq :
        The sampling frequency (Hz).
    new_sfreq :
        If resample is `True`, the new sampling frequency (Hz). Defaults to `1000`.
    method :
        The systolic peaks detection algorithm to use, can be `"rolling_average"` [1]_
        (default) or `"msptd"` [2]_.
    clipping :
        If `True`, will apply the clipping artefact correction described in [1]_.
        Defaults to `True`.
    clipping_thresholds :
        The values of the minumum and maximum clipping thresholds. Can be a float or
        `None`. If `None`, no correction is applied. If "auto" is provided, will use
       :py:func:`systole.utils.find_clipping` to find the values. Defaults to `"auto"`.
        This parameter is only relevant if `cliping` is `True`.
    clean_nan :
        If `True`, will interpolate NaNs values if any before any other operation.
        Defaults to `False`.
    verbose :
        Control function verbosity. Defaults to `False` (do not print processing steps).
    detector_kws :
        Additional keyword arguments that will be passed to the detector function.

    Returns
    -------
    resampled_signal :
        Signal resampled to the `new_sfreq` frequency.
    peaks :
        Boolean array of systolic peaks detection.

    Raises
    ------
    ValueError
        If `clipping_thresholds` is not a tuple, a list or `"auto"`.
        If `method` is not a valid method name.

    Examples
    --------
    >>> from systole import import_ppg
    >>> from systole.detection import ppg_peaks
    >>> ppg = import_ppg().ppg.to_numpy()  # Import PPG signal

    Using the rolling average method (default)
    ******************************************
    >>> signal, peaks = ppg_peaks(signal=ppg, method="rolling_average")
    >>> print(f'{sum(peaks)} peaks detected.')
    378 peaks detected.

    Using the Multi-scale peak and trough detection algorithm
    *********************************************************
    >>> signal, peaks = ppg_peaks(signal=ppg, method="msptd")
    >>> print(f'{sum(peaks)} peaks detected.')
    378 peaks detected.

    References
    ----------
    .. [1] van Gent, P., Farah, H., van Nes, N. and van Arem, B., 2019.
       Analysing Noisy Driver Physiology Real-Time Using Off-the-Shelf Sensors:
       Heart Rate Analysis Software from the Taking the Fast Lane Project. Journal
       of Open Research Software, 7(1), p.32. DOI: http://doi.org/10.5334/jors.241
    .. [2] S. M. Bishop and A. Ercole, 'Multi-scale peak and trough detection optimised
       for periodic and quasi-periodic neuroscience data,' in Intracranial Pressure and
       Neuromonitoring XVI. Acta Neurochirurgica Supplement, T. Heldt, Ed. Springer,
       2018, vol. 126, pp. 189-195. <https://doi.org/10.1007/978-3-319-65798-1_39>

    """

    x = np.asarray(signal)

    # Interpolate NaNs values if any and if requested
    if clean_nan is True:
        if np.isnan(x).any():
            x = nan_cleaning(signal=x, verbose=verbose)

    # Resample signal to the new frequnecy if required
    if sfreq != new_sfreq:
        time = np.arange(0, len(x)) / sfreq
        new_time = np.arange(0, len(x) / sfreq, 1 / new_sfreq)
        x = np.interp(new_time, time, x)

    # Copy resampled signal for output
    resampled_signal = np.copy(x)

    # Remove clipping artefacts with cubic interpolation
    if clipping is True:
        if clipping_thresholds == "auto":
            min_threshold, max_threshold = find_clipping(signal=x)
        elif isinstance(clipping_thresholds, list) | isinstance(
            clipping_thresholds, tuple
        ):
            min_threshold, max_threshold = clipping_thresholds  # type: ignore
        else:
            raise ValueError(
                (
                    "The variable clipping_thresholds should be a list"
                    "or a tuple with length 2 or 'auto'."
                )
            )
        x = interpolate_clipping(
            signal=x, min_threshold=min_threshold, max_threshold=max_threshold
        )

    if method == "msptd":
        peaks_idx = msptd(signal=x, sfreq=new_sfreq, kind="peaks", **detector_kws)
    elif method == "rolling_average":
        peaks_idx = rolling_average_ppg(signal=x, sfreq=new_sfreq, **detector_kws)
    else:
        raise ValueError("Invalid method argument.")

    peaks = np.zeros(len(resampled_signal), dtype=bool)
    peaks[peaks_idx] = True

    return resampled_signal, peaks


def ecg_peaks(
    signal: Union[List, np.ndarray, pd.Series],
    sfreq: int = 1000,
    new_sfreq: int = 1000,
    method: str = "sleepecg",
    find_local: bool = False,
    win_size: float = 0.1,
    clean_nan: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """A simple wrapper for many popular R peaks detectors algorithms.

    This function calls methods from `py-ecg-detectors` [1]_.

    Parameters
    ----------
    signal :
        The raw ECG signal.
    sfreq :
        The sampling frequency. Default is set to `1000` Hz.
    new_sfreq :
        The new sampling frequency. Defaults to `1000` Hz.
    method :
        The method used. Can be one of the following: `'sleepecg'`, `'hamilton'`,
        `'christov'`, `'engelse-zeelenberg'`, `'pan-tompkins'`, `'moving-average'`.
    find_local :
        If *True*, will use peaks indexs to search for local peaks given the
        window size (win_size).
    win_size :
        Size of the time window used by :py:func:`systole.utils.to_neighbour()`
        expressed in seconds. Defaut set to `0.1`.
    clean_nan :
        If `True`, will interpolate NaNs values if any before any other operation.
        Defaults to `False`.
    verbose :
        Control function verbosity. Defaults to `False` (do not print processing steps).

    Returns
    -------
    resampled_signal :
        Signal resampled to the `new_sfreq` frequency.
    peaks :
        Boolean array corresponding to the R peaks detection.

    Raises
    ------
    ValueError
        If `method` is not one of the following: `'hamilton'`, `'christov'`,
            `'engelse-zeelenberg'`, `'pan-tompkins'`, `'moving-average'`

    Notes
    -----
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

    x = np.asarray(signal)

    # Interpolate NaNs values if any and if requested
    if clean_nan is True:
        if np.isnan(x).any():
            x = nan_cleaning(signal=x, verbose=verbose)

    # Resample signal to the new frequnecy if required
    if sfreq != new_sfreq:
        time = np.arange(0, len(x)) / sfreq
        new_time = np.arange(0, len(x) / sfreq, 1 / new_sfreq)
        x = np.interp(new_time, time, x)

    # Copy resampled signal for output
    resampled_signal = np.copy(x)

    if method == "sleepecg":
        peaks_idx = detect_heartbeats(resampled_signal, fs=new_sfreq)
    elif method == "hamilton":
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
            "Invalid method provided, should be: sleepecg, hamilton, "
            "christov, engelse-zeelenberg, pan-tompkins, wavelet-transform, "
            "moving-average"
        )
    peaks = np.zeros(len(resampled_signal), dtype=bool)
    peaks[peaks_idx] = True

    if find_local is True:
        peaks = to_neighbour(resampled_signal, peaks, size=int(win_size * new_sfreq))

    return resampled_signal, peaks


def rsp_peaks(
    signal: Union[List, np.ndarray, pd.Series],
    sfreq: int,
    new_sfreq: int = 1000,
    method: str = "msptd",
    kind: str = "peaks-onsets",
    clean_nan: bool = False,
    verbose: bool = False,
    detector_kws: Dict = {},
) -> Tuple[np.ndarray, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """Identify peaks and/or onsets in respiratory signal.

    Parameters
    ----------
    signal :
        The respiratory signal. Peaks are considered to represent end of inspiration,
        trough represent end of expiration.
    sfreq :
        The sampling frequency.
    new_sfreq :
        If resample is `True`, the new sampling frequency. Defaults to `1000` Hz.
    method :
        The peaks detection algorithm to use, can be `"rolling_average"` for an
        adaptation of [1]_ or `"msptd"` [2]_ (default).
    kind :
        What kind of detection to perform. Peak detection (`"peaks"`), trough detection
        (`"onsets"`) or both (`"peaks-onsets"`, default).
    clean_nan :
        If `True`, will interpolate NaNs values if any before any other operation.
        Defaults to `False`.
    verbose :
        Control function verbosity. Defaults to `False` (do not print processing steps).
    detector_kws :
        Additional keyword arguments that will be passed to the detector function.

    Returns
    -------
    resampled_signal :
        Signal resampled to the `new_sfreq` frequency.
    peaks | trough | (peaks, trough) :
        Boolean arrays of peaks and / or onsets in the respiratory signal.

    Raises
    ------
    ValueError
        If `kind` is not one of the following: `"peaks"`, `"onsets"` or
        `"peaks-onsets"`.
        If `method` is not a valid method name.

    References
    ----------
    .. [1] Torben Noto, Guangyu Zhou, Stephan Schuele, Jessica Templer, Christina
       Zelano,Automated analysis of breathing waveforms using BreathMetrics: a
       respiratory signal processing toolbox, Chemical Senses, Volume 43, Issue 8,
       October 2018, Pages 583-597, https://doi.org/10.1093/chemse/bjy045
    .. [2] S. M. Bishop and A. Ercole, 'Multi-scale peak and trough detection optimised
       for periodic and quasi-periodic neuroscience data,' in Intracranial Pressure and
       Neuromonitoring XVI. Acta Neurochirurgica Supplement, T. Heldt, Ed. Springer,
       2018, vol. 126, pp. 189-195. <https://doi.org/10.1007/978-3-319-65798-1_39>

    """
    if kind not in ["peaks", "onsets", "peaks-onsets"]:
        raise ValueError(
            "Invalid kind parameter. Should be 'peaks', 'onsets' or 'peaks-onsets'"
        )

    x = np.asarray(signal)

    # Interpolate NaNs values if any and if requested
    if clean_nan is True:
        if np.isnan(x).any():
            x = nan_cleaning(signal=x, verbose=verbose)

    # Resample signal to the new frequnecy if required
    if sfreq != new_sfreq:
        time = np.arange(0, len(x)) / sfreq
        new_time = np.arange(0, len(x) / sfreq, 1 / new_sfreq)
        x = np.interp(new_time, time, x)

    # Copy resampled signal for output
    resampled_signal = np.copy(x)

    if method == "msptd":
        idxs = msptd(signal=x, sfreq=new_sfreq, kind=kind, win_len=60, **detector_kws)
    elif method == "rolling_average":
        idxs = rolling_average_resp(
            signal=x, kind=kind, sfreq=new_sfreq, **detector_kws
        )
    else:
        raise ValueError("Invalid method argument.")

    if kind == "peaks":
        peaks = np.zeros(len(resampled_signal), dtype=bool)
        peaks[idxs] = True
        return resampled_signal, peaks
    elif kind == "onsets":
        onsets = np.zeros(len(resampled_signal), dtype=bool)
        onsets[idxs] = True
        return resampled_signal, onsets
    else:
        peaks = np.zeros(len(resampled_signal), dtype=bool)
        peaks[idxs[0]] = True
        onsets = np.zeros(len(resampled_signal), dtype=bool)
        onsets[idxs[1]] = True
        return resampled_signal, (peaks, onsets)


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
    rr :
        1d numpy array of RR intervals (in seconds or miliseconds) or peaks
        vector (boolean array).
    c1 :
        Fixed variable controling the slope of the threshold lines. Default is
        `0.13`.
    c2 :
        Fixed variable controling the intersect of the threshold lines. Default
        is `0.17`.
    alpha :
        Scaling factor used to normalize the RR intervals first deviation.
    input_type :
        The type of input vector. Defaults to `"rr_ms"` for vectors of RR
        intervals, or  interbeat intervals (IBI), expressed in milliseconds.
        Can also be a boolean vector where `1` represents the occurrence of
        R waves or systolic peakspeaks vector `"rr_s"` or IBI expressed in
        seconds.

    Returns
    -------
    artefacts :
        Dictionary storing the parameters of RR artefacts rejection. All the vectors
        outputed have the same length as the provided RR time serie:

        * subspace1 : np.ndarray
            The first dimension. First derivative of R-R interval time serie.
        * subspace2 : np.ndarray
            The second dimension (1st plot).
        * subspace3 : np.ndarray
            The third dimension (2nd plot).
        * mRR : np.ndarray
            The mRR time serie.
        * ectopic : np.ndarray
            Boolean array indexing probable ectopic beats.
        * long : np.ndarray
            Boolean array indexing long RR intervals.
        * short : np.ndarray
            Boolean array indexing short RR intervals.
        * missed : np.ndarray
            Boolean array indexing missed RR intervals.
        * extra : np.ndarray
            Boolean array indexing extra RR intervals.
        * threshold1 : np.ndarray
            Threshold 1.
        * threshold2 : np.ndarray
            Threshold 2.

    Notes
    -----
    This function will use the method proposed by [1]_ to detect ectopic beats, long,
    shorts, missed and extra RR intervals.

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
        43(3), 173-181. https://doi.org/10.1080/03091902.2019.1640306

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
    signal: Union[List, np.ndarray],
    min_threshold: Optional[float] = 0.0,
    max_threshold: Optional[float] = 255.0,
    kind: str = "cubic",
) -> np.ndarray:
    """Interoplate clipping artefacts.

    This function removes all data points greather/lower or equalling the provided
    thresholds and re-creates the missing segments using interpolation (default is
    `"cubic"`).

    Parameters
    ----------
    signal :
        The PPG signal.
    min_threshold, max_threshold : float | None
        Minimum and maximum thresholds for clipping artefacts. If `None`, no correction
        os provided for the given threshold. Defaults to `min_threshold=0.0` and
        `max_threshold=255.0`, which corresponds to the expected values when reading
        data from the Nonin 3012LP Xpod USB pulse oximeter together with Nonin 8000SM
        'soft-clip' fingertip sensors.
    kind :
        Specifies the kind of interpolation to perform(see
       :py:func:`scipy.interpolate.interp1d`).

    Returns
    -------
    clean_signal :
        Interpolated signal.

    Examples
    --------
    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> from systole import import_ppg
        >>> from systole.detection import interpolate_clipping
        >>> df = import_ppg()
        >>> # Create lower and upper clipping artefacts
        >>> df.ppg.loc[df.ppg<=50] = 50
        >>> df.ppg.loc[df.ppg>=230] = 230
        >>> # Correct clipping artefacts
        >>> clean_signal = interpolate_clipping(df.ppg.to_numpy(), min_threshold=50, max_threshold=230)
        >>> # Plot
        >>> plt.plot(df.time, clean_signal, color='#F15854', label="Corrected signal")
        >>> plt.plot(df.time, df.ppg, color='#5DA5DA', label="Clipping artefacts")
        >>> plt.axhline(y=50, linestyle='--', color='k')
        >>> plt.axhline(y=230, linestyle='--', color='k')
        >>> plt.xlabel('Time (s)')
        >>> plt.ylabel('PPG level (a.u)')
        >>> plt.xlim(10, 40)
        >>> plt.legend()

    Notes
    -----
    Correct signal segment greather/smaller or equalling the recording threshold using
    cubic spline interpolation. Adapted from [1]_.

    .. Warning:: If clipping artefact is found at the edge of the signal, this
        function will decrement/increment the first/last value to allow interpolation.

    The first and last values are corrected for interpolation by adding/substracting
    the median step observed in the time series.

    References
    ----------
    .. [1] https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/

    """
    clean_signal = np.asarray(signal)
    time = np.arange(0, len(signal))

    # What is the median step in the time serie?
    # Use that to correct first and last values if required
    step = np.median(np.diff(np.sort(np.unique(clean_signal))))

    if max_threshold is not None:
        # Security check for clipping at signal edge
        if clean_signal[0] >= max_threshold:
            clean_signal[0] = max_threshold - step
        if clean_signal[-1] >= max_threshold:
            clean_signal[-1] = max_threshold - step

        # Interpolate
        f = interp1d(
            time[np.where(clean_signal < max_threshold)[0]],
            clean_signal[np.where(clean_signal < max_threshold)[0]],
            kind=kind,
        )

        # Use the peaks vector as time input
        clean_signal = f(time)

    if min_threshold is not None:
        # Security check for clipping at signal edge
        if clean_signal[0] <= min_threshold:
            clean_signal[0] = min_threshold + step
        if clean_signal[-1] <= min_threshold:
            clean_signal[-1] = min_threshold + step

        # Interpolate
        f = interp1d(
            time[np.where(clean_signal > min_threshold)[0]],
            clean_signal[np.where(clean_signal > min_threshold)[0]],
            kind=kind,
        )

        # Use the peaks vector as time input
        clean_signal = f(time)

    return clean_signal
