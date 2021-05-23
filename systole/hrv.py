# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import welch

from systole.utils import input_conversion


def nnX(x: Union[List, np.ndarray], t: int = 50, input_type: str = "rr_ms") -> float:
    """Number of difference in successive R-R interval > t ms.

    Parameters
    ----------
    x : np.ndarray or list
        Interval time-series (R-R in seconds or miliseconds, peaks or peaks indexes).
    t : int
        Threshold value: Defaut is set to 50 ms to calculate the nn50 index.
    input_type : str
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.

    Returns
    -------
    nnX : float
        The number of successive differences larger than a value.
    """

    x = np.asarray(x)

    if input_type != "rr_ms":
        x = input_conversion(x, input_type=input_type, output_type="rr_ms")

    if len(x.shape) > 1:
        raise ValueError("X must be a 1darray")

    # NN50: number of successive differences larger than t ms
    nn = np.sum(np.abs(np.diff(x)) > t)
    return nn


def pnnX(x: Union[List, np.ndarray], t: int = 50, input_type: str = "rr_ms") -> float:
    """Number of successive differences larger than a value (def = 50ms).

    Parameters
    ----------
    x : np.ndarray or list
        Interval time-series (R-R in seconds or miliseconds, peaks or peaks indexes).
    t : int
        Threshold value: Defaut is set to 50 ms to calculate the nn50 index.
    input_type : str
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.

    Returns
    -------
    nn : float
        The proportion of successive differences larger than a value (%).
    """

    x = np.asarray(x)

    if input_type != "rr_ms":
        x = input_conversion(x, input_type=input_type, output_type="rr_ms")

    if len(x.shape) > 1:
        raise ValueError("X must be a 1darray")

    # nnX: number of successive differences larger than t ms
    nn = nnX(x, t)

    # Proportion of successive differences larger than t ms
    pnnX = 100 * nn / len(np.diff(x))

    return pnnX


def rmssd(x: Union[List, np.ndarray], input_type: str = "rr_ms") -> float:
    """Root Mean Square of Successive Differences.

    Parameters
    ----------
    x : np.ndarray or list
        Interval time-series (R-R in seconds or miliseconds, peaks or peaks indexes).
    input_type : str
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.

    Returns
    -------
    y : float
        The Root Mean Square of Successive Differences (RMSSD).

    Examples
    --------
    >>> rr = [800, 850, 810, 720]
    >>> rmssd(rr)
    """

    x = np.asarray(x)

    if input_type != "rr_ms":
        x = input_conversion(x, input_type=input_type, output_type="rr_ms")

    if len(x.shape) > 1:
        raise ValueError("X must be a 1darray")

    y = np.sqrt(np.mean(np.square(np.diff(x))))

    return y


def time_domain(x: Union[List, np.ndarray], input_type: str = "rr_ms") -> pd.DataFrame:
    """Extract all time domain parameters from R-R intervals.

    Parameters
    ----------
    x : np.ndarray or list
        Interval time-series (R-R in seconds or miliseconds, peaks or peaks indexes).
    input_type : str
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`
        Time domain summary statistics.
        * ``'Mean RR'`` : Mean of R-R intervals.
        * ``'Mean BPM'`` : Mean of beats per minutes.
        * ``'Median RR'`` : Median of R-R intervals'.
        * ``'Median BPM'`` : Meidan of beats per minutes.
        * ``'MinRR'`` : Minimum R-R intervals.
        * ``'MinBPM'`` : Minimum beats per minutes.
        * ``'MaxRR'`` : Maximum R-R intervals.
        * ``'MaxBPM'`` : Maximum beats per minutes.
        * ``'SDNN'`` : Standard deviation of successive differences.
        * ``'RMSSD'`` : Root Mean Square of the Successive Differences.
        * ``'NN50'`` : number of successive differences larger than 50ms.
        * ``'pNN50'`` : Proportion of successive difference larger than 50ms.

    See also
    --------
    frequency_domain, nonlinear

    Notes
    -----
    The dataframe containing the summary statistics is returned in the long
    format to facilitate the creation of group summary data frame that can
    easily be transferred to other plotting or statistics library. You can
    easily convert it into a wide format for a subject-level inline report
    using the py:pandas.pivot_table() function:
    >>> pd.pivot_table(stats, values='Values', columns='Metric')
    """

    x = np.asarray(x)

    if input_type != "rr_ms":
        x = input_conversion(x, input_type=input_type, output_type="rr_ms")

    if len(x.shape) > 1:
        raise ValueError("X must be a 1darray")

    # Mean R-R intervals
    mean_rr = round(np.mean(x))  # type: ignore

    # Mean BPM
    mean_bpm = round(np.mean(60000 / x), 2)  # type: ignore

    # Median BPM
    median_rr = round(np.median(x), 2)

    # Median BPM
    median_bpm = round(np.median(60000 / x), 2)

    # Minimum RR
    min_rr = round(np.min(x), 2)

    # Minimum BPM
    min_bpm = round(np.min(60000 / x), 2)

    # Maximum RR
    max_rr = round(np.max(x), 2)

    # Maximum BPM
    max_bpm = round(np.max(60000 / x), 2)

    # Standard deviation of R-R intervals
    sdnn = round(x.std(ddof=1), 2)  # type: ignore

    # Root Mean Square of Successive Differences (RMSSD)
    rms = round(rmssd(x), 2)

    # NN50: number of successive differences larger than 50ms
    nn = round(nnX(x, t=50), 2)

    # pNN50: Proportion of successive differences larger than 50ms
    pnn = round(pnnX(x, t=50), 2)

    # Create summary dataframe
    values = [
        mean_rr,
        mean_bpm,
        median_rr,
        median_bpm,
        min_rr,
        min_bpm,
        max_rr,
        max_bpm,
        sdnn,
        rms,
        nn,
        pnn,
    ]
    metrics = [
        "MeanRR",
        "MeanBPM",
        "MedianRR",
        "MedianBPM",
        "MinRR",
        "MinBPM",
        "MaxRR",
        "MaxBPM",
        "SDNN",
        "RMSSD",
        "nn50",
        "pnn50",
    ]

    stats = pd.DataFrame({"Values": values, "Metric": metrics})

    return stats


def psd(
    rr: Union[List, np.ndarray],
    sfreq: int = 5,
    method: str = "welch",
    input_type: str = "rr_ms",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the frequency domain features of heart rate variability.

    Parameters
    ----------
    rr : np.ndarray or list
        Interval time-series (R-R in seconds or miliseconds, peaks or peaks indexes).
    sfreq : int
        The sampling frequency (Hz) of the interpolated instantaneous heart
        rate.
    method : str
        The method used to extract freauency power. Default is ``'welch'``.
    input_type : str
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.

    Returns
    -------
    freq, power : np.array
        The frequency and power spectral density of the given signal.

    See also
    --------
    frequency_domain
    """

    rr = np.asarray(rr)

    if input_type != "rr_ms":
        rr = input_conversion(rr, input_type=input_type, output_type="rr_ms")

    # Interpolate R-R interval
    time = np.cumsum(rr)
    f = interpolate.interp1d(time, rr, kind="cubic")
    new_time = np.arange(time[0], time[-1], 1000 / sfreq)  # sfreq = 5 Hz
    x = f(new_time)

    if method == "welch":

        # Define window length
        nperseg = 256 * sfreq
        if nperseg > len(x):
            nperseg = len(x)

        # Compute Power Spectral Density
        freq, power = welch(x=x, fs=sfreq, nperseg=nperseg, nfft=nperseg)

        power = power / 1000000

    return freq, power


def frequency_domain(
    rr: Union[List, np.ndarray],
    sfreq: int = 5,
    method: str = "welch",
    fbands: Optional[Dict[str, Tuple[str, Tuple[float, float], str]]] = None,
    input_type: str = "rr_ms",
) -> pd.DataFrame:
    """Extract the frequency domain features of heart rate variability.

    Parameters
    ----------
    rr : np.ndarray or list
        Interval time-series (R-R in seconds or miliseconds, peaks or peaks indexes).
    sfreq : int
        The sampling frequency (Hz).
    method : str
        The method used to extract freauency power. Default is ``'welch'``.
    fbands : None | dict, optional
        Dictionary containing the names of the frequency bands of interest
        (str), their range (tuples) and their color in the PSD plot. Default is
        >>> {'vlf': ('Very low frequency', (0.003, 0.04), 'b'),
        >>> 'lf': ('Low frequency', (0.04, 0.15), 'g'),
        >>> 'hf': ('High frequency', (0.15, 0.4), 'r')}
    input_type : str
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`
        Frequency domain summary statistics.
        * ``'power_vlf_per'`` : Very low frequency power (%).
        * ``'power_lf_per'`` : Low frequency power (%).
        * ``'power_hf_per'`` : High frequency power (%).
        * ``'power_lf_nu'`` : Low frequency power (normalized units).
        * ``'power_hf_nu'`` : High frequency power (normalized units).

    See also
    --------
    time_domain, nonlinear

    Notes
    -----
    The dataframe containing the summary statistics is returned in the long
    format to facilitate the creation of group summary data frame that can
    easily be transferred to other plotting or statistics library. You can
    easily convert it into a wide format for a subject-level inline report
    using the py:pandas.pivot_table() function:
    >>> pd.pivot_table(stats, values='Values', columns='Metric')

    """

    rr = np.asarray(rr)

    if input_type != "rr_ms":
        rr = input_conversion(rr, input_type=input_type, output_type="rr_ms")

    freq, power = psd(rr=rr, sfreq=sfreq, method=method, input_type="rr_ms")

    if fbands is None:
        fbands = {
            "vlf": ("Very low frequency", (0.003, 0.04), "b"),
            "lf": ("Low frequency", (0.04, 0.15), "g"),
            "hf": ("High frequency", (0.15, 0.4), "r"),
        }

    # Extract HRV parameters
    ########################
    stats = pd.DataFrame([])
    for band in fbands:
        this_psd = power[(freq >= fbands[band][1][0]) & (freq < fbands[band][1][1])]
        this_freq = freq[(freq >= fbands[band][1][0]) & (freq < fbands[band][1][1])]

        # Peaks (Hz)
        peak = round(this_freq[np.argmax(this_psd)], 4)
        stats = stats.append(
            {"Values": peak, "Metric": band + "_peak"}, ignore_index=True
        )

        # Power (ms**2)
        this_power = np.trapz(x=this_freq, y=this_psd) * 1000000
        stats = stats.append(
            {"Values": this_power, "Metric": band + "_power"}, ignore_index=True
        )

    hf = stats.Values[stats.Metric == "hf_power"].values[0]
    lf = stats.Values[stats.Metric == "lf_power"].values[0]
    vlf = stats.Values[stats.Metric == "vlf_power"].values[0]

    # Power (%)
    power_per_vlf = vlf / (vlf + lf + hf) * 100
    power_per_lf = lf / (vlf + lf + hf) * 100
    power_per_hf = hf / (vlf + lf + hf) * 100

    # Power (n.u.)
    power_nu_hf = hf / (hf + lf)
    power_nu_lf = lf / (hf + lf)

    values = [power_per_vlf, power_per_lf, power_per_hf, power_nu_hf, power_nu_lf]
    metrics = [
        "power_vlf_per",
        "power_lf_per",
        "power_hf_per",
        "power_lf_nu",
        "power_hf_nu",
    ]

    stats = stats.append(
        pd.DataFrame({"Values": values, "Metric": metrics}), ignore_index=True
    )

    return stats


def nonlinear(x: Union[List, np.ndarray], input_type: str = "rr_ms") -> pd.DataFrame:
    """Extract the non-linear features of heart rate variability.

    Parameters
    ----------
    x : list or numpy array
        Interval time-series (R-R, beat-to-beat...), in miliseconds.
    input_type : str
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`
        Non-linear domain summary statistics.
        * ``'SD1'`` : SD1.
        * ``'SD2'`` : SD2.

    See also
    --------
    time_domain, frequency_domain

    Notes
    -----
    The dataframe containing the summary statistics is returned in the long
    format to facilitate the creation of group summary data frame that can
    easily be transferred to other plotting or statistics library. You can
    easily convert it into a wide format for a subject-level inline report
    using the py:pandas.pivot_table() function:
    >>> pd.pivot_table(stats, values='Values', columns='Metric')

    """

    x = np.asarray(x)

    if input_type != "rr_ms":
        x = input_conversion(x, input_type=input_type, output_type="rr_ms")

    diff_rr = np.diff(x)
    sd1 = np.sqrt(np.std(diff_rr, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(x, ddof=1) ** 2 - 0.5 * np.std(diff_rr, ddof=1) ** 2)
    values = [sd1, sd2]
    metrics = ["SD1", "SD2"]

    stats = pd.DataFrame({"Values": values, "Metric": metrics})

    return stats
