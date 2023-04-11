# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import jit
from scipy import interpolate
from scipy.signal import welch
from scipy.spatial.distance import cdist

from systole.utils import input_conversion


def nnX(rr: Union[List, np.ndarray], t: int = 50, input_type: str = "rr_ms") -> int:
    """Number of difference in successive R-R interval > t ms.

    Parameters
    ----------
    rr :
        R-R interval time-series, peaks or peaks index vectors. The default expected
        vector is R-R intervals in milliseconds. Other data format can be provided by
        specifying the `"input_type"` (can be `"rr_s"`, `"peaks"` or `"peaks_idx"`).
    t :
        Threshold value: Defaut is set to 50 ms to calculate the nn50 index.
    input_type :
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.

    Returns
    -------
    nnX :
        The number of successive differences larger than a value.

    """

    rr = np.asarray(rr)

    if input_type != "rr_ms":
        rr = input_conversion(rr, input_type=input_type, output_type="rr_ms")

    if len(rr.shape) > 1:
        raise ValueError("X must be a 1darray")

    # NN50: number of successive differences larger than t ms
    nn = np.sum(np.abs(np.diff(rr)) > t).astype(int)

    return nn


def pnnX(rr: Union[List, np.ndarray], t: int = 50, input_type: str = "rr_ms") -> float:
    """Number of successive differences larger than a value (def = 50ms).

    Parameters
    ----------
    rr :
        R-R interval time-series, peaks or peaks index vectors. The default expected
        vector is R-R intervals in milliseconds. Other data format can be provided by
        specifying the `"input_type"` (can be `"rr_s"`, `"peaks"` or `"peaks_idx"`).
    t :
        Threshold value: Defaut is set to 50 ms to calculate the nn50 index.
    input_type :
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.

    Returns
    -------
    nn :
        The proportion of successive differences larger than a value (%).

    """

    rr = np.asarray(rr)

    if input_type != "rr_ms":
        rr = input_conversion(rr, input_type=input_type, output_type="rr_ms")

    if len(rr.shape) > 1:
        raise ValueError("X must be a 1darray")

    # nnX: number of successive differences larger than t ms
    nn = nnX(rr, t)

    # Proportion of successive differences larger than t ms
    pnnX = 100 * nn / len(np.diff(rr))

    return pnnX


def rmssd(rr: Union[List, np.ndarray], input_type: str = "rr_ms") -> float:
    """Root Mean Square of Successive Differences.

    Parameters
    ----------
    rr :
        R-R interval time-series, peaks or peaks index vectors. The default expected
        vector is R-R intervals in milliseconds. Other data format can be provided by
        specifying the `"input_type"` (can be `"rr_s"`, `"peaks"` or `"peaks_idx"`).
    input_type :
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.

    Returns
    -------
    y :
        The Root Mean Square of Successive Differences (RMSSD).

    Examples
    --------
    >>> rr = [800, 850, 810, 720]
    >>> rmssd(rr)
    63.77042156569664

    """

    rr = np.asarray(rr)

    if input_type != "rr_ms":
        rr = input_conversion(rr, input_type=input_type, output_type="rr_ms")

    if len(rr.shape) > 1:
        raise ValueError("X must be a 1darray")

    y = np.sqrt(np.mean(np.square(np.diff(rr))))

    return y


def time_domain(rr: Union[List, np.ndarray], input_type: str = "rr_ms") -> pd.DataFrame:
    """Extract all time domain parameters from R-R intervals.

    Parameters
    ----------
    rr :
        R-R interval time-series, peaks or peaks index vectors. The default expected
        vector is R-R intervals in milliseconds. Other data format can be provided by
        specifying the `"input_type"` (can be `"rr_s"`, `"peaks"` or `"peaks_idx"`).
    input_type :
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.

    Returns
    -------
    stats :
        Time domain summary statistics.
        * 'MeanRR' : Mean of R-R intervals (ms).
        * 'MeanBPM' : Mean of beats per minutes (bpm).
        * 'MedianRR' : Median of R-R intervals' (ms).
        * 'MedianBPM' : Median of beats per minutes (bpm).
        * 'MinRR' : Minimum R-R intervals (ms).
        * 'MinBPM' : Minimum beats per minutes (bpm).
        * 'MaxRR' : Maximum R-R intervals (ms).
        * 'MaxBPM' : Maximum beats per minutes (bpm).
        * 'SDNN' : Standard deviation of RR intervals (ms).
        * 'SDSD' : Standard deviation of the Successive difference (ms).
        * 'RMSSD' : Root Mean Square of the Successive Differences (ms).
        * 'nn50' : number of successive differences larger than 50ms (count).
        * 'pnn50' : Proportion of successive difference larger than 50ms (%).

    See also
    --------
    frequency_domain, nonlinear_domain

    Notes
    -----
    The dataframe containing the summary statistics is returned in the long
    format to facilitate the creation of group summary data frame that can
    easily be transferred to other plotting or statistics library. You can
    easily convert it into a wide format for a subject-level inline report
    using the:py:func:`pandas.pivot_table` function:
    >>> pd.pivot_table(stats, values='Values', columns='Metric')

    All time-domain results have been tested against Kubios HVR 2.2
    (<https://www.kubios.com>).

    """

    rr = np.asarray(rr)

    if input_type != "rr_ms":
        rr = input_conversion(rr, input_type=input_type, output_type="rr_ms")

    if len(rr.shape) > 1:
        raise ValueError("X must be a 1darray")

    # Mean R-R intervals
    mean_rr = round(np.mean(rr), 6)  # type: ignore

    # Mean BPM
    mean_bpm = round(np.mean(60000 / rr), 6)  # type: ignore

    # Median BPM
    median_rr = round(np.median(rr), 6)

    # Median BPM
    median_bpm = round(np.median(60000 / rr), 6)

    # Minimum RR
    min_rr = round(np.min(rr), 6)

    # Minimum BPM
    min_bpm = round(np.min(60000 / rr), 6)

    # Maximum RR
    max_rr = round(np.max(rr), 6)

    # Maximum BPM
    max_bpm = round(np.max(60000 / rr), 6)

    # Standard deviation of R-R intervals
    sdnn = round(rr.std(ddof=1), 6)  # type: ignore

    # Standard deviation of the difference of successive R-R intervals
    sdsd = round(np.diff(rr).std(ddof=1), 6)  # type: ignore

    # Root Mean Square of Successive Differences (RMSSD)
    rms = round(rmssd(rr), 6)

    # NN50: number of successive differences larger than 50ms
    nn = round(nnX(rr, t=50), 6)

    # pNN50: Proportion of successive differences larger than 50ms
    pnn = round(pnnX(rr, t=50), 6)

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
        sdsd,
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
        "SDSD",
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
    rr :
        R-R interval time-series, peaks or peaks index vectors. The default expected
        vector is R-R intervals in milliseconds. Other data format can be provided by
        specifying the `"input_type"` (can be `"rr_s"`, `"peaks"` or `"peaks_idx"`).
    sfreq :
        The sampling frequency (Hz) of the interpolated instantaneous heart rate.
    method :
        The method used to extract freauency power. Default is ``'welch'``.
    input_type :
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.

    Returns
    -------
    freq, power :
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
    rr :
        R-R interval time-series, peaks or peaks index vectors. The default expected
        vector is R-R intervals in milliseconds. Other data format can be provided by
        specifying the `"input_type"` (can be `"rr_s"`, `"peaks"` or `"peaks_idx"`).
    sfreq :
        The sampling frequency (Hz) used to interpolate the instantaneous heart rate
        for PSD computation.
    method :
        The method used to extract the power of the different frequency bands. Default
        is ``'welch'`` (only one method is implemented for now).
    fbands :
        Dictionary containing the names of the frequency bands of interest
        (str), their range (tuples) and their color in the PSD plot. Default is
        >>> {'vlf': ('Very low frequency', (0.003, 0.04), 'b'),
        >>>  'lf': ('Low frequency', (0.04, 0.15), 'g'),
        >>>  'hf': ('High frequency', (0.15, 0.4), 'r')}
    input_type :
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.

    Returns
    -------
    stats :
        Frequency domain summary statistics.
        * 'vlf_peak' : Very low frequency peak (HZ).
        * 'vlf_power' : Very low frequency power (ms²).
        * 'lf_peak' : Low frquency peak (Hz).
        * 'lf_power' : Low frequency power (ms²).
        * 'hf_peak' : High frequency peak (Hz).
        * 'hf_power' : High frequency power (ms²).
        * 'vlf_power_per' : Very low frequency power (%).
        * 'lf_power_per' : Low frequency power (%).
        * 'hf_power_per' : High frequency power (%).
        * 'lf_power_nu' : Low frequency power (normalized units).
        * 'hf_power_nu' : High frequency power (normalized units).
        * 'total_power' : Total frequency power (ms²).
        * 'lf_hf_ratio' : Low / high frequency ratio (normalized units).

    See also
    --------
    time_domain, nonlinear

    Notes
    -----
    The dataframe containing the summary statistics is returned in the long
    format to facilitate the creation of group summary data frame that can
    easily be transferred to other plotting or statistics library. You can
    easily convert it into a wide format for a subject-level inline report
    using the:py:func:`pandas.pivot_table` function:
    >>> pd.pivot_table(stats, values='Values', columns='Metric')

    .. warning::
        All frequency-domain results have been tested against Kubios HVR 2.2
        (<https://www.kubios.com>). These results can slightly differ due to different
        parametrization of the PSD estimation. We recommend to always check your results
        against another software.

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
        peak = round(this_freq[np.argmax(this_psd)], 6)
        stats = pd.concat(
            [
                stats,
                pd.DataFrame({"Values": peak, "Metric": band + "_peak"}, index=[0]),
            ],
            ignore_index=True,
        )

        # Power (ms**2)
        this_power = np.trapz(x=this_freq, y=this_psd) * 1000000
        stats = pd.concat(
            [
                stats,
                pd.DataFrame(
                    {"Values": this_power, "Metric": band + "_power"}, index=[0]
                ),
            ],
            ignore_index=True,
        )

    # Power (ms**2)
    hf = stats.Values[stats.Metric == "hf_power"].values[0]
    lf = stats.Values[stats.Metric == "lf_power"].values[0]
    vlf = stats.Values[stats.Metric == "vlf_power"].values[0]

    total_power = vlf + lf + hf
    lf_hf_ratio = lf / hf

    # Power (%)
    power_per_vlf = vlf / (vlf + lf + hf) * 100
    power_per_lf = lf / (vlf + lf + hf) * 100
    power_per_hf = hf / (vlf + lf + hf) * 100

    # Power (n.u.)
    power_nu_hf = hf / (hf + lf) * 100
    power_nu_lf = lf / (hf + lf) * 100

    values = [
        power_per_vlf,
        power_per_lf,
        power_per_hf,
        power_nu_lf,
        power_nu_hf,
        total_power,
        lf_hf_ratio,
    ]
    metrics = [
        "vlf_power_per",
        "lf_power_per",
        "hf_power_per",
        "lf_power_nu",
        "hf_power_nu",
        "total_power",
        "lf_hf_ratio",
    ]

    stats = pd.concat(
        [stats, pd.DataFrame({"Values": values, "Metric": metrics})], ignore_index=True
    )

    return stats


def nonlinear_domain(
    rr: Union[List, np.ndarray], input_type: str = "rr_ms"
) -> pd.DataFrame:
    """Extract the non-linear features of heart rate variability.

    Parameters
    ----------
    rr :
        R-R interval time-series, peaks or peaks index vectors. The default expected
        vector is R-R intervals in milliseconds. Other data format can be provided by
        specifying the `"input_type"` (can be `"rr_s"`, `"peaks"` or `"peaks_idx"`).
    input_type :
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.

    Returns
    -------
    stats :
        Nonlinear domain summary statistics.
        * 'SD1' : SD1, the standard deviation of the poincare plot orthogonal to the identity line (ms).
        * 'SD2' : SD2, the standard deviation of the poincare plot along the identity line (ms).
        * 'recurrence_rate' : The recurrence rate in the recurrence plot (%).
        * 'l_max' : The maximun diagonal length in the recurrence plot (beats).
        * 'l_mean' : The mean diagonal length in the recurrence plot (beats).
        * 'determinism_rate' : The determinism rate in the recurrence plot (%).
        * 'shannon_entropy' : The Shannon entropy.

    See also
    --------
    time_domain, frequency_domain, poincare, rec

    Notes
    -----
    The dataframe containing the summary statistics is returned in the long
    format to facilitate the creation of group summary data frame that can
    easily be transferred to other plotting or statistics library. You can
    easily convert it into a wide format for a subject-level inline report
    using the py:pandas.pivot_table() function:
    >>> pd.pivot_table(stats, values='Values', columns='Metric')

    .. warning:: The recurrence plots results does not reproduce what is obtained using
       Kubios (3.5.0) and should be used with caution for now.

    References
    ----------
    .. [1] M. Brennan, M. Palaniswami, and P. Kamen. Do existing measures of Poincaré
       plot geometry reflect nonlinear features of heart rate variability. IEEE Trans
       Biomed Eng, 48(11):1342–1347, 2001.

    .. [2] H. Dabire, D. Mestivier, J. Jarnet, M.E. Safar, and N. Phong Chau.
       Quantification of sympathetic and parasympathetic tones by nonlinear indexes in
       normotensive rats. amj, 44:H1290–H1297, 1998.

    """

    rr = np.asarray(rr)

    if input_type != "rr_ms":
        rr = input_conversion(rr, input_type=input_type, output_type="rr_ms")

    # Pointcare plot
    sd1, sd2 = poincare(rr, input_type="rr_ms")

    # Recurrence plot
    recurrence_rate, l_max, l_mean, determinism, shan_entr = recurrence(
        rr, input_type="rr_ms"
    )

    values = [sd1, sd2, recurrence_rate, l_max, l_mean, determinism, shan_entr]
    metrics = [
        "SD1",
        "SD2",
        "recurrence_rate",
        "l_max",
        "l_mean",
        "determinism_rate",
        "shannon_entropy",
    ]

    stats = pd.DataFrame({"Values": values, "Metric": metrics})

    return stats


def poincare(
    rr: Union[List, np.ndarray], input_type: str = "rr_ms"
) -> Tuple[float, float]:
    """Compute SD1 and SD2 from the Poincaré nonlinear method for heart rate variability.

    Parameters
    ----------
    rr :
        R-R interval time-series, peaks or peaks index vectors. The default expected
        vector is R-R intervals in milliseconds. Other data format can be provided by
        specifying the `"input_type"` (can be `"rr_s"`, `"peaks"` or `"peaks_idx"`).
    input_type :
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.

    Returns
    -------
    sd1 :
        The standard deviation of the points perpendicular to the identity line. This
        metric is thought to be influenced mainly by the respiratory sinus arythmia
        (RSA) and reflect short-term heart rate variability.
    sd2 :
        The standard deviation of the points along the identity line. This metric is
        thought to reflect the long-term heart rate variability.

    See also
    --------
    nonlinear_domain, recurrence

    Notes
    -----
    The Poincare plot is a commonly used nonlinear method that is based on the
    graphical representation of the correlation between lagged successive RR intervals
    (the :math:`\\RR_{n}` intervals are plotted as a function of the :math:`\\RR_{n+1}`)
    intervals. The shape of the resulting plot is then analyzed and two metrics are
    extracted, representing the standard deviation of the distribution perpendicular to
    the identity line (SD1) and along the identity line (SD2).

    SD1, which corresponds to the standard deviation of the points perpendicular to the
    identity line, reflects short-term variability and is thought to be caused by
    respiratory sinus arrhythmia (RSA). SD2, on the other side, the standard deviation
    along the identity line, corresponds to the long-term heart rate variability.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Poincar%C3%A9_plot

    .. [2] M. Brennan, M. Palaniswami, and P. Kamen. Do existing measures of Poincaré
       plot geometry reflect nonlinear features of heart rate variability. IEEE Trans
       Biomed Eng, 48(11):1342–1347, 2001.

    """
    rr = np.asarray(rr)

    if input_type != "rr_ms":
        rr = input_conversion(rr, input_type=input_type, output_type="rr_ms")

    sd1, sd2 = _poincare(rr)

    return sd1, sd2


@jit(nopython=True)
def _poincare(rr: np.ndarray) -> Tuple[float, float]:
    """Compute SD1 and SD2 from the Poincaré nonlinear method for heart rate variability."""

    diff_rr = np.diff(rr)
    sd1 = np.sqrt(np.std(diff_rr) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(rr) ** 2 - 0.5 * np.std(diff_rr) ** 2)

    return sd1, sd2


def recurrence(
    rr: Union[List, np.ndarray], input_type: str = "rr_ms"
) -> Tuple[float, int, float, float, float]:
    """Compute quantitative metrics from the recurrence plot for heart rate variability.

    Parameters
    ----------
    rr :
        R-R interval time-series, peaks or peaks index vectors. The default expected
        vector is R-R intervals in milliseconds. Other data format can be provided by
        specifying the `"input_type"` (can be `"rr_s"`, `"peaks"` or `"peaks_idx"`).
    input_type :
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.

    Returns
    -------
    recurrence_rate :
        The percentage of recurence in the time series. This corresponds to the ratio
        of ones and zeros in the recurrence plot.
    l_max :
        Maximum lenght of the diagonale in the reccurence plot.
    l_mean :
        Mean of the diagonals lengths observed in the recurence plot.
    determinism_rate :
        The percentage of determinism in the time series.
    shan_entr :
        Shannon information entropy.

    .. warning:: The recurrence plots results does not reproduce what is obtained using
       Kubios (3.5.0) and should be used with caution for now.

    See also
    --------
    nonlinear_domain, poincare

    References
    ----------
    .. [1] H. Dabire, D. Mestivier, J. Jarnet, M.E. Safar, and N. Phong Chau.
       Quantification of sympathetic and parasympathetic tones by nonlinear indexes in
       normotensive rats. amj, 44:H1290–H1297, 1998.

    .. [2] C.L. Webber Jr. and J.P. Zbilut. Dynamical assessment of physiological
       systems and states using recurrence plot strategies. J Appl Physiol, 76:965–973,
       1994.

    .. [3] Zbilut J. P., Webber C. L., Zak M.Quantification of heart rate variability
       using methods derived from nonlinear dynamics.Assessment and Analysis of
       Cardiovascular Function, Drzewiecki G., Li J. K.-J. Springer New York.

    """
    rr = np.asarray(rr)

    if input_type != "rr_ms":
        rr = input_conversion(rr, input_type=input_type, output_type="rr_ms")

    recurrence_rate, l_max, l_mean, determinism_rate, shan_entr = _recurrence(rr)

    return recurrence_rate, l_max, l_mean, determinism_rate, shan_entr


def _recurrence(
    rr: np.ndarray, m: int = 10, l_min: int = 2
) -> Tuple[float, int, float, float, float]:
    """Compute recurrence scores"""

    # Recurrence matrix
    rc = recurrence_matrix(rr)

    # Compute the recurrence rate - Exclude the main identity line
    j = rc.shape[0]
    recurrence_rate = np.triu(rc).sum() / ((j**2 - j) / 2) * 100

    # Find diagonale lines
    total_lines: List[int] = []
    for i in range(1, rc.shape[0] // 2):
        # All diagonals except the main one
        diag = np.diagonal(rc, offset=i)

        # Lenght of each diagonale found with consecutive `True` values
        d = np.diff(
            np.where(np.concatenate(([diag[0]], diag[:-1] != diag[1:], [True])))[0]
        )[::2]

        # Store the result if any
        if d.shape[0] > 0:
            total_lines.extend(d)

    # Compute scores
    l_max = np.max(total_lines)

    # Diagonales from upper and lower triangle
    l_lines = np.asarray(total_lines).repeat(2)

    # Exclude small digonales (< l_min, default to 2)
    l_lines = l_lines[np.where(l_lines > l_min)[0]]

    # Average length of diagonales
    l_mean = l_lines.mean()

    # Determinism - Do not include the main diagonale
    determinism_rate = (l_lines.sum() / (np.sum(rc) - j)) * 100

    # Shannon information entropy
    _, counts = np.unique(l_lines, return_counts=True)
    shan_entr = -(np.log(counts / len(l_lines)) * (counts / len(l_lines))).sum()

    return (
        float(recurrence_rate),
        int(l_max),
        float(l_mean),
        float(determinism_rate),
        float(shan_entr),
    )


def recurrence_matrix(rr: np.ndarray, m: int = 10, tau: int = 1) -> np.ndarray:
    """Compute the recurrence matrix from an array of RR intervals [1]_.

    Parameters
    ----------
    rr :
        R-R interval time-series. Can be in seconds or miliseconds.
    m :
        The embedding dimension. This corresponds to the length of the subsamples.
        Defaults to `10`.
    tau :
        The embedding lag. This corresponds to the number of datapoints that are skipped
        when creating the sub-sample. Defaults to `1` (take all values).

    Returns
    -------
    rc :
        The recurrence matrix.

    References
    ----------
    .. [1] H. Dabire, D. Mestivier, J. Jarnet, M.E. Safar, and N. Phong Chau.
       Quantification of sympathetic and parasympathetic tones by nonlinear indexes in
       normotensive rats. amj, 44:H1290–H1297, 1998.

    """
    r = np.sqrt(m) * np.std(rr)  # Threshold for the Euclidean distance
    lag = (m - 1) * tau  # Lag
    N = rr.shape[0]  # Size of the input signal
    j = N - lag  # Dimension of the recurrence matrix

    Y = np.zeros((m, j))  # Initialize the time embedding matrix

    # Create a 2d array with segments of the signal lagged
    # according to tau and m (the embedding dimension)
    for i in range(m):
        k = i * tau
        Y[i] = rr[k : k + j]
    embedded = Y.T

    # Compute Euclidean distance
    d = cdist(embedded, embedded, metric="euclidean")

    # Initialize the recurrence matrix filled with 0s
    rc = np.zeros((j, j))

    # If lower or equal to threshold, then 1
    rc[d <= r] = 1

    return rc


def all_domain(rr: Union[List, np.ndarray], input_type: str = "rr_ms") -> pd.DataFrame:
    """Extract all the HRV indices implemented for the time domain, frequency domain
    and linear domain.

    Parameters
    ----------
    rr :
        R-R interval time-series, peaks or peaks index vectors. The default expected
        vector is R-R intervals in milliseconds. Other data format can be provided by
        specifying the `"input_type"` (can be `"rr_s"`, `"peaks"` or `"peaks_idx"`).
    input_type :
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.

    Returns
    -------
    stats :
        Summary of the HRV indices extracted.
        * 'MeanRR' : Mean of R-R intervals (ms).
        * 'MeanBPM' : Mean of beats per minutes (bpm).
        * 'MedianRR' : Median of R-R intervals' (ms).
        * 'MedianBPM' : Median of beats per minutes (bpm).
        * 'MinRR' : Minimum R-R intervals (ms).
        * 'MinBPM' : Minimum beats per minutes (bpm).
        * 'MaxRR' : Maximum R-R intervals (ms).
        * 'MaxBPM' : Maximum beats per minutes (bpm).
        * 'SDNN' : Standard deviation of RR intervals (ms).
        * 'SDSD' : Standard deviation of the Successive difference (ms).
        * 'RMSSD' : Root Mean Square of the Successive Differences (ms).
        * 'nn50' : number of successive differences larger than 50ms (count).
        * 'pnn50' : Proportion of successive difference larger than 50ms (%).
        * 'vlf_peak' : Very low frequency peak (HZ).
        * 'vlf_power' : Very low frequency power (ms²).
        * 'lf_peak' : Low frquency peak (Hz).
        * 'lf_power' : Low frequency power (ms²).
        * 'hf_peak' : High frequency peak (Hz).
        * 'hf_power' : High frequency power (ms²).
        * 'vlf_power_per' : Very low frequency power (%).
        * 'lf_power_per' : Low frequency power (%).
        * 'hf_power_per' : High frequency power (%).
        * 'lf_power_nu' : Low frequency power (normalized units).
        * 'hf_power_nu' : High frequency power (normalized units).
        * 'total_power' : Total frequency power (ms²).
        * 'lf_hf_ratio' : Low / high frequency ratio (normalized units).
        * 'SD1' : SD1, the standard deviation of the poincare plot orthogonal to the identity line (ms).
        * 'SD2' : SD2, the standard deviation of the poincare plot along the identity line (ms).
        * 'recurrence_rate' : The recurrence rate in the recurrence plot (%).
        * 'l_max' : The maximun diagonal length in the recurrence plot (beats).
        * 'l_mean' : The mean diagonal length in the recurrence plot (beats).
        * 'determinism_rate' : The determinism rate in the recurrence plot (%).
        * 'shannon_entropy' : The Shannon entropy.

    See also
    --------
    time_domain, frequency_domain, nonlinear_domain

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

    time_df = time_domain(rr, input_type="rr_ms")
    frequency_df = frequency_domain(rr, input_type="rr_ms")
    nonlinear_df = nonlinear_domain(rr, input_type="rr_ms")

    return pd.concat([time_df, frequency_df, nonlinear_df])
