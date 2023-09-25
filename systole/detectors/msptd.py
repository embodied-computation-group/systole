# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
from typing import Tuple, Union

import numpy as np
from numba import jit
from scipy.signal import detrend


def msptd(
    signal: np.ndarray,
    sfreq: int,
    kind: str = "peaks-onsets",
    win_len: int = 6,
    overlap: float = 0.2,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """The Multi-scale peak and trough detection algorithm (MSPTD) to detects
    heartbeats in a photoplethysmogram (PPG) signal.

    Parameters
    ----------
    signal :
        The raw PPG time series.
    win_len :
        Length of the window when splitting the signal (seconds). Defaults to `6.0`.
    overlap :
        Proportion of overlap between consecutive windows. Defaults to `0.2`.
    sfreq :
        The sampling frequency (Hz).
    kind :
        The type of detection to perform. Can be `"peaks"`, `"onsets"` or
        `"peaks-onsets"`. Defaults to `"peaks-onsets"`.

        .. tip:
           Using `"peaks"` and `"onsets"` will skip the non required part of the
           detection process and run faster. This should be prefered when performance
           is a concern.

    Returns
    -------
    Depending on `kind`, will return `peaks`, `onsets` or a tuple `peaks, onsets`.
        peaks :
            Indices of detected pulse peaks.
        onsets :
            Indices of detected pulse onsets (i.e. onsets).

    Examples
    --------
    >>> from systole.detectors import msptd
    >>> from systole import import_ppg
    >>> ppg = import_ppg().ppg.to_numpy()
    Detect both peaks and onsets.
    >>> peaks, onsets = msptd(signal=ppg, sfreq=75, kind="both")
    Only detect peaks.
    >>> peaks = msptd(signal=ppg, sfreq=75, kind="peaks")

    References
    ----------
    .. [1] S. M. Bishop and A. Ercole, 'Multi-scale peak and trough detection optimised
       for periodic and quasi-periodic neuroscience data,' in Intracranial Pressure and
       Neuromonitoring XVI. Acta Neurochirurgica Supplement, T. Heldt, Ed. Springer,
       2018, vol. 126, pp. 189-195. <https://doi.org/10.1007/978-3-319-65798-1_39>

    Acknowledgements
    ----------------
    This function was adapted from the Matlab version provided in the ppg-beats package
    (https://github.com/peterhcharlton/ppg-beats).

    """
    # Window signal
    # -------------

    no_samps_in_win = win_len * sfreq

    if len(signal) <= no_samps_in_win:
        win_starts = np.array(1)
        win_ends = np.array(len(signal))
    else:
        win_offset = round(no_samps_in_win * (1 - overlap))
        win_starts = np.arange(1, len(signal) - no_samps_in_win, win_offset)
        win_ends = win_starts + no_samps_in_win

        # This ensures that the windows include the entire signal duration
        if win_ends[-1] < len(signal):
            win_starts = np.append(win_starts, len(signal) - no_samps_in_win)
            win_ends = np.append(win_ends, len(signal))

    # Downsample signal
    # -----------------

    # Set up downsampling if the sampling frequency is particularly high
    min_fs, do_ds, ds_factor = 20, False, 1
    if sfreq > min_fs:
        ds_factor = np.floor(sfreq / min_fs).astype(int)
        do_ds = True

    # Detect peaks and onsets in each window
    # --------------------------------------
    if "peaks" in kind:
        all_peaks: np.ndarray = np.array([])
        for win_s, win_e in zip(win_starts, win_ends):
            # Extract the window's data
            this_win = signal[win_s:win_e]

            # Downsample signal
            rel_sig = this_win[::ds_factor] if do_ds else this_win

            # Detect peaks
            peaks = msptd_peaks_and_onsets(rel_sig, kind="peaks")

            # Resample indexes
            peaks *= ds_factor
            # Correct peak indices by finding highest point within
            # tolerance either side of detected peaks
            tol = np.ceil(sfreq * 0.05)
            for i, p in enumerate(peaks):
                tol_start = int(p - tol)
                tol_end = int(p + tol)
                temp = np.argmax(this_win[tol_start:tol_end])
                peaks[i] = p - tol + temp - 1
            # Store peaks and onsets
            all_peaks = np.append(all_peaks, peaks + win_s - 1)
        all_peaks = np.sort(np.unique(all_peaks)).astype(int)

    if "onsets" in kind:
        all_onsets: np.ndarray = np.array([])
        for win_s, win_e in zip(win_starts, win_ends):
            # Extract the window's data
            this_win = signal[win_s:win_e]

            # Downsample signal
            rel_sig = this_win[::ds_factor] if do_ds else this_win

            # Detect onsets
            onsets = msptd_peaks_and_onsets(rel_sig, kind="onsets")

            # Resample indexes
            onsets *= ds_factor
            # Correct onset indices by finding highest point within
            # tolerance either side of detected onsets
            tol = np.ceil(sfreq * 0.05)
            for i, t in enumerate(onsets):
                tol_start = int(t - tol)
                tol_end = int(t + tol)
                temp = np.argmin(this_win[tol_start:tol_end])
                onsets[i] = t - tol + temp - 1
            # Store peaks and onsets
            all_onsets = np.append(all_onsets, onsets + win_s - 1)
        all_onsets = np.sort(np.unique(all_onsets)).astype(int)

    # Tidy up detected peaks and onsets
    # (by ordering them and only retaining unique ones)
    if kind == "peaks":
        return all_peaks
    elif kind == "onsets":
        return all_onsets
    else:
        return all_peaks, all_onsets


@jit(nopython=True)
def lms(window_length: int, signal_length: int) -> Tuple:
    """ "Internal jitted function. Returns the indexes of the lms matrices that will be
    tested."""

    valid_idx = np.array(
        [
            (k, i)
            for k in range(window_length)
            for i in range(k + 1, signal_length - k + 1)
        ]
    )

    k = valid_idx[:, 0]
    i = valid_idx[:, 1]

    return k, i


def msptd_peaks_and_onsets(signal: np.ndarray, kind: str = "both"):
    """Internal function. The MSTPD algorithm applied to a windowed sample of the
    recording.

    Parameters
    ----------
    signal :
        The windowed PPG signal.
    kind :
        The type of detection to perform. Can be `"peaks"`, `"onsets"` or `"both"`.
        Defaults to `"both"`.

    Returns
    -------
    p :
        Peaks indexes.
    t :
        Trough indexes.

    """

    signal_length = len(signal)  # Length of the input signal
    window_length = int(np.ceil(signal_length / 2) - 1)  # max window length

    # Step 1 : calculate local maxima and local minima scalograms
    # -----------------------------------------------------------

    # detrend the input signal
    signal = detrend(signal, type="linear")

    # Populate LMS matrices
    k, i = lms(window_length, signal_length)  # Create k and i indices

    if kind in ["both", "peaks"]:
        m_max = np.zeros(
            (window_length, signal_length), dtype=bool
        )  # initialise LMS matrices
        m_max[k, i - 1] = (
            (signal[i - 1] > signal[i - k - 1]) & (signal[i - 1] > signal[i + k - 1])
        ).astype(bool)

        # Step 2 : Find the scale with the most local maxima (or local minima)
        # --------------------------------------------------------------------

        # Row-wise summation (i.e. sum each row)
        gamma_max = np.sum(m_max, axis=1)

        # Find scale with the most local maxima (or local minima)
        lambda_max = np.argmax(gamma_max).astype(int)

        # Step 3 : Use lambda to remove all elements of m for which k > lambda
        # --------------------------------------------------------------------
        m_max = m_max[:lambda_max, :]

        # Step 4 : Find peaks
        # -------------------

        # Column-wise summation
        m_max_sum = np.sum(~m_max, axis=0)
        p = np.where(m_max_sum == 1)[0]

    if kind in ["both", "onsets"]:
        m_min = np.zeros(
            (window_length, signal_length), dtype=bool
        )  # initialise LMS matrices
        m_min[k, i - 1] = (
            (signal[i - 1] < signal[i - k - 1]) & (signal[i - 1] < signal[i + k - 1])
        ).astype(bool)

        # Step 2 : Find the scale with the most local maxima (or local minima)
        # --------------------------------------------------------------------

        # Row-wise summation (i.e. sum each row)
        gamma_min = np.sum(m_min, axis=1)

        # Find scale with the most local maxima (or local minima)
        lambda_min = np.argmax(gamma_min).astype(int)

        # Step 3 : Use lambda to remove all elements of m for which k > lambda
        # --------------------------------------------------------------------
        m_min = m_min[:lambda_min, :]

        # Step 4 : Find peaks
        # -------------------

        # Column-wise summation
        m_min_sum = np.sum(~m_min, axis=0)
        o = np.where(m_min_sum == 1)[0]

    if kind == "both":
        return p, o
    elif kind == "peaks":
        return p
    elif kind == "onsets":
        return o
    else:
        return None
