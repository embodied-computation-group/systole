# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Optional, Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm

from systole.detection import rr_artefacts
from systole.utils import input_conversion


def correct_extra(rr: Union[List, np.ndarray], idx: int) -> np.ndarray:
    """Correct extra beats by removing the RR interval.

    Parameters
    ----------
    rr : np.ndarray | list
        RR intervals.
    idx : int
        Index of the extra RR interval.

    Returns
    -------
    clean_rr : np.ndarray
        Corrected RR intervals.

    """

    rr = np.asarray(rr)

    clean_rr = rr

    # If this is the last interval in the time series
    if idx == len(clean_rr):
        clean_rr = np.delete(clean_rr, idx - 1)
    else:
        # Transfer the extra time to the next interval
        clean_rr[idx + 1] = clean_rr[idx + 1] + clean_rr[idx]
        # Remove the extra interval
        clean_rr = np.delete(clean_rr, idx)

    return clean_rr


def correct_missed(rr: Union[List, np.ndarray], idx: int) -> np.ndarray:
    """Correct missed beats by adding a new RR interval.

    Parameters
    ----------
    rr : np.ndarray | list
        RR intervals.
    idx : int
        Index of the missed RR interval.

    Returns
    -------
    clean_rr : np.ndarray
        Corrected RR intervals.
    """

    rr = np.asarray(rr)

    clean_rr = rr

    # Divide current interval by 2
    clean_rr[idx] /= 2

    # Add a second interval
    clean_rr = np.insert(clean_rr, idx, clean_rr[idx])

    return clean_rr


def interpolate_bads(
    rr: Union[List, np.ndarray], idx: Union[int, List, np.ndarray]
) -> np.ndarray:
    """Correct long and short beats using interpolation.

    Parameters
    ----------
    rr : np.ndarray or list
        RR intervals (ms).
    idx : int, np.ndarray or list
        Index of the RR interval to correct.

    Returns
    -------
    clean_rr : np.ndarray
        Corrected RR intervals.
    """
    if isinstance(rr, list):
        rr = np.asarray(rr)

    x = np.arange(0, len(rr))

    # Correction of artefacts
    f = interp1d(np.delete(x, idx), np.delete(rr, idx))
    clean_rr = f(x)

    return clean_rr


def correct_rr(
    rr: Union[List, np.ndarray],
    extra_correction: bool = True,
    missed_correction: bool = True,
    short_correction: bool = True,
    long_correction: bool = True,
    ectopic_correction: bool = True,
    n_iterations: int = 2,
    input_type: str = "rr_ms",
    verbose: bool = True,
) -> Dict[str, Union[int, np.ndarray]]:
    """Correct long and short beats using interpolation.

    Parameters
    ----------
    rr : np.ndarray
        RR intervals (ms).
    extra_correction : bool
      If `True` (deault), correct extra beats in the RR time series.
    missed_correction : bool
      If `True` (deault), correct missed beats in the RR time series.
    short_correction : bool
      If `True` (deault), correct short beats in the RR time series.
    long_correction : bool
      If `True` (deault), correct long beats in the RR time series.
    ectopic_correction : bool
      If `True` (deault), correct ectopic beats in the RR time series.
    n_iterations : int
        How many time to repeat the detection-correction process. Defaults to `2`.
    input_type : str
        The type of input vector. Default is `"rr_ms"` (a boolean vector where
        `1` represents the occurrence of R waves or systolic peaks).
        Can also be `"peaks_idx"`, the idexs of samples where a peaks is detected,
        `"peaks"`, or `"rr_s"` for vectors of RR intervals expressed in seconds.
    verbose : bool
        Control the verbosity of the function. Defaults to `True`.

    Returns
    -------
    correction : dictionary
        The corrected RR time series and the number of artefacts corrected:

        * clean_rr: np.ndarray
            The corrected RR time-serie.
        * ectopic: int
            The number of ectopic beats corrected.
        * short: int
            The number of short beats corrected.
        * long: int
            The number of long beats corrcted.
        * extra: int
            The number of extra beats corrected.
        * missed: int
            The number of missed beats corrected.

    See also
    --------
    correct_peaks

    """
    rr = np.asarray(rr)

    if input_type != "rr_ms":
        if input_type not in ["peaks", "rr_s", "peaks_idx"]:
            raise ValueError("Invalid input type")
        else:
            rr = input_conversion(rr, input_type=input_type, output_type="rr_ms")

    clean_rr = rr.copy()

    artefacts = rr_artefacts(clean_rr)

    if verbose:
        print(f"Cleaning the RR interval time series using {n_iterations} iterations.")

    nEctopic, nShort, nLong, nExtra, nMissed = 0, 0, 0, 0, 0

    # Loop across n_iterations and perform the requested corrections
    for n_it in range(n_iterations):

        if verbose:
            print(f" - Iteration {n_it+1} - ")

        # Correct missed beats
        if missed_correction:
            if np.any(artefacts["missed"]):
                this_nmissed = 0  # How many corrected missed beat this iteration
                for this_id in np.where(artefacts["missed"])[0]:
                    this_id += (
                        this_nmissed  # Adjust idx according to previous corrections
                    )
                    clean_rr = correct_missed(clean_rr, this_id)
                    this_nmissed += 1
                if verbose:
                    print(f"... correcting {this_nmissed} missed interval(s).")
                nMissed += this_nmissed  # Update the total of corrected missed beats
                artefacts = rr_artefacts(clean_rr)

        # Correct extra beats
        if extra_correction:
            if np.any(artefacts["extra"]):
                this_nextra = 0  # How many corrected extra beat this iteration
                for this_id in np.where(artefacts["extra"])[0]:
                    this_id -= (
                        this_nextra  # Adjust idx according to previous corrections
                    )
                    clean_rr = correct_extra(clean_rr, this_id)
                    this_nextra += 1
                if verbose:
                    print(f"... correcting {this_nextra} extra interval(s).")
                nExtra += this_nextra  # Update the total of corrected missed beats
                artefacts = rr_artefacts(clean_rr)

        # Correct ectopic beats
        if ectopic_correction:
            if np.any(artefacts["ectopic"]):
                # Also correct the beat before
                for i in np.where(artefacts["ectopic"])[0]:
                    if (i > 0) & (i < len(artefacts["ectopic"])):
                        artefacts["ectopic"][i - 1] = True
                this_id = np.where(artefacts["ectopic"])[0]
                clean_rr = interpolate_bads(clean_rr, [this_id])
                this_nectopic = int(np.sum(artefacts["ectopic"]))
                if verbose:
                    print(f"... correcting {this_nectopic} ectopic interval(s).")
                artefacts = rr_artefacts(clean_rr)
                nEctopic += (
                    this_nectopic  # Update the total number of corrected ectopic beats
                )

        # Correct short beats
        if short_correction:
            if np.any(artefacts["short"]):
                this_nshort = 0  # How many corrected short beat this iteration
                this_id = np.where(artefacts["short"])[0]
                clean_rr = interpolate_bads(clean_rr, this_id)
                this_nshort += this_id.shape[0]
                if verbose:
                    print(f"... correcting {this_nshort} short interval(s).")
                artefacts = rr_artefacts(clean_rr)
                nShort += (
                    this_nshort  # Update the total number of corrected short beats
                )

        # Correct long beats
        if long_correction:
            if np.any(artefacts["long"]):
                this_nlong = 0
                this_id = np.where(artefacts["long"])[0]
                clean_rr = interpolate_bads(clean_rr, this_id)
                this_nlong += this_id.shape[0]
                if verbose:
                    print(f"... correcting {this_nlong} long interval(s).")
                artefacts = rr_artefacts(clean_rr)
                nLong += this_nlong  # Update the total number of corrected long beats

    return {
        "clean_rr": clean_rr,
        "ectopic": nEctopic,
        "short": nShort,
        "long": nLong,
        "extra": nExtra,
        "missed": nMissed,
    }


def correct_peaks(
    peaks: Union[List, np.ndarray],
    input_type: str = "peaks",
    extra_correction: bool = True,
    missed_correction: bool = True,
    n_iterations: int = 1,
    verbose: bool = True,
) -> Dict[str, Union[int, np.ndarray]]:
    """Correct long, short, extra, missed and ectopic beats in peaks vector.

    Parameters
    ----------
    peaks : np.ndarray
        Boolean vector of peaks.
    input_type : str
            The type of input vector. Defaults to `"rr_ms"` for vectors of RR
            intervals, or  interbeat intervals (IBI), expressed in milliseconds.
            Can also be a boolean vector where `1` represents the occurrence of
            R waves or systolic peakspeaks vector `"rr_s"` or IBI expressed in
            seconds.
    extra_correction : bool
      If `True` (default), correct extra peaks in the peaks time series.
    missed_correction : bool
      If `True` (default), correct missed peaks in the peaks time series.
    n_iterations : int
        How many time to repeat the detection-correction process. Defaults to `1`.
    verbose : bool
        Control the verbosity of the function. Defaults to `True`.

    Returns
    -------
    correction : dictionary
        The corrected RR time series and the number of artefacts corrected:

        * clean_peaks: np.ndarray
            The corrected boolean time-serie.
        * extra: int
            The number of extra beats corrected.
        * missed: int
            The number of missed beats corrected.

    See also
    --------
    correct_rr

    Notes
    -----
    This function wil operate at the `peaks` vector level to keep the length of the
    signal constant after peaks correction.

    """

    peaks = np.asarray(peaks)

    if input_type != "peaks":
        peaks = input_conversion(peaks, input_type, output_type="peaks")

    clean_peaks = peaks.copy()
    nExtra, nMissed = 0, 0

    if verbose:
        print(f"Cleaning the peaks vector using {n_iterations} iterations.")

    for n_it in range(n_iterations):

        if verbose:
            print(f" - Iteration {n_it+1} - ")

        # Correct extra peaks
        if extra_correction:

            # Artefact detection
            artefacts = rr_artefacts(clean_peaks, input_type="peaks")

            if np.any(artefacts["extra"]):

                peaks_idx = np.where(clean_peaks)[0][1:]

                # Convert the RR interval idx to sample idx
                extra_idx = peaks_idx[np.where(artefacts["extra"])[0]]

                # Number of extra peaks to correct
                this_nextra = int(artefacts["extra"].sum())
                if verbose:
                    print(f"... correcting {this_nextra} extra peak(s).")

                nExtra += this_nextra

                # Removing peak n+1 to correct RR interval n
                clean_peaks[extra_idx] = False

                artefacts = rr_artefacts(clean_peaks, input_type="peaks")

        # Correct missed peaks
        if missed_correction:
            if np.any(artefacts["missed"]):

                peaks_idx = np.where(clean_peaks)[0][1:]

                # Convert the RR interval idx to sample idx
                missed_idx = peaks_idx[np.where(artefacts["missed"])[0]]

                # Number of missed peaks to correct
                this_nmissed = int(artefacts["missed"].sum())
                if verbose:
                    print(f"... correcting {this_nmissed} missed peak(s).")

                nMissed += this_nmissed

                # Correct missed peaks using sample index
                for this_idx in missed_idx:
                    clean_peaks = correct_missed_peaks(clean_peaks, this_idx)

                # Artefact detection
                artefacts = rr_artefacts(clean_peaks, input_type="peaks")

    return {
        "clean_peaks": clean_peaks,
        "extra": nExtra,
        "missed": nMissed,
    }


def correct_missed_peaks(peaks: Union[List, np.ndarray], idx: int) -> np.ndarray:
    """Correct missed peaks in boolean peak vector.

    The new peak is placed in the middle of the previous interval.

    Parameters
    ----------
    peaks : np.ndarray | list
        Boolean vector of peaks detection.
    idx : int
        Index of the peaks corresponding to the missed RR interval. The new peaks will
        be placed between this one and the previous one.

    Returns
    -------
    clean_peaks : np.ndarray
        Corrected boolean vector of peaks.
    """
    peaks = np.asarray(peaks, dtype=bool)

    if not peaks[idx]:
        raise (ValueError("The index provided does not match with a peaks."))

    clean_peaks = peaks.copy()

    # The index of the previous peak
    previous_idx = np.where(clean_peaks[:idx])[0][-1]

    # Estimate new interval
    interval = int((idx - previous_idx) / 2)

    # Add peak in vector
    clean_peaks[previous_idx + interval] = True

    return clean_peaks


def correct_ectopic_peaks(
    peaks: Union[List, np.ndarray],
    idx: int,
    signal: Optional[Union[List, np.ndarray]] = None,
) -> np.ndarray:
    """Correct pseudo ectopic heartbeat in boolean peak vector.

    Parameters
    ----------
    peaks : np.ndarray | list
        Boolean vector of peaks detection.
    idx : int
        Index (sample number) of the peaks corresponding to the etopic interval
        (i.e. the long interval following the short interval).
    signal : np.ndarray | list | None
        (Optional) The raw ECG or PPG signal. If provided, the pseudo-ectopic
        interval is re-estimated using this signal.

    Returns
    -------
    clean_peaks : np.ndarray
        Corrected boolean vector of peaks.

    Notes
    -----
    This function aims to correct misdetection or R wave (e.g. in the T wave) that
    are labelled as ectopic beats by the artefact detection algorithm, in the form
    of a short interval followed by a long interval. If the raw (ECG or PPG) signal
    is provided, the most probable real peak location will be re-estimated using this
    signal.

    Raises
    ------
    ValueError
        If the artefact index is outside the range of the peaks vector.

    """

    peaks = np.asarray(peaks, dtype=bool)
    clean_peaks = peaks.copy()

    if not peaks[idx]:
        raise (ValueError("The index provided does not match with a peaks."))

    # Position of artefact in the peaks time series
    n = len(np.where(peaks[:idx])[0])

    idx_1 = np.where(peaks)[0][n - 1]
    idx_2 = np.where(peaks)[0][n - 2]

    # Remove the intermediate peak
    clean_peaks[idx_1] = False

    if signal is not None:

        signal = np.asarray(signal)

        # Extract the signal of interest (n-2 -> n)
        sub_signal = signal[idx_2:idx]

        # Evidence for amplitude
        amp = sub_signal - sub_signal.min()
        amp = amp / amp.max()

        # Evidence for centrality
        cntr = norm.pdf(
            np.arange(0, len(sub_signal)),
            loc=len(sub_signal) / 2,
            scale=len(sub_signal) / 6,
        )

        new_idx = idx_2 + np.argmax(amp * cntr)
        clean_peaks[new_idx] = True

    else:

        # Estimate new interval
        interval = int((idx - idx_2) / 2)

        # Add peak in vector
        clean_peaks[idx_2 + interval] = True

    return clean_peaks
