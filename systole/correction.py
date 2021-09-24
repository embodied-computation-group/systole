# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Union

import numpy as np
from scipy.interpolate import interp1d

from systole.detection import rr_artefacts
from systole.utils import input_conversion


def correct_extra(rr: Union[List, np.ndarray], idx: int) -> np.ndarray:
    """Correct extra beats by removing the RR interval.

    Parameters
    ----------
    rr : np.ndarray or list
        RR intervals.
    idx : int
        Index of the extra RR interval.

    Returns
    -------
    clean_rr : np.ndarray
        Corrected RR intervals.
    """
    if isinstance(rr, list):
        rr = np.asarray(rr)

    clean_rr = rr

    if idx == len(clean_rr):
        clean_rr = np.delete(clean_rr, idx - 1)
    else:
        # Add the extra interval to the next one
        clean_rr[idx + 1] = clean_rr[idx + 1] + clean_rr[idx]
        # Remove current interval
        clean_rr = np.delete(clean_rr, idx)

    return clean_rr


def correct_missed(rr: Union[List, np.ndarray], idx: int) -> np.ndarray:
    """Correct missed beats by adding a new RR interval.

    Parameters
    ----------
    rr : np.ndarray or list
        RR intervals.
    idx : int
        Index of the missed RR interval.

    Returns
    -------
    clean_rr : np.ndarray
        Corrected RR intervals.
    """
    if isinstance(rr, list):
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
) -> Dict[str, Union[int, np.ndarray]]:
    """Correct long and short beats using interpolation.

    Parameters
    ----------
    rr : np.ndarray
        RR intervals (ms).
    correct_extra : boolean
      If `True`, correct extra beats in the RR time series.
    correct_missed : boolean
      If `True`, correct missed beats in the RR time series.
    correct_short : boolean
      If `True`, correct short beats in the RR time series.
    correct_long : boolean
      If `True`, correct long beats in the RR time series.
    correct_ectopic : boolean
      If `True`, correct ectopic beats in the RR time series.

    Returns
    -------
    correction : dictionnary
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
    """
    rr = np.asarray(rr)

    clean_rr = rr.copy()
    nEctopic, nShort, nLong, nExtra, nMissed = 0, 0, 0, 0, 0

    artefacts = rr_artefacts(clean_rr)

    # Correct missed beats
    if missed_correction:
        if np.any(artefacts["missed"]):
            for this_id in np.where(artefacts["missed"])[0]:
                this_id += nMissed
                clean_rr = correct_missed(clean_rr, this_id)
                nMissed += 1
        artefacts = rr_artefacts(clean_rr)

    # Correct extra beats
    if extra_correction:
        if np.any(artefacts["extra"]):
            for this_id in np.where(artefacts["extra"])[0]:
                this_id -= nExtra
                clean_rr = correct_extra(clean_rr, this_id)
                nExtra += 1
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
            nEctopic = np.sum(artefacts["ectopic"])  # type: ignore

    # Correct short beats
    if short_correction:
        if np.any(artefacts["short"]):
            this_id = np.where(artefacts["short"])[0]
            clean_rr = interpolate_bads(clean_rr, this_id)
            nShort = len(this_id)

    # Correct long beats
    if long_correction:
        if np.any(artefacts["long"]):
            this_id = np.where(artefacts["long"])[0]
            clean_rr = interpolate_bads(clean_rr, this_id)
            nLong = len(this_id)

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

    Returns
    -------
    correction : dictionnary
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

    artefacts = rr_artefacts(peaks, input_type="peaks")
    peaks_idx = np.where(peaks)[0][1:]

    # Convert the RR interval idx to sample idx
    if extra_correction:
        extra_idx = peaks_idx[np.where(artefacts["extra"])[0]]
    if missed_correction:
        missed_idx = peaks_idx[np.where(artefacts["missed"])[0]]

    # Correct extra beats
    if extra_correction:
        if np.any(artefacts["extra"]):

            # Number of extra peaks to correct
            nExtra = artefacts["extra"].sum()
            print(f"... correcting {artefacts['extra'].sum()} extra peak(s).")

            # Removing peak n+1 to correct RR interval n
            clean_peaks[extra_idx] = False

    # Correct missed beats
    if missed_correction:
        if np.any(artefacts["missed"]):

            # Number of missed peaks to correct
            nMissed = artefacts["missed"].sum()
            print(f"... correcting {artefacts['missed'].sum()} missed peak(s).")

            # Correct missed peaks using sample index
            for this_idx in missed_idx:
                clean_peaks = correct_missed_peaks(clean_peaks, this_idx)

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
    peaks : np.ndarray
        Boolean vector of peaks.
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
