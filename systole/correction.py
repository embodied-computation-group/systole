# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numba import jit
from scipy.stats import norm

from systole.detection import rr_artefacts
from systole.utils import input_conversion


@jit(nopython=True)
def correct_extra_rr(
    rr: np.ndarray,
    extra_idx: np.ndarray,
    artefacts: np.ndarray = np.array([[], []], dtype=np.bool_),
) -> Tuple[np.ndarray, np.ndarray]:
    """Correct extra heartbeat(s) by removing the RR interval(s). Will also update the
    artefacts array accordingly if this one is provided.

    Parameters
    ----------
    rr :
        1d array of RR intervals (ms).
    extra_idx :
        1d array of the index(es) of the extra RR interval(s).
    artefacts :
        A numpy array (artefacts x time) of artefacts logs, build form the dictionary
        returned by py:`func:systole.detection.rr_artefacts()`. This array will be
        updated as intervals are removed.

    Returns
    -------
    clean_rr :
        Corrected RR intervals.
    artefacts :
        The updated dictionary of artefacts logs originally returned by
        py:`func:systole.detection.rr_artefacts()`.

    """

    # Set defaults values for the outputs
    clean_rr, new_artefacts = rr.copy(), artefacts.copy()

    for i in range(len(extra_idx)):
        # Which RR interval should be corrected
        idx = extra_idx[i]

        # If this is the last interval in the time series the interval is removed as we
        # cannot merge its value with the next one, otherwise transfer this value
        if idx < len(clean_rr) - 1:
            # Transfer the extra time to the next interval
            clean_rr[idx + 1] = clean_rr[idx + 1] + clean_rr[idx]

        # Remove the extra interval
        clean_rr = np.delete(clean_rr, idx)

        # If the artefact array was provided, update it accordingly by removing a col.
        if artefacts.any():
            new_artefacts = np.zeros(shape=(5, artefacts.shape[1] - 1), dtype=np.bool_)
            for i in range(5):
                new_artefacts[i, :] = np.delete(artefacts[i, :], idx)
            artefacts = new_artefacts.copy()

        # Update the artifacts indexes accordingly
        # Here we have removed a RR interval
        extra_idx = extra_idx - 1

    return clean_rr, new_artefacts


@jit(nopython=True)
def correct_missed_rr(
    rr: np.ndarray,
    missed_idx: np.ndarray,
    artefacts: np.ndarray = np.array([[], []], dtype=np.bool_),
) -> Tuple[np.ndarray, np.ndarray]:
    """Correct missed heartbeat(s) by adding new RR intervals. Will also update the
    artefacts array accordingly if this one is provided.

    Parameters
    ----------
    rr :
        RR intervals (ms).
    missed_idx :
        Index of the extra RR interval.
    artefacts :
        A numpy array (artefacts x time) of artefacts logs, build form the dictionary
        returned by py:`func:systole.detection.rr_artefacts()`. This array will be
        updated as intervals are added.

    Returns
    -------
    clean_rr :
        Corrected RR intervals.
    artefacts :
        If the artefacts dictionary was provided, also return the updated version of
        this dictionary with the new artifacts idexes.

    """

    # Set defaults values for the outputs
    clean_rr, new_artefacts = rr.copy(), artefacts.copy()

    for i in range(len(missed_idx)):
        # Which RR interval should be corrected
        idx = missed_idx[i]

        # Divide this interval by 2
        clean_rr[idx] /= 2

        # Add a another interval
        clean_rr = np.concatenate(
            (clean_rr[:idx], np.array([clean_rr[idx]]), clean_rr[idx:])
        )

        # If the artefact array was provided, update it accordingly by adding a col.
        if artefacts.any():
            new_artefacts = np.zeros(shape=(5, artefacts.shape[1] + 1), dtype=np.bool_)
            for i in range(5):
                new_artefacts[i, :] = np.concatenate(
                    (
                        artefacts[i, :][:idx],
                        np.array([False], dtype=np.bool_),
                        artefacts[i, :][idx:],
                    )
                )
            artefacts = new_artefacts.copy()

        # Update the artifacts indexes accordingly
        # Here we have added a new RR interval
        missed_idx = missed_idx + 1

    return clean_rr, new_artefacts


@jit(nopython=True)
def interpolate_rr(rr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Correct long or short beat(s) using linear interpolation.

    Parameters
    ----------
    rr :
        RR intervals (ms).
    idx : int, np.ndarray
        Index of the RR interval that should be interpolated.

    Returns
    -------
    clean_rr :
        The corrected RR intervals.

    """

    # Create time vector
    time = np.arange(0, len(rr))

    # Correction of artefacts
    clean_rr = np.interp(time, np.delete(time, idx), np.delete(rr, idx))

    return clean_rr


@jit(nopython=True)
def _correct_rr(
    rr: np.ndarray,
    artefacts: np.ndarray,
    extra_correction: bool = True,
    missed_correction: bool = True,
    short_correction: bool = True,
    long_correction: bool = True,
    ectopic_correction: bool = True,
    verbose: bool = True,
) -> Tuple[
    np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """Internal jited function to correct RR artefacts from artefacts dictionary."""

    if verbose:
        print("Cleaning the RR interval time series.")

    clean_rr = rr.copy()

    # Correct missed beats
    if missed_correction:
        nMissed = np.sum(artefacts[0, :])
        if nMissed > 0:
            # Correct the missed artefacts and update the artefacts dictonary
            clean_rr, artefacts = correct_missed_rr(
                rr=clean_rr,
                missed_idx=np.where(artefacts[0, :])[0],
                artefacts=artefacts.copy(),
            )

            if verbose:
                print(f"... correcting {nMissed} missed interval(s).")

    # Correct extra beats
    if extra_correction:
        nExtra = np.sum(artefacts[1, :])
        if nExtra > 0:
            # Correct the extra artefacts and update the artefacts dictonary
            clean_rr, artefacts = correct_extra_rr(
                rr=clean_rr,
                extra_idx=np.where(artefacts[1, :])[0],
                artefacts=artefacts.copy(),
            )

            if verbose:
                print(f"... correcting {nExtra} extra interval(s).")

    # Correct for ectopic, short and long heartbeats beats - If multiple corrections
    # are required, interpolate everything in a single pass to avoid interpolating from
    # corrupted RR intervals.
    if sum([ectopic_correction, short_correction, long_correction]) > 0:
        # Create a boolean vector of correction that will be updated accordingly
        correction_vector = np.zeros(artefacts.shape[1], dtype=np.bool_)

        if ectopic_correction:
            nEctopic = np.sum(artefacts[2, :])
            if nEctopic > 0:
                # Also correct the heartbeat before - ectopic beats are preceeded by short
                # or long intervals. Here we automatically correct for both, and mark those
                # interval as corrected.
                for i in np.where(artefacts[2, :])[0]:
                    if (i > 0) & (i < len(artefacts[2, :])):
                        artefacts[2, :][i - 1] = True
                        artefacts[3, :][i - 1] = False
                        artefacts[4, :][i - 1] = False

                assert len(correction_vector) == len(artefacts[2, :])
                correction_vector = correction_vector | artefacts[2, :]

                if verbose:
                    print(f"... correcting {nEctopic} ectopic interval(s).")

        if short_correction:
            nShort = np.sum(artefacts[3, :])
            if nShort > 0:
                assert len(correction_vector) == len(artefacts[3, :])
                correction_vector = correction_vector | artefacts[3, :]

                if verbose:
                    print(f"... correcting {nShort} short interval(s).")

        if long_correction:
            nLong = np.sum(artefacts[4, :])
            if nLong > 0:
                assert len(correction_vector) == len(artefacts[4, :])
                correction_vector = correction_vector | artefacts[4, :]

                if verbose:
                    print(f"... correcting {nLong} long interval(s).")

        # Interpolate the interval marked as incorrect (short, long and/or ectopic)
        assert len(correction_vector) == len(clean_rr)
        clean_rr = interpolate_rr(rr=clean_rr, idx=np.where(correction_vector)[0])

    return clean_rr, (nMissed, nExtra, nEctopic, nShort, nLong)


def correct_rr(
    rr: Union[List, np.ndarray],
    extra_correction: bool = True,
    missed_correction: bool = True,
    short_correction: bool = True,
    long_correction: bool = True,
    ectopic_correction: bool = True,
    input_type: str = "rr_ms",
    verbose: bool = True,
) -> Tuple[
    np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """Correct artefacts in RR time series using the method described in [1]_.

    Parameters
    ----------
    rr :
        RR intervals (expressed in seconds or miliseconds), boolean peaks vector or
        peaks indexes. The function will automatically convert the input into RR
        intervals (ms).
    extra_correction :
      If `True` (deault), correct extra beats in the RR time series.
    missed_correction :
      If `True` (deault), correct missed beats in the RR time series.
    short_correction :
      If `True` (deault), correct short beats in the RR time series.
    long_correction :
      If `True` (deault), correct long beats in the RR time series.
    ectopic_correction :
      If `True` (deault), correct ectopic beats in the RR time series.
    n_iterations :
        Number of iterations (artifacts detection and artifacts correction). Defaults
        to `1`.
    input_type :
        The type of input vector. Default is `"rr_ms"` (a boolean vector where
        `1` represents the occurrence of R waves or systolic peaks).
        Can also be `"peaks_idx"`, the idexs of samples where a peaks is detected,
        `"peaks"`, or `"rr_s"` for vectors of RR intervals expressed in seconds.
    verbose :
        Control the verbosity of the function. Defaults to `True`.

    Returns
    -------
    clean_rr, (nMissed, nExtra, nEctopic, nShort, nLong) :
        The corrected RR time series and the number of artefacts corrected.

    Examples
    --------
    >>> from systole import import_rr
    >>> from systole.correction import correct_rr

    >>> # Load an example RR time series
    >>> rr = import_rr().rr

    >>> corrected_rr, (nMissed, nExtra, nEctopic, nShort, nLong) = correct_rr(rr)
    Cleaning the RR interval time series.
    ... correcting 1 ectopic interval(s).
    ... correcting 1 long interval(s).

    Notes
    -----
    This function will correct artifacts in RR intervals time series (ms) following the
    method presented in [1]_. First, artifacts are labelled using
    :py:func:`systole.detection.rr_artefacts`. Then, artifacts are corrected in the
    following order:

    #. Missed heartbeats (add one heartbeat).
    #. Extra heartbeats (remove one heartbeat).
    #. Ectopic heartbeats (interpolate the values of the two heartbeats).
    #. Short heartbeats (interpolate the value of the heartbeat).
    #. Long heartbeats (interpolate the value of the heartbeat).

    These steps can be repeated by changing the `n_iterations` parameter.

    When adding or removing an RR interval (missed and extra artefacts), the `artefacts`
    arrays are updated accordingly.

    See also
    --------
    correct_peaks

    References
    ----------
    .. [1] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for heart
        rate variability time series artefact correction using novel beat
        classification. Journal of Medical Engineering & Technology, 43(3), 173-181.
        https://doi.org/10.1080/03091902.2019.1640306

    """
    rr = np.asarray(rr)

    if input_type != "rr_ms":
        if input_type not in ["peaks", "rr_s", "peaks_idx"]:
            raise ValueError("Invalid input type")
        else:
            rr = input_conversion(rr, input_type=input_type, output_type="rr_ms")

    # Artefacts detection
    artefacts_dict = rr_artefacts(rr)

    # Create the array from dict
    artefacts = np.array(
        [
            artefacts_dict["missed"],
            artefacts_dict["extra"],
            artefacts_dict["ectopic"],
            artefacts_dict["short"],
            artefacts_dict["long"],
        ],
        dtype=bool,
    )

    return _correct_rr(
        rr=rr,
        artefacts=artefacts,
        missed_correction=missed_correction,
        extra_correction=extra_correction,
        ectopic_correction=ectopic_correction,
        short_correction=short_correction,
        long_correction=long_correction,
        verbose=verbose,
    )


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
    peaks :
        Boolean vector of peaks.
    input_type :
        The type of input vector. Defaults to `"rr_ms"` for vectors of RR intervals, or
        interbeat intervals (IBI), expressed in milliseconds. Can also be a boolean
        vector where `1` represents the occurrence of R waves or systolic peakspeaks
        vector `"rr_s"` or IBI expressed in seconds.
    extra_correction :
      If `True` (default), correct extra peaks in the peaks time series.
    missed_correction :
      If `True` (default), correct missed peaks in the peaks time series.
    n_iterations :
        How many time to repeat the detection-correction process. Defaults to `1`.
    verbose :
        Control the verbosity of the function. Defaults to `True`.

    Returns
    -------
    correction :
        The corrected RR time series and the number of artefacts corrected:
        - clean_peaks: The corrected boolean time-serie.
        - extra: The number of extra beats corrected.
        - missed: The number of missed beats corrected.

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
    peaks :
        Boolean vector of peaks detection.
    idx : int
        Index of the peaks corresponding to the missed RR interval. The new peaks will
        be placed between this one and the previous one.

    Returns
    -------
    clean_peaks :
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
    peaks :
        Boolean vector of peaks detection.
    idx :
        Index (sample number) of the peaks corresponding to the etopic interval
        (i.e. the long interval following the short interval).
    signal :
        (Optional) The raw ECG or PPG signal. If provided, the pseudo-ectopic
        interval is re-estimated using this signal.

    Returns
    -------
    clean_peaks :
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
