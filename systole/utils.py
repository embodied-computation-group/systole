# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Optional, Tuple, Union

import numpy as np
from numba import jit
from scipy.interpolate import interp1d

ppg_strings = ["ppg", "pleth", "photoplethysmogram", "cardiac"]
ecg_strings = ["ecg", "electrocardiogram", "ekg", "cardiac"]
resp_strings = ["resp", "rsp", "res", "respiration", "breath"]


@jit(nopython=True)
def norm_triggers(
    triggers: Union[List, np.ndarray],
    threshold: int = 1,
    n: int = 5,
    direction: str = "higher",
) -> np.ndarray:
    """Ceaning noisy triggers into boolean vecor with a unique spike for each event.

    Parameters
    ----------
    triggers :
        The triggers array.
    threshold :
        Threshold for triggering values. Default is 1.
    n :
        The number of values to force to 0 after each triggers. Default is 5.
    direction :
        Indicates if triggers are higher or lower than threshold. Can be`"higher"` or
        `"lower"`. Default sets to `"higher"`.

    Returns
    -------
    y :
        The filterd triggers array.

    """
    triggers = np.asarray(triggers)

    if direction == "higher":
        y = triggers >= threshold
    elif direction == "lower":
        y = triggers <= threshold
    else:
        raise ValueError("Invalid direction")

    # Keep only the first trigger in window size
    for i in range(len(y)):
        if y[i]:
            if (len(y) - i) < n:  # If close to the end
                y[i + 1 :] = False
            else:
                y[i + 1 : i + n + 1] = False
    return y


def time_shift(
    x: Union[List, np.ndarray], events: Union[List, np.ndarray], order: str = "after"
) -> np.ndarray:
    """Return the delay between x and events.

    Parameters
    ----------
    x :
        Timing of the baseline events.
    events :
        Timing of events of interest.
    order :
        Consider event occurung before of after baseline. Default is 'after'.

    Returns
    -------
    time_shift :
        The delay between X and events (a.u).
    """
    if isinstance(x, list):
        x = np.asarray(x)
    if isinstance(events, list):
        events = np.asarray(events)

    lag = []
    for e in events:
        # Find the closest reference before the event of interest
        r = x[x < e].max()
        # Event timing
        lag.append(e - r)

    return np.array(lag)


def heart_rate(
    x: Union[List, np.ndarray],
    sfreq: int = 1000,
    unit: str = "rr",
    kind: str = "cubic",
    input_type: str = "peaks",
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform peaks data or RR intervals into continuous heart rate time series.

    Parameters
    ----------
    x :
        Boolean vector of peaks detection or RR intervals.
    sfreq :
        The sampling frequency of the desired output.
    unit :
        The heart rate unit in use. Can be `'rr'` (R-R intervals, in ms)
        or `'bpm'` (beats per minutes). Default is `'rr'`.
    kind :
        The method to use (parameter of `scipy.interpolate.interp1d`). The
        possible relevant methods for instantaneous heart rate are `'cubic'`
        (defalut), `'linear'`, `'previous'` and `'next'`.
    input_type :
        The type of input vector. Default is `"peaks"` (a boolean vector where
        `1` represents the occurrence of R waves or systolic peaks).
        Can also be `"rr_s"` or `"rr_ms"` for vectors of RR intervals, or
        interbeat intervals (IBI), expressed in seconds or milliseconds
        (respectively).

    Returns
    -------
    heartrate :
        The heart rate frequency.
    time :
        Time array.

    Examples
    --------
    1. From a boolean vector of peaks position:

    >>> from systole import import_ppg
    >>> ppg = import_ppg().ppg.to_numpy()  # Import PPG recording
    >>> _, peaks = ppg_peaks(ppg)  # Find systolic peaks
    >>> heartrate, time = heart_rate(peaks)  # Create continuous time series

    2. From a vector of RR intervals (miliseconds):
    >>> from systole import import_rr
    >>> rr = import_rr().rr.values
    >>> heartrate, time = heart_rate(rr, unit="bpm", input_type="rr_ms")

    Notes
    -----
    If the input is in the `peaks` format, it should be a boolean vector encoding the
    position of R wave, or systolic peaks.

    If it is in the form of RR intervals, it can be expressed in seconds or
    milliseconds, using `rr_s` and `rr_ms` parameters, respectively.

    The time and heart rate output will have the same length. Values before the first
    peak and after the last peak will be filled with NaN values.
    """
    x = np.asarray(x)

    # A peaks vector
    if input_type == "peaks":
        if ((x == 1) | (x == 0)).all():
            # Find peak indices
            peaks_idx = np.where(x)[0]

            time = (peaks_idx / sfreq)[1:]  # Create time vector (seconds)

            rr = np.diff(peaks_idx)

            # Use the peaks vector as time input
            new_time = np.arange(0, len(x) / sfreq, 1 / sfreq)

        else:
            raise ValueError("Input vector should only contain 0 and 1")

    # A vector of peaks indexs
    elif input_type == "peaks_idx":
        if (np.diff(x) > 0).all():
            time = (x / sfreq)[1:]  # Create time vector (seconds)

            rr = np.diff(x)

            # Use the peaks vector as time input
            new_time = np.arange(0, time[-1], 1 / sfreq)

        else:
            raise ValueError("Input vector should only contain increasing integers")

    # A vector of RR intervals
    elif input_type == "rr_s":
        if (x > 0).all():
            time = np.cumsum(x)  # Create time vector (seconds)
            rr = x * 1000
            new_time = np.arange(0, time[-1], 1 / sfreq)
        else:
            raise ValueError("RR intervals cannot be less than 0")

    elif input_type == "rr_ms":
        if (x > 0).all():
            time = np.cumsum(x) / 1000  # Create time vector (seconds)
            rr = x
            new_time = np.arange(0, time[-1], 1 / sfreq)
        else:
            raise ValueError("RR intervals cannot be less than 0")

    # R-R intervals (in miliseconds)
    heartrate = (rr / sfreq) * 1000
    if unit == "bpm":
        # Beats per minutes
        heartrate = 60000 / heartrate

    if kind is not None:
        # Interpolate
        f = interp1d(
            time, heartrate, kind=kind, bounds_error=False, fill_value=(np.nan, np.nan)
        )
        heartrate = f(new_time)

    return heartrate, new_time


def to_angles(
    x: Union[List, np.ndarray], events: Union[List, np.ndarray]
) -> np.ndarray:
    """Angular values of events according to x cycle peaks.

    Parameters
    ----------
    x :
        The reference time serie. Time points can be unevenly spaced.
    events :
        The events time serie.

    Returns
    -------
    ang :
        The angular value of events in the cycle of interest (radians).
    """
    x = np.asarray(x)
    events = np.asarray(events)

    # If data is provided in bollean format
    if not any(x > 1):
        x = np.where(x == 1)[0]
        events = np.where(events == 1)[0]

    ang = []  # Where to store angular data
    for i in events:
        if (i >= np.min(x)) & (i < np.max(x)):
            # Length of current R-R interval
            ln = np.min(x[x > i]) - np.max(x[x <= i])

            # Event timing after previous R peak
            i -= np.max(x[x <= i])

            # Convert into radian [0 to pi*2]
            ang.append((i * np.pi * 2) / ln)

        elif i == np.max(x):
            ang.append(0.0)

    return np.asarray(ang)


def to_epochs(
    signal: Union[List, np.ndarray],
    triggers: Optional[Union[List, np.ndarray]] = None,
    triggers_idx: Optional[Union[List, np.ndarray]] = None,
    sfreq: int = 1000,
    tmin: float = -1.0,
    tmax: float = 10.0,
    event_val: int = 1,
    apply_baseline: Optional[Union[float, Tuple[float, float]]] = 0.0,
    verbose: bool = False,
    reject: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Epoch signal based on event triggers.

    Parameters
    ----------
    signal :
        The raw signal that should be epoched. The first dimension is time and should match
        with `len(triggers)` if `triggers` is provided. If `triggers_idx` is provided,
        `np.max(triggers_idx)` should be less than `signal.shape[0]`.
    triggers :
        The boolean indices of the events, shape=(times*sfreq, 1).
    triggers_idx :
        Trigger indexes. Each value encode the sample where an event occured (see
        also `sfreq`). Different conditions should be provided separately as list of
        arrays (can have different lenght).
    sfreq :
        The sampling frequency (default is 1000 Hz).
    tmin :
        Start time before event, in seconds, default is `-1.0`.
    tmax :
        End time after event, in seconds, defautl is `10.0`.
    event_val :
        The index of event of interest. Default is `1`. Only relevant if `triggers` is
        not `None`.
    apply_baseline :
        If int or tuple, use the point or interval to apply a baseline
        (method: mean). If `None`, no baseline is applied. Default is set to `0`.
    verbose :
        If True, will return warnings if epoc are droped.
    reject :
        Segments of the signal that should be rejected.

    Returns
    -------
    epochs :
        List of (n Tials * Time) array.
    reject :
        List of rejected trials for each condition.

    Examples
    --------

    # Load dataset

    >>> ecg_df = import_dataset1(modalities=['ECG', 'Stim'])

    >>> triggers_idx = [
    >>>     np.where(ecg_df.stim.to_numpy() == 2)[0],
    >>>     np.where(ecg_df.stim.to_numpy() == 1)[0]
    >>> ]
    >>> signal = ecg_df.ecg.to_numpy()

    # Using event idx

    >>> epoch, rejected = to_epochs(signal=signal, triggers_idx=triggers_idx)

    # Using event triggers

    >>> epoch, rejected = to_epochs(signal=signal, triggers=ecg_df.stim.to_numpy(),
    >>>                             event_val=2, apply_baseline=(-1.0, 0.0))

    # Using a rejection vector
    >>> reject = np.zeros(len(signal))
    >>> reject[768285:] = 1  # Reject the second part of the recording
    >>> epoch, rejected = to_epochs(
    >>>     signal=signal, triggers=ecg_df.stim.to_numpy(), event_val=2,
    >>>     apply_baseline=(-1.0, 0.0), reject=reject
    >>>     )

    """
    # To numpy array
    signal = np.asarray(signal)

    # Create a list of triggers_idx arays and check that
    # the lengths are compatible with the signal array
    if triggers is not None:
        if isinstance(triggers, np.ndarray):
            triggers = [triggers]
        triggers_idx = []
        for tr in triggers:
            if signal.shape[0] != tr.shape[0]:
                raise ValueError(
                    """The length of the event and signal vector should match."""
                )
            triggers_idx.append(np.where(tr == event_val)[0])  # Find idx of events

    else:
        if triggers_idx is None:
            raise ValueError("""No triggers or triggers_idx provided.""")
        if isinstance(triggers_idx, np.ndarray):
            triggers_idx = [triggers_idx]
        for tr_idx in triggers_idx:
            if np.max(tr_idx) > signal.shape[0]:
                raise ValueError(
                    """The triggers_idx array contains values that are greater that the signal length."""
                )

    # Create a default bad array if not already provided
    if reject is None:
        reject = np.zeros(len(signal), dtype="bool")

    # Initialize counters
    n_rejected, n_outside_signal = 0, 0

    # How many sample to epoch before and after triggers
    this_min, this_max = round(tmin * sfreq), round(tmax * sfreq)

    all_epochs, all_rejected = [], []
    # Loop across conditions
    for this_triggers_idx in triggers_idx:
        epochs, rejected = [], []
        # Loop across events
        for ev in this_triggers_idx:
            # Check that the epoch is not outside the signal range
            if (ev + this_min < 0) | (ev + this_max > len(signal)):
                n_outside_signal += 1
                rejected.append(True)
                continue

            # Check if the signal contains an artefact
            if np.any(reject[ev + this_min : ev + this_max]):
                n_rejected += 1
                rejected.append(True)
                continue

            # If no problem, store the array in the epoch list
            trial = signal[ev + this_min : ev + this_max]
            if apply_baseline is None:
                epochs.append(trial)
            else:
                if isinstance(apply_baseline, float):
                    baseline = signal[ev + round(apply_baseline * sfreq)]
                if isinstance(apply_baseline, tuple):
                    low = ev + round(apply_baseline[0] * sfreq)
                    high = ev + round(apply_baseline[1] * sfreq)
                    baseline = signal[low:high].mean()
                epochs.append(trial - baseline)
                rejected.append(False)

        # Append to the condition level
        all_epochs.append(np.array(epochs))
        all_rejected.append(np.asarray(rejected))

    # Print % of rejected items
    if (n_rejected > 0) & (verbose is True):
        print(str(n_rejected) + " trial(s) droped due to artefacts")
    if (n_outside_signal > 0) & (verbose is True):
        print(str(n_outside_signal) + " trial(s) droped due to signal range")

    # Return list of epochs and rejected arrays
    return all_epochs, all_rejected


def simulate_rr(
    n_rr: int = 350,
    extra_idx: List = [50],
    missed_idx: List = [100],
    short_idx: List = [150],
    long_idx: List = [200],
    ectopic1_idx: List = [250],
    ectopic2_idx: List = [300],
    random_state: int = 42,
    as_peaks: bool = False,
    artefacts: bool = True,
) -> np.ndarray:
    """RR time series simulation with artefacts.

    n_rr :
        Number of RR intervals. Default is 350.
    extra_idx :
        Index of extra interval. Default is [50].
    missed_idx :
        Index of missed interval. Default is [100].
    short_idx :
        Index of short interval. Default is [150].
    long_idx :
        Index of long interval. Default is [200].
    ectopic1_idx :
        Index of ectopic interval. Default is [250].
    ectopic2_idx :
        Index of ectopic interval. Default is [300].
    random_state :
        Random state. Default is `42`.
    artefacts :
        If `True`, simulate artefacts in the signal.

    Returns
    -------
    rr :
        The RR time series.
    """
    np.random.seed(random_state)

    rr = np.array(
        [800 + 50 * np.random.normal(i, 0.6) for i in np.sin(np.arange(0, n_rr, 1.0))]
    )

    if artefacts is True:
        # Insert extra beats
        if extra_idx:
            n_extra = 0
            for i in extra_idx:
                rr[i - n_extra] -= 100
                rr = np.insert(rr, i, 100)
                n_extra += 1

        # Insert missed beats
        if missed_idx:
            n_missed = 0
            for i in missed_idx:
                rr[i + n_missed] += rr[i + 1]
                rr = np.delete(rr, i + 1)
                n_missed += 1

        # Add short interval
        if short_idx:
            for i in short_idx:
                rr[i] /= 2

        # Add long interval
        if long_idx:
            for i in long_idx:
                rr[i] *= 1.5

        # Add ectopic beat type 1 (NPN)
        if ectopic1_idx:
            for i in ectopic1_idx:
                rr[i] *= 0.7
                rr[i + 1] *= 1.3

        # Add ectopic beat type 2 (PNP)
        if ectopic2_idx:
            for i in ectopic2_idx:
                rr[i] *= 1.3
                rr[i + 1] *= 0.7

    # Transform to peaks vector if needed
    if as_peaks is True:
        peaks = np.zeros(np.cumsum(np.rint(rr).astype(int))[-1] + 50)
        peaks[(np.cumsum(np.rint(rr).astype(int)))] = 1
        peaks = peaks.astype(bool)
        peaks[0] = True
        return peaks
    else:
        return rr


def to_neighbour(
    signal: np.ndarray, peaks: np.ndarray, kind: str = "max", size: int = 50
) -> np.ndarray:
    """Replace peaks with max/min neighbour in a given window.

    Parameters
    ----------
    signal :
        Signal used to maximize/minimize peaks values.
    peaks:
        Boolean vector of peaks position.
    kind :
        Can be 'max' or 'min'.
    size :
        Size of the time window used to find max/min (samples). Defaults to `50`.

    Returns
    -------
    new_peaks:
        Boolean vector of peaks position.

    Raises
    ------
    ValueError
        If `kind` is not `"max"` or `"min"`.

    """
    new_peaks = peaks.copy()
    for pk in np.where(peaks)[0]:
        if kind == "max":
            x = signal[pk - size : pk + size].argmax()
        elif kind == "min":
            x = signal[pk - size : pk + size].argmin()
        else:
            raise ValueError("Invalid argument, kind should be " "max" " or " "min" "")

        new_peaks[pk] = False
        new_peaks[pk + (x - size)] = True

    return new_peaks


def input_conversion(
    x: Union[List, np.ndarray],
    input_type: str,
    output_type: str,
    sfreq: int = 1000,
) -> np.ndarray:
    """Convert input time series to the desired output format.

    This function is called by functions to convert time series to a different format.
    The input and outputs can be:
    * `peaks`: a boolean vector where `1` denote the detection of an event in the
    time-series.
    * `peaks_idx`: a 1d NumPy array of integers where each item is the sample index
    of an event in the time series.
    * `rr_ms`: a 1d NumPy array (integers or floats) of RR /peak-to-peak intervals
    in milliseconds.
    * `rr_s`: a 1d NumPy array (integers or floats) of RR /peak-to-peak intervals
    in seconds.

    Parameters
    ----------
    x :
        The input time series.
    input_type :
        The type of input provided (can be `"peaks"`, `"peaks_idx"`, `"rr_ms"`,
        `"rr_s"`).
    output_type :
        The type of desired output (can be `"peaks"`, `"peaks_idx"`, `"rr_ms"`,
        `"rr_s"`).
    sfreq :
        The sampling frequency (default is 1000 Hz). Only applies when `iput_type` is
        `"peaks"` or `"peaks_idx"`.

    Returns
    -------
    output :
        The time series converted to the desired format.

    """

    if output_type not in ["peaks", "peaks_idx", "rr_ms", "rr_s"]:
        raise ValueError("Invalid output type.")

    if input_type == output_type:
        raise ValueError("Input type and output type are the same.")

    x = np.asarray(x)

    if input_type == "peaks":
        if ((x == 1) | (x == 0)).all():
            if output_type == "rr_ms":
                output = (np.diff(np.where(x)[0]) / sfreq) * 1000
            elif output_type == "rr_s":
                output = np.diff(np.where(x)[0]) / sfreq
            elif output_type == "peaks_idx":
                output = np.where(x)[0]  # type: ignore
        else:
            raise ValueError("The peaks vector should only contain boolean values.")

    elif input_type == "peaks_idx":
        if (np.diff(x) > 0).all() & (np.rint(x) == x).all():
            if output_type == "rr_ms":
                output = (np.diff(x) / sfreq) * 1000
            elif output_type == "rr_s":
                output = np.diff(x) / sfreq
            elif output_type == "peaks":
                output = np.zeros(x[-1] + 1, dtype=bool)
                output[x] = True
        else:
            raise ValueError("Invalid peaks index provided.")

    elif input_type == "rr_ms":
        if (x > 0).all():
            if output_type == "rr_s":
                output = x / 1000
            elif output_type == "peaks":
                output = np.zeros(int(np.sum(x)) + 1, dtype=bool)
                output[np.cumsum(x)] = True
                output[0] = True
            elif output_type == "peaks_idx":
                output = np.cumsum(x)
                output = np.insert(output, 0, 0)
        else:
            raise ValueError("Invalid intervals provided.")

    elif input_type == "rr_s":
        if (x > 0).all():
            if output_type == "rr_ms":
                output = x * 1000  # type: ignore
            elif output_type == "peaks":
                output = np.zeros(np.sum(x * 1000).astype(int) + 1, dtype=bool)
                output[np.cumsum(x * 1000).astype(int)] = True
                output[0] = True
            elif output_type == "peaks_idx":
                output = np.cumsum(x * 1000).astype(int)
                output = np.insert(output, 0, 0)
        else:
            raise ValueError("Invalid intervals provided.")

    else:
        raise ValueError("Invalid input type.")

    return output


@jit(nopython=True)
def nan_cleaning(signal: np.ndarray, verbose: bool = True) -> np.ndarray:
    """Interpolate NaNs values.

    Parameters
    ----------
    signal :
        The physiological signal.
    verbose :
        Control function verbosity. Defaults to `True` (print)

    Returns
    -------
    clean_signal :
        The physiological signal without NaNs values.

    """

    arg_nans = np.where(np.isnan(signal))[0]
    if len(arg_nans) > 0:
        if verbose:
            print(
                f"... NaNs cleaning : interpolating {len(arg_nans)} NaN values found in the signal {int(100 * len(arg_nans)/len(signal))} %."
            )
        arg_float = np.where(~np.isnan(signal))[0]
        xp = np.arange(0, len(signal))[arg_float]
        fp = signal[arg_float]
        signal[arg_nans] = np.interp(arg_nans, xp=xp, fp=fp)

    return signal


def find_clipping(signal: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Automatically find the max and/or min threshold value of clipping artefacts.

    Parameters
    ----------
    signal :
        The physiological signal.

    Returns
    -------
    min_threshold, max_threshold :
        The minimum and maximum threshold corresponding to the clipping artefact. If no
        threshold is found, returns `None`.

    """
    min_threshold, max_threshold = None, None

    signal_max = np.max(signal)
    signal_min = np.min(signal)

    if (len(np.where(signal == signal_max)[0]) == 0) & (
        len(np.where(signal == signal_min)[0]) == 0
    ):
        return min_threshold, max_threshold

    signal_diff = np.diff(signal)

    # If max is not unique
    if len(np.where(signal == signal_max)[0]) > 1:
        # Count the number of occurence of max values with derivative = 0
        n_clip = np.isin(
            np.where(signal_diff == 0)[0] + 1, np.where(signal == signal_max)[0]
        ).sum()

        if n_clip > 2:
            max_threshold = signal_max

    # If min is not unique
    if len(np.where(signal == signal_min)[0]) > 1:
        # Count the number of occurence of min values with derivative = 0
        n_clip = np.isin(
            np.where(signal_diff == 0)[0] + 1, np.where(signal == signal_min)[0]
        ).sum()

        if n_clip > 2:
            min_threshold = signal_min

    return min_threshold, max_threshold


def get_valid_segments(
    signal: np.ndarray,
    bad_segments: Optional[Union[np.ndarray, List[Tuple[int, int]]]] = None,
) -> List[np.ndarray]:
    """Return the longest signal or intervals time series after dropping segments marked
    as bads.

    Parameters
    ----------
    signal :
        The physiological time serie that should be filtered.
    bad_segments :
        Bad segments in the signal where RR interval or peaks to peaks intervals could
        not be accurately estimated. Intervals from these segments will be dropped and
        the time domain parameters will be computed over the remaining intervals. A
        warning is printed if the final intervals range is less than 2 minutes.

    Return
    ------
    valid :
        The filtered signal / interval time serie given the bad segments, sorted
        according to the signal length.

    Examples
    --------
    From a given physiological signal, and a set of bad segments, return the list of
    valid time series, sorted according to their length.

    >>> import numpy as np
    >>> from systole.utils import get_valid_segments
    >>> signal = np.random.normal(size=1000)
    >>> bad_segments = [(500, 550), (700, 800)]
    >>> valids = get_valid_segments(signal=signal, bad_segments=bad_segments)
    >>> [len(sig) for sig in valids]
    `[500, 200, 150]`

    """

    # Return a list of tuple and merge overlapping intervals
    bad_segments = norm_bad_segments(bad_segments)

    # Create a list of valid signal array
    valids, start_valid = [], 0
    for bads in bad_segments:
        valids.append(signal[start_valid : bads[0]])

        # First bad segment at the beginning
        if bads[0] != 0:
            start_valid = bads[1]

    if bad_segments[-1][1] < len(signal):
        valids.append(signal[bad_segments[-1][1] :])

    return sorted(valids, key=len, reverse=True)


def norm_bad_segments(bad_segments) -> List[Tuple[int, int]]:
    """Normalize bad segments. Return a list of tuples and merge overlapping intervals.

    Parameters
    ----------
    bad_segments :
        The time stamps of the bad segments. Can be a boolean 1d numpy array where
        `True` is a bad segment (will be rejected), or a list of tuples such as
        (start_idx, end_idx) in signal samples or milliseconds when intervals are
        provided.

    Returns
    -------
    bad_segments_tuples :
        The merged bad segments as tuples such as (start_idx, end_idx) indicate when
        each bad segment starts and ends.

    Raises
    ------
    ValueError :
        If `bad_segments` is not a tuple or a np.ndarray.

    Examples
    --------

    Get a list of tuples such as `[(start_idx, end_idx)]` from a boolean vector that
    is of the same length that the associated signal and where `True` is a bad
    segment/sample.

    >>> import numpy as np
    >>> from systole.utils import norm_bad_segments
    >>> bool_bad_segments = np.zeros(100, dtype=bool)
    >>> bool_bad_segments[10:20] = True
    >>> bool_bad_segments[50:60] = True
    >>> new_segments = norm_bad_segments(bool_bad_segments)
    >>> new_segments
    `[(10, 20), (50, 60)]`

    From a list of tuples such as `[(start_idx, end_idx)]`, where some intervals are
    overlapping, get a clean version of these bad segments by merging the overlapping
    intervals.

    >>> bad_segments = [(100, 200), (150, 250)]
    >>> new_segments = norm_bad_segments(bad_segments)
    >>> assert new_segments
    `[(100, 250)]`

    """
    if isinstance(bad_segments, list):
        # Create boolean representation
        t_max = np.array(bad_segments).max()
        boolean_segments = np.zeros(t_max, dtype=int)

        for bads in bad_segments:
            boolean_segments[bads[0] : bads[1]] = 1

    elif isinstance(bad_segments, np.ndarray):
        boolean_segments = bad_segments.astype(int)

    else:
        raise ValueError(
            "bad_segments should be a list of tuples or a boolean 1d array."
            f" Got {type(bad_segments)}"
        )

    # Find the start and end of each bad segments
    bad_segments_list = [0] if boolean_segments[0] == 1 else []

    bad_segments_list.extend(
        [
            idx
            for idx in range(1, len(boolean_segments) - 1)
            if (boolean_segments[idx] == 1) & (boolean_segments[idx - 1] == 0)
            | (boolean_segments[idx] == 0) & (boolean_segments[idx - 1] == 1)
        ]
    )
    if boolean_segments[-1] == 1:
        bad_segments_list.append(len(boolean_segments))

    # Make it a list of tuples (start, end)
    bad_segments_tuples = [
        (bad_segments_list[i], bad_segments_list[i + 1])
        for i in range(0, len(bad_segments_list), 2)
    ]

    return bad_segments_tuples
