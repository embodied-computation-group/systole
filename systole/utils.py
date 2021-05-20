# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d


def norm_triggers(
    x: Union[List, np.ndarray],
    threshold: int = 1,
    n: int = 5,
    direction: str = "higher",
) -> Union[List, np.ndarray]:
    """Turns noisy triggers into boolean.

    Keep the first trigger and set to 0 the n following values.

    Parameters
    ----------
    x : np.ndarray or list
        The triggers to convert.
    threshold : float
        Threshold for triggering values. Default is 1.
    n : int
        Number of values to force to 0 following each triggers. Default is 5.
    direction : str
        Indicates if triggers are higher or lower than threshold. Can be
        `higher` or `lower`. Default is `higher`.

    Returns
    -------
    y : np.ndarray
        The filterd triggers array.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("x must be a Numpy array")

    if direction == "higher":
        y = x >= threshold
    elif direction == "lower":
        y = x <= threshold
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
    x : np.ndarray or list
        Timing of reference events.
    events : np.ndarray or list
        Timing of events of heartrateest.
    order : str
        Consider event occurung before of after baseline. Default is 'after'.

    Returns
    -------
    time_shift : np.ndarray
        The delay between X and events (a.u).
    """
    if isinstance(x, list):
        x = np.asarray(x)
    if isinstance(events, list):
        events = np.asarray(events)

    lag = []
    for e in events:
        # Find the closest reference before the event of heartrateest
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
    x : np.ndarray or list
        Boolean vector of peaks detection or RR intervals.
    sfreq : int
        The sampling frequency of the desired output.
    unit : str
        The heart rate unit in use. Can be `'rr'` (R-R intervals, in ms)
        or `'bpm'` (beats per minutes). Default is `'rr'`.
    kind : str
        The method to use (parameter of `scipy.interpolate.interp1d`). The
        possible relevant methods for instantaneous heart rate are `'cubic'`
        (defalut), `'linear'`, `'previous'` and `'next'`.
    input_type : str
        The type of input vector. Default is `"peaks"` (a boolean vector where
        `1` represents the occurrence of R waves or systolic peaks).
        Can also be `"rr_s"` or `"rr_ms"` for vectors of RR intervals, or
        interbeat intervals (IBI), expressed in seconds or milliseconds
        (respectively).

    Returns
    -------
    heartrate : np.ndarray
        The heart rate frequency.
    time : np.ndarray
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
    If the input is in the `peaks` format, it should be a boolean vector
    encoding the position of R wave, or systolic peaks.

    If it is in the form of RR intervals, it can be expressed in seconds or
    milliseconds, using `rr_s` and `rr_ms` parameters, respectively.

    The time and heart rate output will have the same length. Values before
    the first peak and after the last peak will be filled with NaN values.
    """
    x = np.asarray(x)

    # A peaks vector
    if input_type == "peaks":
        if ((x == 1) | (x == 0)).all():

            # Find peak indices
            peaks_idx = np.where(x)[0]

            # Create time vector (seconds):
            time = (peaks_idx / sfreq)[1:]

            rr = np.diff(peaks_idx)

            # Use the peaks vector as time input
            new_time = np.arange(0, len(x) / sfreq, 1 / sfreq)

        else:
            raise ValueError("Input vector should only contain 0 and 1")

    # A vector of RR intervals
    elif input_type == "rr_s":
        time = np.cumsum(x)
        rr = x * 1000
        new_time = np.arange(0, time[-1], 1 / sfreq)

    elif input_type == "rr_ms":
        time = np.cumsum(x) / 1000
        rr = x
        new_time = np.arange(0, time[-1], 1 / sfreq)

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
    x : np.ndarray or list
        The reference time serie. Time points can be unevenly spaced.
    events : np.ndarray or list
        The events time serie.

    Returns
    -------
    ang : numpy array
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
    x: Union[List, np.ndarray],
    events: Union[List, np.ndarray],
    sfreq: int = 1000,
    tmin: float = -1.0,
    tmax: float = 10.0,
    event_val: int = 1,
    apply_baseline: Optional[Union[int, Tuple]] = 0,
    verbose: bool = False,
    reject: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Epoch signal based on events indices.

    Parameters
    ----------
    x : 1darray-like or list
        An instance of Raw
    events : 1d array-like
        The boolean indices of the events, shape (times*sfreq, 1)
    sfreq : int
        The sampling frequency (default is 1000 Hz).
    tmin : float
        Start time before event, in seconds, default is -1.
    tmax : float
        End time after event, in seconds, defautl is 10.
    event_val : int
        The index of event of interest. Default is *1*.
    apply_baseline : int, tuple or None
        If int or tuple, use the point or interval to apply a baseline
        (method: mean). If *None*, no baseline is applied.
    verbose : boolean
        If True, will return warnings if epoc are droped.
    reject : np.ndarray or None
        Segments of the signal that should be rejected.

    Returns
    -------
    epochs : 2d array-like
        Event * Time array.
    """
    if len(x) != len(events):
        raise ValueError(
            """The length of the event and signal vector
                                shoul match exactly"""
        )
    # To numpy array
    if isinstance(x, list):
        x = np.asarray(x)
    if isinstance(events, list):
        events = np.asarray(events)

    # From boolean to event indices
    events = np.where(events == event_val)[0]

    # Bads array
    if reject is None:
        reject = np.zeros(len(x))

    rejected = 0
    epochs = np.zeros(shape=(len(events), int((np.abs(tmin) + np.abs(tmax)) * sfreq)))
    for i, ev in enumerate(events):

        # Security check (epochs is not outside signal limits)
        if (ev + round(tmin * sfreq) < 0) | (ev + round(tmax * sfreq) > len(x)):
            if verbose is True:
                print("Drop 1 epoch due to signal limits.")
            rejected += 1
            epochs[i, :] = np.nan

        # Security check (trial contain bad peak)
        elif np.any(reject[ev + round(tmin * sfreq) : ev + round(tmax * sfreq)]):
            if verbose is True:
                print("Drop 1 epoch due to artefact.")
            rejected += 1
            epochs[i, :] = np.nan
        else:
            trial = x[ev + round(tmin * sfreq) : ev + round(tmax * sfreq)]
            if apply_baseline is None:
                epochs[i, :] = trial
            else:
                if isinstance(apply_baseline, int):
                    baseline = x[ev + round(apply_baseline * sfreq)]
                if isinstance(apply_baseline, tuple):
                    low = ev + round(apply_baseline[0] * sfreq)
                    high = ev + round(apply_baseline[1] * sfreq)
                    baseline = x[low:high].mean()
                epochs[i, :] = trial - baseline

    # Print % of rejected items
    if (rejected > 0) & (verbose is True):
        print(str(rejected) + " trial(s) droped due to inconsistent recording")

    return epochs


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

    n_rr : int
        Number of RR intervals. Default is 350.
    extra_idx : list
        Index of extra interval. Default is [50].
    missed_idx : list
        Index of missed interval. Default is [100].
    short_idx : list
        Index of short interval. Default is [150].
    long_idx : list
        Index of long interval. Default is [200].
    ectopic1_idx : list
        Index of ectopic interval. Default is [250].
    ectopic2_idx : list
        Index of ectopic interval. Default is [300].
    random_state : int
        Random state. Default is *42*.
    artefacts : bool
        If True, simulate artefacts in the signal.

    Returns
    -------
    rr : np.ndarray
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
    signal : np.ndarray
        Signal used to maximize/minimize peaks.
    peaks: np.ndarray
        Boolean vector of peaks position.
    kind : str
        Can be 'max' or 'min'.
    size : int
        Size of the time window used to find max/min (samples).

    Returns
    -------
    new_peaks: np.ndarray
        Boolean vector of peaks position.
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


def to_rr(peaks: Union[List[float], np.ndarray], sfreq: int = 1000) -> np.ndarray:
    """Convert peaks index to intervals time series (RR, beat-to-beat...).

    Parameters
    ----------
    peaks : np.ndarray or list
        Either a boolean array or sample index. Default is *boolean*. If the
        input array does not only contain 0 or 1, will automatically try sample
        index.
    sfreq : int
        The sampling frequency (default is 1000 Hz).

    Returns
    -------
    rr : np.ndarray
        Interval time series (in miliseconds).
    """
    if isinstance(peaks, list):
        peaks = np.asarray(peaks)
    if ((peaks == 1) | (peaks == 0)).all():
        rr = (np.diff(np.where(peaks)[0]) / sfreq) * 1000
    else:
        rr = (np.diff(peaks) / sfreq) * 1000

    return rr
