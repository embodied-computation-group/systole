# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
from scipy.interpolate import interp1d


def norm_triggers(x, threshold=1, n=5, direction='higher'):
    """Turns noisy triggers into unique boolean.

    Keep the first trigger and set to 0 the n following values.

    Parameters
    ----------
    x : 1d array-like
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
    y : 1d array-like
        The filterd triggers
    """
    if not isinstance(x, np.ndarray):
        raise ValueError('x must be a Numpy array')

    if direction == 'higher':
        y = x >= threshold
    elif direction == 'lower':
        y = x <= threshold
    else:
        raise ValueError('Invalid direction')

    # Keep only the first trigger in window size
    for i in range(len(y)):
        if y[i]:
            if (len(y) - i) < n:  # If close to the end
                y[i+1:] = False
            else:
                y[i+1:i+n+1] = False
    return y


def time_shift(x, events, order='after'):
    """Return the delay between x and events.

    Parameters
    ----------
    x : 1d array-like
        Timing of reference events.
    events : 1d array-like
        Timing of events of heartrateest.
    order : str
        Consider event occurung before of after baseline. Default is 'after'.

    Returns
    -------
    time_shift : 1d array-like
        The delay between X and events (a.u)
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

    return lag


def heart_rate(x, sfreq=1000, unit='rr', kind='cubic'):
    """Transform peaks data into heart rate time series.

    Parameters
    ----------
    x : 1d array-like
        Boolean vector of heartbeat detection.
    sfreq : int
        Sampling frequency.
    unit : str
        The heartrate unit in use. Can be 'rr' (R-R intervals, in ms)
        or 'bpm' (beats per minutes). Default is 'rr'.
    kind : str
        The method to use (parameter of `scipy.interpolate.interp1d`).

    Returns
    -------
    heartrate : 1d array-like
        The heart rate frequency.
    time : 1d array-like
        Time array.

    Notes
    -----
    The input should be in the form of a boolean vector encoding the position
    of the peaks. The time and heart rate output will have the same
    length. Values before the first peak and after the last peak will be filled
    with NaN values.
    """
    if isinstance(x, list):
        x = np.asarray(x)
    if not ((x == 1) | (x == 0)).all():
        raise ValueError('Input vector should only contain 0 and 1')

    # Find peak indices
    peaks_idx = np.where(x)[0]

    # Create time vector (seconds):
    time = (peaks_idx/sfreq)[1:]

    rr = np.diff(peaks_idx)

    # R-R heartratevals (in miliseconds)
    heartrate = (rr / sfreq) * 1000
    if unit == 'bpm':
        # Beats per minutes
        heartrate = (60000 / heartrate)

    # Use the peaks vector as time input
    new_time = np.arange(0, len(x)/sfreq, 1/sfreq)

    if kind is not None:
        # Interpolate
        f = interp1d(time, heartrate, kind=kind, bounds_error=False,
                     fill_value=(np.nan, np.nan))
        heartrate = f(new_time)

    return heartrate, new_time


def to_angles(x, events):
    """Angular values of events according to x cycle peaks.

    Parameters
    ----------
    x : list or 1d array-like
        The reference time serie. Time points can be unevenly spaced.
    events : list or 1d array-like
        The events time serie.

    Returns
    -------
    ang : numpy array
        The angular value of events in the cycle of interest (radians).
    """
    if isinstance(x, list):
        x = np.asarray(x)
    if isinstance(events, list):
        events = np.asarray(events)

    # If data is provided in bollean format
    if not any(x > 1):
        x = np.where(x == 1)[0]
        events = np.where(events == 1)[0]

    ang = []  # Where to store angular data
    for i in events:

        if (i >= x.min()) & (i < x.max()):

            # Length of current R-R interval
            ln = np.min(x[x > i]) - np.max(x[x <= i])

            # Event timing after previous R peak
            i -= np.max(x[x <= i])

            # Convert into radian [0 to pi*2]
            ang.append((i*np.pi*2)/ln)

        elif i == x.max():
            ang.append(0.0)

    return ang


def to_epochs(x, events, sfreq=1000, tmin=-1, tmax=10, event_val=1,
              sigma=10, apply_baseline=0, verbose=False, reject=None):
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
        If int or tuple, use the point or interval to apply a baseline (method:
        mean). If *None*, no baseline is applied.
    verbose : boolean
        If True, will return warnings if epoc are droped.
    reject : 1d array-like or None
        Segments of the signal that should be rejected.

    Returns
    -------
    epochs : 2d array-like
        Event * Time array.
    """
    if len(x) != len(events):
        raise ValueError("""The length of the event and signal vector
                                shoul match exactly""")
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
    epochs = np.zeros(
                shape=(len(events), ((np.abs(tmin) + np.abs(tmax)) * sfreq)))
    for i, ev in enumerate(events):

        # Security check (epochs is not outside signal limits)
        if (ev+round(tmin*sfreq) < 0) | (ev+round(tmax*sfreq) > len(x)):
            if verbose is True:
                print('Drop 1 epoch due to signal limits.')
            rejected += 1
            epochs[i, :] = np.nan

        # Security check (trial contain bad peak)
        elif np.any(reject[ev+round(tmin*sfreq):ev+round(tmax*sfreq)]):
            if verbose is True:
                print('Drop 1 epoch due to artefact.')
            rejected += 1
            epochs[i, :] = np.nan
        else:
            trial = x[ev+round(tmin*sfreq):ev+round(tmax*sfreq)]
            if apply_baseline is None:
                epochs[i, :] = trial
            else:
                if isinstance(apply_baseline, int):
                    baseline = x[ev+round(apply_baseline*sfreq)]
                if isinstance(apply_baseline, tuple):
                    low = ev+round(apply_baseline[0]*sfreq)
                    high = ev+round(apply_baseline[1]*sfreq)
                    baseline = x[low:high].mean()
                epochs[i, :] = trial - baseline

    # Print % of rejected items
    if (rejected > 0) & (verbose is True):
        print(str(rejected) + ' trial(s) droped due to inconsistent recording')

    return epochs


def simulate_rr(n_rr=350, extra_idx=[50], missed_idx=[100], short_idx=[150],
                long_idx=[200], ectopic1_idx=[250], ectopic2_idx=[300],
                random_state=42, as_peaks=False, artefacts=True):
    """ RR time series simulation with artefacts.

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
    rr : 1d array-like
        The RR time series.
    """
    np.random.seed(random_state)

    rr = np.array(
        [800 + 50 * np.random.normal(i, .6) for i in np.sin(
            np.arange(0, n_rr, 1.0))])

    if artefacts is True:

        # Insert extra beats
        if extra_idx:
            n_extra = 0
            for i in extra_idx:
                rr[i-n_extra] -= 100
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
                rr[i] *= .7
                rr[i+1] *= 1.3

        # Add ectopic beat type 2 (PNP)
        if ectopic2_idx:
            for i in ectopic2_idx:
                rr[i] *= 1.3
                rr[i+1] *= .7

    # Transform to peaks vector if needed
    if as_peaks is True:
        peaks = np.zeros(np.cumsum(np.rint(rr).astype(int))[-1]+50)
        peaks[(np.cumsum(np.rint(rr).astype(int)))] = 1
        peaks = peaks.astype(bool)
        peaks[0] = True
        return peaks
    else:
        return rr
