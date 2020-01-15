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
        Sampling frequency
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
    if not ((x == 1) | (x == 0)).all():
        raise ValueError('Input vector should only contain 0 and 1')
    if isinstance(x, list):
        x = np.asarray(x)

    # Find peak indexes
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
              sigma=10, apply_baseline=0, verbose=False):
    """Epoch signal based on events indexes.

    Parameters
    ----------
    x : 1darray-like or list
        An instance of Raw
    events : 1d array-like
        The boolean indexes of the events, shape (times*sfreq, 1)
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

    Returns
    -------
    epochs : 2d array-like
        Event * Time array.
    """
    if len(x) != len(events):
        raise ValueError("""The length of the event and signal vector
                                shoul match exactly""")

    # From boolean to event indexes
    events = np.where(events == event_val)[0]

    rejected = 0
    epochs = np.zeros(
                shape=(len(events), ((np.abs(tmin) + np.abs(tmax)) * sfreq)))
    for i, ev in enumerate(events):

        # Security check (epochs is not outside signal limits)
        if (ev+round(tmin*sfreq) < 0) | (ev+round(tmax*sfreq) > len(x)):
            if verbose is True:
                print('Drop 1 epoch due to signal limits.')
                rejected += 1
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
