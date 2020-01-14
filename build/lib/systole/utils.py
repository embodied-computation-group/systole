# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def norm_triggers(x, threshold=1, n=5, direction='higher'):
    """Turns noisy triggers into unique boolean.

    Keep the first trigger and set to 0 the n following values.

    Parameters
    ----------
    x : NumPy array
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
    y : array
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
    x : NumPy array
        Timing of reference events.

    events : NumPy array
        Timing of events of heartrateest.

    order : str
        Consider event occurung before of after baseline. Default is 'after'.

    Returns
    -------
    time_shift : NumPy array
        The delay between X and events (a.u)
    """
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
    The input should be in the form of a boolean vector encoding the peaks
    position. The time and heartrate output will have the same length. Values
    before the first peak anf after the last peak will be filled with the
    adjacent heartrate.
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
    x : list or numpy array
        The reference time serie. Time points can be unevenly spaced.
    events : list or numpy array
        The events time serie.

    Returns
    -------
    ang : numpy array
        The angular value of events in the reference (in radians).
    """
    if isinstance(x, list):
        x = np.asarray(x)
    if isinstance(events, list):
        events = np.asarray(events)

    # If data is provided as bollean format
    if not any(x > 1):
        x = np.where(x == 1)[0]
        events = np.where(events == 1)[0]

    ang = []  # Where to store angular data
    for i in events:

        if (i >= x.min()) & (i <= x.max()):

            # Length of current R-R interval
            ln = np.min(x[x >= i]) - np.max(x[x < i])

            # Event timing after previous R peak
            i -= np.max(x[x <= i])

            # Convert into radian [0 to pi*2]
            ang.append((i*np.pi*2)/ln)

    return ang


def to_epochs(x, events, sfreq=1000, tmin=-1, tmax=10, event_val=1,
              smooth=True, sigma=10, apply_baseline=0, verbose=False):
    """Epoch signal based on events indexes.

    Parameters
    ----------
    x : ndarray | list
        An instance of Raw
    events : boolean array
        The events, shape (times*sfreq, 1)
    sfreq : int
        The sampling frequency (default is 1000 Hz).
    tmin : float, default to -1
        Start time before event, in seconds.
    tmax : float, defautl to 10
        End time after event, in seconds.
    event_idx : int
        The index of event of interest. Default is `1`.
    apply_baseline : int, tuple, None
        If int or tuple, use the point or interval to apply a baseline (method:
        mean). If None, no baseline is applied.
    verbose : boolean
        If True, will return warnings.

    Returns
    -------
    epochs : ndarray
        Event * Time array.
    """
    if len(x) != len(events):
        raise ValueError("""The length of the event and signal vector
                                shoul match exactly""")

    # From boolean to event indexes
    events = np.where(events == event_val)[0]

    if smooth:
        x = gaussian_filter1d(x, sigma=sigma)

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
