import numpy as np
from scipy import interpolate


def norm_triggers(x, threshold, n, direction='higher'):
    """Turns noisy triggers into unique boolean.

    Keep the first trigger and set to 0 the n following values.

    Parameters
    ----------
    x : NumPy array
        The triggers to convert.
    threshold : float
        Threshold for triggering values.
    n : int
        Number of values to force to 0 after each triggers.
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
        y = x > threshold
    elif direction == 'lower':
        y = x < threshold
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


def r_shift(raw, events):
    """Plot the absolute time deviation between events and r waves.

    Parameters
    ----------
    raw : NumPy array
        Raw ECG data.

    events : NumPy array
        Event timing.

    direction : str
        Events should occur before or after the r peak. Default is 'after'.

    Returns
    -------
    ax : Matplotlib axes

    lag : Numpy array
        If `show=False`, return the lag instead.

    """


def heart_rate(peaks, sfreq, unit='rr', method=None):
    """Transform peaks data into heart rate time series.

    Parameters
    ----------
    peaks : array
        Peaks indexes.
    sfreq : int
        Sampling frequency
    method : str
        The method to use.

    Retruns
    -------
    heartrate : array
        The heart rate frequency
    time : array
        Time array.
    """
    time = peaks / 75

    # R-R heartratevals (in miliseconds)
    heartrate = (np.diff(peaks, prepend=peaks[0]) / sfreq) * 1000
    if unit == 'bpm':
        # Beats per minutes
        heartrate = (60 / heartrate) * 1000

    if method is not None:
        if method == 'interpolate':
            f = interpolate.interp1d(time, heartrate, fill_value="extrapolate")
            time = np.arange(0, time[-1], 1/sfreq)
            heartrate = f(time)

        elif method == 'staircase':
            # From 0 to first peak
            heartrate = np.repeat((peaks[0]/sfreq) * 1000, peaks[0])
            for i in range(len(peaks)-1):
                rr = peaks[i+1] - peaks[i]
                a = np.repeat((rr/sfreq) * 1000, rr)
                heartrate = np.append(heartrate, a)
                time = np.arange(0, peaks[-1]/sfreq, 1/sfreq)

            if unit == 'bpm':
                # Beats per minutes
                heartrate = 60000/heartrate
        else:
            raise ValueError('Invalid method')

    return heartrate, time


def moving_function(x, win=0.2, sfreq=75, function=np.mean):
    """Return the moving average of the signal.

    Parameters
    ----------
    x : array
        The time-serie.
    win : float
        The size of the windows (in seconds)
    sfreq : int
        The sampling frequency.
    function : funct
        The operation to perform. Default is mean (moving average).

    Return
    ------
    y : array
        The averaged signal.
    """
    # Compute moving average
    win = int(win * sfreq)
    y = []
    for i in range(len(x)):
        if i < win/2:
            y.append(function(x[:win]))
        elif (i >= win/2) & (i < len(x - win)):
            y.append(function(x[i-int(win/2):i+int(win/2)]))
        else:
            y.append(function(x[-win:]))
    return np.asarray(y)
