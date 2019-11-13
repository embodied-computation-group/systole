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
        Number of values to force to 0 following each triggers.
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


def heart_rate(x, sfreq, unit='rr', method=None):
    """Transform peaks data into heart rate time series.

    Parameters
    ----------
    x : array
        Boolean vector of heartbeat detection.
    sfreq : int
        Sampling frequency
    unit : str
        The heartrate unit in use. Can be 'rr' (R-R intervals, in ms)
        or 'bpm' (beats per minutes). Default is 'rr'.
    method : str
        The method to use. Can be None or 'staircase'.

    Retruns
    -------
    heartrate : array
        The heart rate frequency
    time : array
        Time array.

    Notes:
    ------
    The input should be in the form of a boolean vector encoding the peaks
    position. The time and heartrate output will have the same length. Values
    before the first peak anf after the last peak will be filled with the
    adjacent heartrate.
    """
    if np.any((np.abs(np.diff(x)) > 1)):
        raise ValueError('Input vector should only contain 0 and 1')

    # Find peak indexes
    peaks = np.where(x)[0]

    # Create time vector
    if method is None:
        time = peaks / sfreq
    else:
        time = np.arange(0, len(x)) / sfreq

    # R-R heartratevals (in miliseconds)
    heartrate = (np.diff(peaks) / sfreq) * 1000
    if unit == 'bpm':
        # Beats per minutes
        heartrate = (60 / heartrate) * 1000

    if method == 'staircase':

        # From 0 to first peak
        heartrate = np.repeat((heartrate[0]/sfreq) * 1000, peaks[0])

        for i in range(len(peaks)-1):
            rr = peaks[i+1] - peaks[i]
            a = np.repeat((rr/sfreq) * 1000, rr)
            heartrate = np.append(heartrate, a)

        # From last peak to end
        heartrate = np.append(heartrate,
                              np.repeat(heartrate[-1],
                                        len(x) - len(heartrate)))

        if unit == 'bpm':
            # Beats per minutes
            heartrate = 60000/heartrate
    else:
        raise ValueError('Invalid method')

    # Security checks
    if method is not None:
        if len(heartrate) != len(x):
            raise ValueError('Inconsistent output length')

    if len(heartrate) != len(time):
        raise ValueError('Inconsistent time vector')

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
