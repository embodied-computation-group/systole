# -*- coding: utf-8 -*-

import numpy as np


def epochs(x, events, sfreq, tmin=-1, tmax=10):
    """Epoch signal.

    Parameters
    ----------
    x : ndarray | list
        An instance of Raw
    events : boolean array
        The events, shape (times*sfreq, 1)
    sfreq : int
        The sampling frequency.
    tmin : float, default to -1
        Start time before event, in seconds.
    tmax : float, defautl to 10
        End time after event, in seconds.

    Returns
    -------
    epochs : ndarray
        The epoched signal.

    Examples
    --------
    """
    if len(events) == x.shape[0]:
        # From boolean to event indexes
        events = np.where(events == 1)[0]

    epochs = []
    for i in events:
        trial = x[i+round(tmin*sfreq):i+round(tmax*sfreq)]
        epochs.append(trial)

    return np.asarray(epochs)
