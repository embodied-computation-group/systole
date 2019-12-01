# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import gaussian_filter1d


def to_epochs(x, events, sfreq=1000, tmin=-1, tmax=10, event_idx=1,
              smooth=True, sigma=10, apply_baseline=0):
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

    Returns
    -------
    epochs : ndarray
        Event * Time array.
    """
    if len(x) != len(events):
        raise ValueError("""The length of the event and signal vector
                                shoul match exactly""")

    # From boolean to event indexes
    events = np.where(events == event_idx)[0]

    if smooth:
        x = gaussian_filter1d(x, sigma=sigma)

    rejected = 0
    epochs = np.zeros(
                shape=(len(events), ((np.abs(tmin) + np.abs(tmax)) * sfreq)))
    for i, ev in enumerate(events):
        if (ev+round(tmax*sfreq)) < len(x):
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
        else:
            rejected += 1

    if rejected > 0:
        print(str(rejected) + ' trial(s) droped due to inconsistent recording')

    return epochs
