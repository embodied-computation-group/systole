import numpy as np
import matplotlib.pyplot as plt


def plot_hr(oximeter, ax=None):
    """Given a peaks vector, returns frequency plots.

    Parameters
    ----------
    oximeter : instance of Oximeter
        The recording instance, where additional channels track different
        events using boolean recording.

    Returns
    -------
    ax : Matplotlib instance
        Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(oximeter.times, oximeter.instant_rr)
    ax.set_xlabel('Time (s)', size=20)
    ax.set_ylabel('R-R (ms)', size=20)

    return ax


def plot_events(oximeter, ax=None):
    """Plot events distribution.

    Parameters
    ----------
    oximeter : instance of Oximeter
        The recording instance, where additional channels track different
        events using boolean recording.

    Returns
    -------
    ax : Matplotlib instance
        The axe instance of the Matplotlib figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    events = oximeter.channels
    for i, ev in enumerate(events):
        events[ev] = np.asarray(events[ev]) == 1
        ax.fill_between(x=oximeter.times, y1=i, y2=i+1, where=events[ev])

    # Add y ticks with channels names
    ax.set_yticks(np.arange(len(events)) + 0.5)
    ax.set_yticklabels([key for key in events])
    ax.set_xlabel('Time (s)', size=20)

    return ax


def plot_oximeter(oximeter, ax=None):
    """Plot recorded PPG signal.

    Parameters
    ----------
    oximeter : Oximeter instance
        The Oximeter instance used to record the signal.

    Return
    ------
    fig, ax : Matplotlib instances.
        The figure and axe instances.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(oximeter.times, oximeter.threshold, linestyle='--', color='gray',
            label='Threshold')
    ax.fill_between(x=oximeter.times,
                    y1=oximeter.threshold,
                    y2=np.asarray(oximeter.recording).min(),
                    alpha=0.2,
                    color='gray')
    ax.plot(oximeter.times, oximeter.recording, label='Recording')
    ax.fill_between(x=oximeter.times,
                    y1=oximeter.recording,
                    y2=np.asarray(oximeter.recording).min(),
                    color='w')
    ax.plot(np.asarray(oximeter.times)[np.where(oximeter.peaks)[0]],
            np.asarray(oximeter.recording)[np.where(oximeter.peaks)[0]],
            'ro', label='Online estimation')
    ax.set_ylabel('PPG level', size=20)
    ax.set_xlabel('Time (s)', size=20)
    ax.legend()

    return ax


def plot_peaks(peaks, samples=75, kind='lines', frequency='rr'):
    """Peaks vector to continuous time serie.

    Parameters
    ----------
    peaks : array like
        Boolean vector of peaks in Oxi data.
    samples : int
        Sampling frequency of the recording.

    Returns
    -------
    ax : Matplotlib instance
        Figure.
    """
    if isinstance(peaks, list):
        peaks = np.asarray(peaks)

    # Check if peaks is in the form of triggers or time indexes
    if np.isin(np.unique(peaks), [0, 1]).all():
        changes = np.where(peaks)[0]
    else:
        changes = peaks

    if frequency == 'bpm':
        ylab = 'BPM'
    else:
        ylab = 'R-R (ms)'

    fig, ax = plt.subplots()
    if kind == 'lines':
        if frequency == 'rr':
            rr = np.diff(changes)/samples
            ax.plot(np.cumsum(rr) + (np.where(peaks)[0][0]/75),
                    rr * 1000, color='grey', linestyle='--')
            ax.plot(np.cumsum(rr) + (np.where(peaks)[0][0]/75),
                    rr * 1000, 'o', color='grey', markersize=5)
            plt.ylabel(ylab, size=15)
        elif frequency == 'bpm':
            rr = np.diff(changes)/samples
            ax.plot(np.cumsum(rr) + (np.where(peaks)[0][0]/75),
                    60 / rr, color='grey', linestyle='--')
            ax.plot(np.cumsum(rr) + (np.where(peaks)[0][0]/75),
                    60 / rr, 'o', color='grey', markersize=5)
            plt.ylabel(ylab, size=15)
        else:
            raise ValueError('Invalid kind, must be `bpm` or `rr`')
    else:

        staircase = np.array([])
        for i in range(len(peaks)-1):
            rr = peaks[i+1] - peaks[i]
            a = np.repeat((rr/samples) * 1000, rr)
            staircase = np.append(staircase, a)

        if kind == 'heatmap':
            heatmap = np.tile(staircase, (2, 1))
            if frequency == 'bpm':
                heatmap = 60000 / heatmap
            im = ax.imshow(heatmap, aspect='auto', cmap='Blues',
                           extent=[0, len(heatmap)/samples, 0, 1])
            plt.colorbar(im, ax=ax, label=ylab)
            ax.set_xlabel('Times (s)', size=15)
            ax.get_yaxis().set_visible(False)

        elif kind == 'staircase':
            if frequency == 'bpm':
                staircase = 60000 / staircase
            ax.plot(np.arange(0, len(staircase))/samples,
                    staircase, color='grey')
            ax.set_ylabel(ylab, size=15)
            ax.set_xlabel('Times (s)', size=15)

    plt.xlabel('Times (s)', size=15)

    return ax
