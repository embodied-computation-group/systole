import numpy as np
import matplotlib.pyplot as plt
from systole.detection import hrv_subspaces


def plot_hr(oximeter, ax=None):
    """Given a peaks vector, returns frequency plots.

    Parameters
    ----------
    oximeter : instance of Oximeter
        The recording instance, where additional channels track different
        events using boolean recording.

    Returns
    -------
    ax : Matplotlib.Axes instance.
        The figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(oximeter.times, oximeter.instant_rr)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('R-R (ms)')

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
    ax : Matplotlib.Axes instance.
        The figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))
    events = oximeter.channels
    for i, ev in enumerate(events):
        events[ev] = np.asarray(events[ev]) == 1
        ax.fill_between(x=oximeter.times, y1=i, y2=i+1, where=events[ev])

    # Add y ticks with channels names
    ax.set_yticks(np.arange(len(events)) + 0.5)
    ax.set_yticklabels([key for key in events])
    ax.set_xlabel('Time (s)')

    return ax


def plot_oximeter(oximeter, ax=None):
    """Plot recorded PPG signal.

    Parameters
    ----------
    oximeter : Oximeter instance
        The Oximeter instance used to record the signal.

    Return
    ------
    ax : Matplotlib.Axes instance.
        The figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))
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
    ax.set_ylabel('PPG level')
    ax.set_xlabel('Time (s)')
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
    ax : Matplotlib.Axes instance.
        The figure.
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
            plt.ylabel(ylab)
        elif frequency == 'bpm':
            rr = np.diff(changes)/samples
            ax.plot(np.cumsum(rr) + (np.where(peaks)[0][0]/75),
                    60 / rr, color='grey', linestyle='--')
            ax.plot(np.cumsum(rr) + (np.where(peaks)[0][0]/75),
                    60 / rr, 'o', color='grey', markersize=5)
            plt.ylabel(ylab)
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
            ax.set_xlabel('Times (s)')
            ax.get_yaxis().set_visible(False)

        elif kind == 'staircase':
            if frequency == 'bpm':
                staircase = 60000 / staircase
            ax.plot(np.arange(0, len(staircase))/samples,
                    staircase, color='grey')
            ax.set_ylabel(ylab)
            ax.set_xlabel('Times (s)')

    ax.set_xlabel('Times (s)')

    return ax


def plot_subspaces(x, subspace2=None, subspace3=None, c1=0.13, c2=0.17,
                   xlim=10, ylim=5, ax=None):
    """Plot hrv subspace as described by Lipponen & Tarvainen (2019).

    Parameters
    ----------
    x : array
        Array of RR intervals or subspace1. If subspace1 is provided, subspace2
        and 3 must also be provided.
    subspace2, subspace3 : array | None
        Default is `None` (expect x to be RR time serie).
    c1 : float
        Fixed variable controling the slope of the threshold lines. Default set
        to 0.13.
    c2 : float
        Fixed variable controling the slope of the threshold lines. Default set
        to 0.17.
    ax : Matplotlib axe
        Where to plot the figure.

    Return
    ------
    ax : Matplotlib.Axes instance.
        The figure.

    References
    ----------
    [1] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for
        heart rate variability time series artefact correction using novel beat
        classification. Journal of Medical Engineering & Technology, 43(3),
        173–181. https://doi.org/10.1080/03091902.2019.1640306
    """
    if (subspace3 is not None) & (subspace3 is not None):
        subspace1 = x
    else:
        assert isinstance(x, (np.ndarray, np.generic))
        subspace1, subspace2, subspace3 = hrv_subspaces(x)

    # Rescale to show outlier in scatterplot
    if xlim is not None:
        subspace1[subspace1 < -xlim] = -xlim
        subspace1[subspace1 > xlim] = xlim
    if ylim is not None:
        subspace2[subspace2 < -ylim] = -ylim
        subspace2[subspace2 > ylim] = ylim

        subspace3[subspace3 < -ylim] = -ylim
        subspace3[subspace3 > ylim] = ylim

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax[0].set_title('Subspace of successive RR interval differences')
    ax[0].plot(subspace1, subspace2, 'bo')

    # Upper area
    ax[0].plot([-1, -10], [1, -c1*-10 - c2], 'k')
    ax[0].plot([-1, -1], [1, 10], 'k')

    # Lower area
    ax[0].plot([1, 10], [-1, -c1*10 + c2], 'k')
    ax[0].plot([1, 1], [-1, -10], 'k')

    ax[0].set_xlabel('Subspace $S_11$')
    ax[0].set_ylabel('Subspace $S_12$')
    ax[0].set_ylim(-5, 5)
    ax[0].set_xlim(-10, 10)

    ax[1].plot(subspace1, subspace3, 'bo')

    # Upper area
    ax[1].plot([-1, -10], [1, 1], 'k')
    ax[1].plot([-1, -1], [1, 10], 'k')

    # Lower area
    ax[1].plot([1, 10], [-1, -1], 'k')
    ax[1].plot([1, 1], [-1, -10], 'k')

    ax[1].set_xlabel('Subspace $S_11$')
    ax[1].set_ylabel('Subspace $S_12$')
    ax[1].set_ylim(-5, 5)
    ax[1].set_ylim(-10, 10)

    return ax
