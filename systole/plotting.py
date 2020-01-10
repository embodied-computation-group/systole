import numpy as np
import matplotlib.pyplot as plt
from systole.detection import hrv_subspaces
from systole.utils import heart_rate


def plot_hr(oximeter, ax=None):
    """Given a peaks vector, returns frequency plots.

    Parameters
    ----------
    oximeter : instance of Oximeter
        The recording instance, where additional channels track different
        events using boolean recording.
    ax : `Matplotlib.Axes` or None
        Where to draw the plot. Default is *None* (create a new figure).

    Returns
    -------
    ax : `Matplotlib.Axes`
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
    ax : `Matplotlib.Axes` or None
        Where to draw the plot. Default is *None* (create a new figure).

    Returns
    -------
    ax : `Matplotlib.Axes`
        The figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))
    events = oximeter.channels.copy()
    for i, ch in enumerate(events):
        for id in np.where(events[ch])[0]:
            ax.plot(oximeter.times[id], i+0.5, 'bo')

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
    ax : `Matplotlib.Axes` or None
        Where to draw the plot. Default is *None* (create a new figure).

    Returns
    -------
    ax : `Matplotlib.Axes`
        The figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_title('Oximeter recording', fontweight='bold')
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


def plot_peaks(peaks, sfreq=1000, kind='lines', unit='rr', ax=None):
    """Peaks vector to continuous time serie.

    Parameters
    ----------
    peaks : array like
        Boolean vector of peaks in Oxi data.
    sfreq : int
        Sampling frequency. Default is 100 Hz.
    kind : str
        The method to use (parameter of `scipy.interpolate.interp1d`).
    unit : str
        The heartrate unit in use. Can be 'rr' (R-R intervals, in ms)
        or 'bpm' (beats per minutes). Default is 'rr'.
    ax : Matplotlib.Axes instance | None
        Where to draw the plot. Default is ´None´ (create a new figure).

    Returns
    -------
    ax : `Matplotlib.Axes`
        The figure.
    """
    if isinstance(peaks, list):
        peaks = np.asarray(peaks)

    hr, time = heart_rate(peaks, sfreq=sfreq)

    if unit == 'bpm':
        ylab = 'BPM'
    elif unit == 'rr':
        ylab = 'R-R (ms)'
    else:
        raise ValueError('Invalid unit. Should be ´rr´ or ´bpm´')

    if ax is None:
        fig, ax = plt.subplots()

    # Plot continuous HR
    ax.plot(time, hr, color='gray', linestyle='--')
    ax.set_ylabel(ylab)
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
    ax : `Matplotlib.Axes` or None
        Where to draw the plot. Default is *None* (create a new figure).

    Returns
    -------
    ax : `Matplotlib.Axes`
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
        fig, ax = plt.subplots(2, 1, figsize=(8, 5))

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


def circular(data, bins=32, density='area', offset=0, mean=False, norm=True,
             units='radians', color=None, ax=None):
    """Plot polar histogram.

    Parameters
    ----------
    data : array-like or list
        Angular values, in radians.
    bins : int
        Use even value to have a bin edge at zero.
    density : str
        Is the density represented via the height or the area of the bars.
        Default set to 'area' (avoid misleading representation).
    offset : float
        Where 0 will be placed on the circle, in radians. Default set to 0
        (right).
    mean : bool
        If True, show the mean and 95% CI. Default set to `False`
    norm : boolean
        Normalize the distribution between 0 and 1.
    units : str
        Unit of the angular representation. Can be 'degree' or 'radian'.
        Default set to 'radians'.
    color : Matplotlib color
        The color of the bars.
    ax : Matplotlib.Axes instance | None
        Where to draw the plot. Default is ´None´ (create a new figure).

    Returns
    -------
    ax : Matplotlib Axes instance
        Axes object with the plot.

    Notes
    -----
    The density function can be represented using the area of the bars, the
    height or the transparency (alpha). The default behaviour will use the
    area. Using the heigth can visually biase the importance of the largest
    values. Adapted from: https://jwalton.info/Matplotlib-rose-plots/

    Examples
    --------
    Plot polar data.

    .. plot::

       import numpy as np
       from systole.circular import circular
       x = np.random.normal(np.pi, 0.5, 100)
       circular(x)
    """
    if isinstance(data, list):
        data = np.asarray(data)

    if color is None:
        color = '#539dcc'

    if ax is None:
        ax = plt.subplot(111, polar=True)

    # Bin data and count
    count, bin = np.histogram(data, bins=bins, range=(0, np.pi*2))

    # Compute width
    widths = np.diff(bin)[0]

    if density == 'area':  # Default
        # Area to assign each bin
        area = count / data.size
        # Calculate corresponding bin radius
        radius = (area / np.pi)**.5
        alpha = (count * 0) + 1
    elif density == 'height':  # Using height (can be misleading)
        radius = count / data.size
        alpha = (count * 0) + 1
    elif density == 'alpha':  # Using transparency
        radius = (count * 0) + 1
        # Alpha level to each bin
        alpha = count / data.size
        alpha = alpha / alpha.max()
    else:
        raise ValueError('Invalid method')

    if norm is True:
        radius = radius / radius.max()

    # Plot data on ax
    for b, r, a in zip(bin[:-1], radius, alpha):
        plt.bar(b, r, align='edge', width=widths,
                edgecolor='k', linewidth=1, color=color, alpha=a)

    # Plot mean and CI
    if mean:
        ax.plot(circ_mean(data), radius.max(), 'ko')

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels
    ax.set_yticks([])

    if units == "radians":
        label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                 r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
        ax.set_xticklabels(label)
    plt.tight_layout()

    return ax


def plot_circular(data, y=None, hue=None, **kwargs):
    """Plot polar histogram.

    Parameters
    ----------
    data : DataFrame
        Angular data (rad.)
    y : str | list
        If data is a pandas instance, column containing the angular values.
    hue : str or list of strings
        Columns in data encoding the different conditions.
    **kwargs : Additional `_circular()` arguments.

    Returns
    -------
    ax : Matplotlib axes
        The circulars plot, one per condition.

    Examples
    --------
    .. plot::

       import numpy as np
       import pandas as pd
       from systole.circular import plot_circular
       x = np.random.normal(np.pi, 0.5, 100)
       y = np.random.uniform(0, np.pi*2, 100)
       data = pd.DataFrame(data={'x': x, 'y': y}).melt()
       plot_circular(data=data, y='value', hue='variable')
    """
    # Check data format
    if isinstance(data, pd.DataFrame):
        assert data.shape[0] > 0, 'Data must have at least 1 row.'
    elif isinstance(data, list):
        data = np.asarray(data)
        assert data.shape[0] > 1, 'Data must be 1d array.'
    elif isinstance(data, np.ndarray):
        assert data.shape[0] > 1, 'Data must be 1d array.'
    else:
        raise ValueError('Data must be instance of Numpy, Pandas or list.')

    palette = itertools.cycle(sns.color_palette('deep'))

    if hue is None:
        ax = circular(data, **kwargs)

    else:
        n_plot = data[hue].nunique()

        fig, axs = plt.subplots(1, n_plot, subplot_kw=dict(projection='polar'))

        for i, cond in enumerate(data[hue].unique()):

            x = data[y][data[hue] == cond]

            ax = circular(x, color=next(palette), ax=axs[i], **kwargs)

    return ax
