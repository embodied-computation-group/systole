# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from systole.detection import hrv_subspaces
from systole.utils import heart_rate
from scipy.interpolate import interp1d


def plot_hr(x, sfreq=75, outliers=None, unit='rr', kind='cubic', ax=None):
    """Plot the instantaneous heart rate time course.

    Parameters
    ----------
    x : 1d array-like or `systole.recording.Oximeter`
        The recording instance, where additional channels track different
        events using boolean recording. If a 1d array is provided, should be
        a peaks vector.
    sfreq : int
        Signal sampling frequency. Default is 75 Hz.
    outliers : 1d array-like
        If not None, boolean array indexing RR intervals considered as outliers
        and plotted separately.
    unit : str
        The heartrate unit in use. Can be 'rr' (R-R intervals, in ms)
        or 'bpm' (beats per minutes). Default is 'rr'.
    kind : str
        The method to use (parameter of `scipy.interpolate.interp1d`).
    ax : `Matplotlib.Axes` or None
        Where to draw the plot. Default is *None* (create a new figure).

    Returns
    -------
    ax : `Matplotlib.Axes`
        The figure.
    """
    if isinstance(x, list):
        x = np.asarray(x)
    if not isinstance(x, np.ndarray):
        x = np.asarray(x.peaks)

    # If a RR time serie is provided, transform to peaks vector
    if not ((x == 0) | (x == 1)).all():
        x = np.round(x).astype(int)
        peaks = np.zeros(np.cumsum(x)[-1])
        peaks = np.insert(peaks, 0, 1)
        peaks[np.cumsum(x)] = 1
        sfreq = 1000
    else:
        peaks = x

    # Compute the interpolated instantaneous heart rate
    hr, times = heart_rate(peaks, sfreq=sfreq, unit=unit, kind=kind)

    # New peaks vector
    f = interp1d(np.arange(0, len(peaks)/sfreq, 1/sfreq), peaks,
                 kind='linear', bounds_error=False, fill_value=(0, 0))
    new_peaks = f(times)

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))

    # Interpolate instantaneous HR
    ax.plot(times, hr, linestyle='--', color='gray')

    # Heart beats
    ax.plot(times[np.where(new_peaks)[0]], hr[np.where(new_peaks)[0]], 'bo',
            alpha=0.5)

    # Show outliers
    if outliers is not None:
        idx = np.where(peaks)[0][1:][np.where(outliers)[0]]
        ax.plot(times[idx], hr[idx], 'ro')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('R-R (ms)')
    ax.set_title('Instantaneous Heart rate', fontweight='bold')

    return ax


def plot_events(oximeter, ax=None):
    """Plot events distribution.

    Parameters
    ----------
    oximeter : `systole.recording.Oximeter`
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
    palette = itertools.cycle(sns.color_palette('deep'))
    events = oximeter.channels.copy()
    for i, ch in enumerate(events):
        ax.fill_between(x=oximeter.times, y1=i, y2=i+0.5,
                        color=next(palette),
                        where=np.array(events[ch]) == 1)

    # Add y ticks with channels names
    ax.set_yticks(np.arange(len(events)) + 0.5)
    ax.set_yticklabels([key for key in events])
    ax.set_xlabel('Time (s)')
    ax.set_title('Events', fontweight='bold')

    return ax


def plot_oximeter(oximeter, ax=None):
    """Plot recorded PPG signal.

    Parameters
    ----------
    oximeter : `systole.recording.Oximeter`
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
    ax.plot(oximeter.times, oximeter.recording, label='Recording',
            color='#4c72b0')
    ax.fill_between(x=oximeter.times,
                    y1=oximeter.recording,
                    y2=np.asarray(oximeter.recording).min(),
                    color='w')
    ax.plot(np.asarray(oximeter.times)[np.where(oximeter.peaks)[0]],
            np.asarray(oximeter.recording)[np.where(oximeter.peaks)[0]],
            color='#c44e52', linestyle=None, label='Online estimation')
    ax.set_ylabel('PPG level')
    ax.set_xlabel('Time (s)')
    ax.legend()

    return ax


def plot_subspaces(x, subspace2=None, subspace3=None, c1=0.13, c2=0.17,
                   xlim=10, ylim=5, ax=None):
    """Plot hrv subspace as described by Lipponen & Tarvainen (2019).

    Parameters
    ----------
    x : 1d array-like
        Array of RR intervals or subspace1. If subspace1 is provided, subspace2
        and 3 must also be provided.
    subspace2, subspace3 : 1d array-like or None
        Default is `None` (expect x to be RR time serie).
    c1 : float
        Fixed variable controling the slope of the threshold lines. Default is
        0.13.
    c2 : float
        Fixed variable controling the intersect of the threshold lines. Default
        is 0.17.
    xlim : int
        Absolute range of the x axis. Default is 10.
    ylim : int
        Absolute range of the y axis. Default is 5.
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
        if not isinstance(x, (np.ndarray, np.generic)):
            x = np.asarray(x)
        subspace1, subspace2, subspace3 = hrv_subspaces(x)

    # Rescale to show outlier in scatterplot
    if xlim is not None:
        subspace1[subspace1 < -xlim] = -xlim
        subspace1[subspace1 > xlim] = xlim
    if ylim is not None:
        subspace2[subspace2 < -ylim] = -ylim
        subspace2[subspace2 > ylim] = ylim

        subspace3[subspace3 < -ylim*2] = -ylim*2
        subspace3[subspace3 > ylim*2] = ylim*2

    # Outliers
    ##########

    # Find ectobeats
    cond1 = (subspace1 > 1) & (subspace2 < (-c1 * subspace1-c2))
    cond2 = (subspace1 < -1) & (subspace2 > (-c1 * subspace1+c2))
    rejection1 = cond1 | cond2

    # Find long or shorts
    cond1 = (subspace1 > 1) & (subspace3 < -1)
    cond2 = (subspace1 < -1) & (subspace3 > 1)
    rejection2 = cond1 | cond2

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot data points
    ax[0].scatter(subspace1[~rejection1],
                  subspace2[~rejection1], color='b', edgecolors='k', zorder=10)
    ax[0].scatter(subspace1[rejection1],
                  subspace2[rejection1], color='r', edgecolors='k', zorder=10)

    # Upper area
    def f1(x): return -c1*x + c2
    ax[0].plot([-1, -10], [f1(-1), f1(-10)], 'k', linewidth=1, linestyle='--')
    ax[0].plot([-1, -1], [f1(-1), 10], 'k', linewidth=1, linestyle='--')
    x = [-10, -10, -1, -1]
    y = [f1(-10), 10, 10, f1(-1)]
    ax[0].fill(x, y, color='#fcddcb', alpha=0.8)

    # Lower area
    def f2(x): return -c1*x - c2
    ax[0].plot([1, 10], [f2(1), f2(10)], 'k', linewidth=1, linestyle='--')
    ax[0].plot([1, 1], [f2(1), -10], 'k', linewidth=1, linestyle='--')
    x = [1, 1, 10, 10]
    y = [f2(1), -10, -10, f2(10)]
    ax[0].fill(x, y, color='#fcddcb', alpha=0.8)

    # Blue area
    x = [-10, -10, -1, -1, 10, 10, 1, 1]
    y = [-10, f1(-10), f1(-1), 10, 10, f2(10), f2(1), -10]
    ax[0].fill(x, y, color='#c7dbef')

    ax[0].set_xlabel('Subspace $S_{11}$')
    ax[0].set_ylabel('Subspace $S_{12}$')
    ax[0].set_ylim(-ylim, ylim)
    ax[0].set_xlim(-xlim, xlim)
    ax[0].set_title('Subspace 1 \n (ectobeats detection)', fontweight='bold')

    ############

    # Plot data points
    ax[1].scatter(subspace1[~rejection2], subspace3[~rejection2], color='b',
                  edgecolors='k', zorder=10)
    ax[1].scatter(subspace1[rejection2], subspace3[rejection2], color='r',
                  edgecolors='k', zorder=10)

    # Upper area
    ax[1].plot([-1, -10], [1, 1], 'k', linewidth=1, linestyle='--')
    ax[1].plot([-1, -1], [1, 10], 'k', linewidth=1, linestyle='--')
    x = [-10, -10, -1, -1]
    y = [1, 10, 10, 1]
    ax[1].fill(x, y, color='#fcddcb', alpha=0.8)

    # Lower area
    ax[1].plot([1, 10], [-1, -1], 'k', linewidth=1, linestyle='--')
    ax[1].plot([1, 1], [-1, -10], 'k', linewidth=1, linestyle='--')
    x = [1, 1, 10, 10]
    y = [-1, -10, -10, -1]
    ax[1].fill(x, y, color='#fcddcb', alpha=0.8)

    # Blue area
    x = [-10, -10, -1, -1, 10, 10, 1, 1]
    y = [-10, 1, 1, 10, 10, -1, -1, -10]
    ax[1].fill(x, y, color='#c7dbef')

    ax[1].set_xlabel('Subspace $S_{21}$')
    ax[1].set_ylabel('Subspace $S_{22}$')
    ax[1].set_ylim(-ylim*2, ylim*2)
    ax[1].set_xlim(-xlim, xlim)
    ax[1].set_title('Subspace 2 \n (long and short beats detection)',
                    fontweight='bold')
    plt.tight_layout()

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
    ax : `Matplotlib.Axes` or None
        Where to draw the plot. Default is *None* (create a new figure).

    Returns
    -------
    ax : `Matplotlib.Axes`
        The figure.

    Notes
    -----
    The density function can be represented using the area of the bars, the
    height or the transparency (alpha). The default behaviour will use the
    area. Using the heigth can visually biase the importance of the largest
    values. Adapted from [#]_.

    The circular mean was adapted from the implementation of the pingouin
    python package [#]_

    Examples
    --------
    Plot polar data.

    .. plot::

       import numpy as np
       from systole.circular import circular
       x = np.random.normal(np.pi, 0.5, 100)
       circular(x)

    References
    ----------
    .. [#] https://jwalton.info/Matplotlib-rose-plots/

    .. [#] https://pingouin-stats.org/_modules/pingouin/circular.html#circ_mean
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
        # Use pingouin.circ_mean() method
        alpha = np.array(data)
        w = np.ones_like(alpha)
        circ_mean = np.angle(np.multiply(w, np.exp(1j * alpha)).sum(axis=0))
        ax.plot(circ_mean, radius.max(), 'ko')

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
    ax : `Matplotlib.Axes`
        The figure.

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
