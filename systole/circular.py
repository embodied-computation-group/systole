# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import pandas as pd
import itertools
from pingouin import circ_mean
import seaborn as sns
import matplotlib.pyplot as plt


def circular(data, bins=32, density='area', offset=0, mean=False, norm=True,
             units='radians', color=None, axis=None):
    """Plot polar histogram.

    Parameters
    ----------
    data : array | list
        Angular values, in radians.
    bins : int
        Use even value to have a bin edge at zero.
    density : str
        Is the density represented via the height or the area of the bars.
        Default set to `area` (avoid misleading representation).
    offset : float
        Where 0 will be placed on the circle, in radians. Default set to 0
        (right).
    mean : boolean
        If True, show the mean and 95% CI. Default set to `False`
    norm : boolean
        Normalize the distribution between 0 and 1.
    units : str
        Unit of the angular representation. Can be `degree` or `radian`.
        Default set to `radians`.
    color : Matplotlib color.
        The color of the bars.
    axis : Matplotlib axis.
        Specify the axis where to plot. Default is `None`.

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

    if axis is None:
        ax = plt.subplot(111, polar=True)
    else:
        ax = axis

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

            ax = circular(x, color=next(palette), axis=axs[i], **kwargs)

    return ax


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
