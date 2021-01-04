# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from systole.detection import rr_artefacts, oxi_peaks
from systole.utils import heart_rate
from scipy.interpolate import interp1d
from scipy.signal import welch


def plot_raw(signal, sfreq=75, type='ppg', ecg_method='hamilton', ax=None, figsize=(13, 5)):
    """Interactive visualization of PPG signal and beats detection.

    Parameters
    ----------
    signal : :py:class:`pandas.DataFrame` or 1d array-like
        Dataframe of signal recording in the long format. Should contain at
        least one ``'time'`` and one signal colum (can be ``'ppg'`` or
        ``'ecg'``). If an array is provided, will automatically create the
        DataFrame using the array as signal and ``sfreq`` as sampling
        frequency.
    sfreq : int
        Signal sampling frequency. Default is 75 Hz.
    type : str
        The recording modality. Can be ``'ppg'`` (pulse oximeter) or ``'ecg'``
        (electrocardiography).
    ecg_method : str
        Peak detection algorithm used by the
        :py:func:`systole.detection.ecg_peaks` function. Default is 'hamilton'.
    figsize : tuple
        Figure size. Default set to `(13, 5)`
    """
    if isinstance(signal, pd.DataFrame):
        # Find peaks - Remove learning phase
        if type == 'ppg':
            signal, peaks = oxi_peaks(signal.ppg, noise_removal=False)
        elif type == 'ecg':
            signal, peaks = ecg_peaks(signal.ecg, method=ecg_method,
                                      find_local=True)
    else:
        if type == 'ppg':
            signal, peaks = oxi_peaks(signal, noise_removal=False, sfreq=sfreq)
        elif type == 'ecg':
            signal, peaks = ecg_peaks(signal, method=ecg_method, sfreq=sfreq,
                                      find_local=True)
    time = np.arange(0, len(signal))/1000

    # Extract heart rate
    hr, time = heart_rate(peaks, sfreq=1000, unit='rr', kind='linear')

    #############
    # Upper panel
    #############
    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=2, figsize=figsize, sharex=True)

    # Signal
    ax[0].plot(time, signal, label='PPG signal', linewidth=1, color='#c44e52')
    
    # Peaks
    ax[0].scatter(x=time[peaks], y=signal[peaks], marker='o', label='Peaks', s=30,
                  color='white', edgecolors='DarkSlateGrey')
    ax[0].set_title('PPG recording')
    ax[0].set_ylabel('PPG level (a.u.)')
    ax[0].grid(True)
    
    #############
    # Lower panel
    #############

    # Instantaneous Heart Rate - Lines
    ax[1].plot(time, hr, label='R-R intervals', linewidth=1, color='#4c72b0')

    # Instantaneous Heart Rate - Peaks
    ax[1].scatter(x=time[peaks], y=hr[peaks], marker='o', label='R-R intervals', s=20,
                  color='white', edgecolors='DarkSlateGrey')
    ax[1].set_title('Instantaneous heart rate')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('R-R interval (ms)')
    ax[1].grid(True)
    
    plt.tight_layout()
    sns.despine()

    return fig, ax


def plot_events(oximeter, ax=None):
    """Plot events occurence across recording.

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


def plot_oximeter(x, sfreq=75, ax=None):
    """Plot PPG signal.

    Parameters
    ----------
    x : 1d array-like or `systole.recording.Oximeter`
        The ppg signal, or the Oximeter instance used to record the signal.
    sfreq : int
        Signal sampling frequency. Default is 75 Hz.
    ax : `Matplotlib.Axes` or None
        Where to draw the plot. Default is *None* (create a new figure).

    Returns
    -------
    ax : `Matplotlib.Axes`
        The figure.
    """
    if isinstance(x, (list, np.ndarray)):
        times = np.arange(0, len(x)/sfreq, 1/sfreq)
        recording = np.asarray(x)
        signal, peaks = oxi_peaks(x, new_sfreq=sfreq)
        threshold = None
        label = 'Offline estimation'
    else:
        times = np.asarray(x.times)
        recording = np.asarray(x.recording)
        peaks = np.asarray(x.peaks)
        threshold = np.asarray(x.threshold)
        label = 'Online estimation'

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_title('Oximeter recording', fontweight='bold')

    if threshold is not None:
        ax.plot(times, threshold, linestyle='--', color='gray',
                label='Threshold')
        ax.fill_between(x=times,
                        y1=threshold,
                        y2=recording.min(),
                        alpha=0.2,
                        color='gray')
    ax.plot(times, recording, label='Recording',
            color='#4c72b0')
    ax.fill_between(x=times,
                    y1=recording,
                    y2=recording.min(),
                    color='w')
    ax.plot(times[np.where(peaks)[0]], recording[np.where(peaks)[0]], 'o',
            color='#c44e52', label=label)
    ax.set_ylabel('PPG level')
    ax.set_xlabel('Time (s)')
    ax.legend()

    return ax


def plot_subspaces(rr, c1=.17, c2=.13, xlim=10, ylim=5, ax=None, figsize=(10, 5)):
    """Plot hrv subspace as described by Lipponen & Tarvainen (2019).

    Parameters
    ----------
    rr : 1d array-like
        Array of RR intervals or subspace1. If subspace1 is provided, subspace2
        and 3 must also be provided.
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
    figsize : tuple
        Figure size. Default set to `(10, 5)`

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
    if not isinstance(rr, (np.ndarray, np.generic)):
        rr = np.asarray(rr)
    artefacts = rr_artefacts(rr)

    # Rescale to show outlier in scatterplot
    if xlim is not None:
        artefacts['subspace1'][artefacts['subspace1'] < -xlim] = -xlim
        artefacts['subspace1'][artefacts['subspace1'] > xlim] = xlim
    if ylim is not None:
        artefacts['subspace2'][artefacts['subspace2'] < -ylim] = -ylim
        artefacts['subspace2'][artefacts['subspace2'] > ylim] = ylim

        artefacts['subspace3'][artefacts['subspace3'] < -ylim*2] = -ylim*2
        artefacts['subspace3'][artefacts['subspace3'] > ylim*2] = ylim*2

    # Filter for normal beats
    normalBeats = ((~artefacts['ectopic']) & (~artefacts['short']) &
                   (~artefacts['long']) & (~artefacts['missed']) &
                   (~artefacts['extra']))

    #############
    # First panel
    #############

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=figsize)

    # Plot normal beats
    ax[0].scatter(artefacts['subspace1'][normalBeats],
                  artefacts['subspace2'][normalBeats],
                  color='gray', edgecolors='k', s=15,
                  alpha=0.2, zorder=10, label='Normal')

    # Plot outliers
    ax[0].scatter(artefacts['subspace1'][artefacts['ectopic']],
                  artefacts['subspace2'][artefacts['ectopic']],
                  color='r', edgecolors='k', zorder=10, label='Ectopic')
    ax[0].scatter(artefacts['subspace1'][artefacts['short']],
                  artefacts['subspace2'][artefacts['short']],
                  color='b', edgecolors='k', zorder=10,
                  marker='s', label='Short')
    ax[0].scatter(artefacts['subspace1'][artefacts['long']],
                  artefacts['subspace2'][artefacts['long']],
                  color='g', edgecolors='k', zorder=10,
                  marker='s', label='Long')
    ax[0].scatter(artefacts['subspace1'][artefacts['missed']],
                  artefacts['subspace2'][artefacts['missed']],
                  color='g', edgecolors='k', zorder=10, label='Missed')
    ax[0].scatter(artefacts['subspace1'][artefacts['extra']],
                  artefacts['subspace2'][artefacts['extra']],
                  color='b', edgecolors='k', zorder=10, label='Extra')
    # Upper area
    def f1(x): return -c1*x + c2
    ax[0].plot([-1, -10], [f1(-1), f1(-10)], 'k', linewidth=1, linestyle='--')
    ax[0].plot([-1, -1], [f1(-1), 10], 'k', linewidth=1, linestyle='--')
    x = [-10, -10, -1, -1]
    y = [f1(-10), 10, 10, f1(-1)]
    ax[0].fill(x, y, color='gray', alpha=0.3)

    # Lower area
    def f2(x): return -c1*x - c2
    ax[0].plot([1, 10], [f2(1), f2(10)], 'k', linewidth=1, linestyle='--')
    ax[0].plot([1, 1], [f2(1), -10], 'k', linewidth=1, linestyle='--')
    x = [1, 1, 10, 10]
    y = [f2(1), -10, -10, f2(10)]
    ax[0].fill(x, y, color='gray', alpha=0.3)

    ax[0].set_xlabel('Subspace $S_{11}$')
    ax[0].set_ylabel('Subspace $S_{12}$')
    ax[0].set_ylim(-ylim, ylim)
    ax[0].set_xlim(-xlim, xlim)
    ax[0].set_title('Subspace 1 \n (ectopic beats detection)')
    ax[0].legend()

    ##############
    # Second panel
    ##############

    # Plot normal beats
    ax[1].scatter(artefacts['subspace1'][normalBeats],
                  artefacts['subspace3'][normalBeats],
                  color='gray', edgecolors='k', alpha=0.2,
                  zorder=10, s=15, label='Normal')

    # Plot outliers
    ax[1].scatter(artefacts['subspace1'][artefacts['ectopic']],
                  artefacts['subspace3'][artefacts['ectopic']],
                  color='r', edgecolors='k', zorder=10, label='Ectopic')
    ax[1].scatter(artefacts['subspace1'][artefacts['short']],
                  artefacts['subspace3'][artefacts['short']],
                  color='b', edgecolors='k', zorder=10,
                  marker='s', label='Short')
    ax[1].scatter(artefacts['subspace1'][artefacts['long']],
                  artefacts['subspace3'][artefacts['long']],
                  color='g', edgecolors='k', zorder=10,
                  marker='s', label='Long')
    ax[1].scatter(artefacts['subspace1'][artefacts['missed']],
                  artefacts['subspace3'][artefacts['missed']],
                  color='g', edgecolors='k', zorder=10, label='Missed')
    ax[1].scatter(artefacts['subspace1'][artefacts['extra']],
                  artefacts['subspace3'][artefacts['extra']],
                  color='b', edgecolors='k', zorder=10, label='Extra')
    # Upper area
    ax[1].plot([-1, -10], [1, 1], 'k', linewidth=1, linestyle='--')
    ax[1].plot([-1, -1], [1, 10], 'k', linewidth=1, linestyle='--')
    x = [-10, -10, -1, -1]
    y = [1, 10, 10, 1]
    ax[1].fill(x, y, color='gray', alpha=0.3)

    # Lower area
    ax[1].plot([1, 10], [-1, -1], 'k', linewidth=1, linestyle='--')
    ax[1].plot([1, 1], [-1, -10], 'k', linewidth=1, linestyle='--')
    x = [1, 1, 10, 10]
    y = [-1, -10, -10, -1]
    ax[1].fill(x, y, color='gray', alpha=0.3)

    ax[1].set_xlabel('Subspace $S_{21}$')
    ax[1].set_ylabel('Subspace $S_{22}$')
    ax[1].set_ylim(-ylim*2, ylim*2)
    ax[1].set_xlim(-xlim, xlim)
    ax[1].set_title('Subspace 2 \n (long and short beats detection)')
    ax[1].legend()

    plt.tight_layout()

    return ax


def plot_psd(x, sfreq=5, method='welch', fbands=None, low=0.003,
             high=0.4, show=True, ax=None):
    """Plot PSD of heart rate variability.

    Parameters
    ----------
    x : 1d array-like
        Length of R-R intervals (default is in miliseconds).
    sfreq : int
        The sampling frequency.
    method : str
        The method used to extract freauency power. Default set to `'welch'`.
    fbands : None or dict, optional
        Dictionary containing the names of the frequency bands of interest
        (str), their range (tuples) and their color in the PSD plot. Default is
        {'vlf': ['Very low frequency', (0.003, 0.04), 'b'],
        'lf': ['Low frequency', (0.04, 0.15), 'g'],
        'hf': ['High frequency', (0.15, 0.4), 'r']}
    show : bool
        Plot the power spectrum density. Default is *True*.
    ax : `Matplotlib.Axes` or None
        Where to draw the plot. Default is *None* (create a new figure).

    Returns
    -------
    ax or (freq, psd) : `Matplotlib.Axes` or 1d array-like
        If show is *True*, return the PSD plot. If show is *False*, will return
        the frequencies and PSD level as arrays.
    """
    # Interpolate R-R interval
    time = np.cumsum(x)
    f = interp1d(time, x, kind='cubic')
    new_time = np.arange(time[0], time[-1], 1000/sfreq)  # Sampling rate = 5 Hz
    x = f(new_time)

    if method == 'welch':

        # Define window length
        nperseg = 256 * sfreq
        if nperseg > len(x):
            nperseg = len(x)

        # Compute Power Spectral Density
        freq, psd = welch(x=x, fs=sfreq, nperseg=nperseg, nfft=nperseg*10)

        psd = psd/1000000

    if fbands is None:
        fbands = {'vlf': ['Very low frequency', (0.003, 0.04), '#4c72b0'],
                  'lf':	['Low frequency', (0.04, 0.15), '#55a868'],
                  'hf':	['High frequency', (0.15, 0.4), '#c44e52']}

    if show is True:
        # Plot the PSD
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(freq, psd, 'k')
        for f in ['vlf', 'lf', 'hf']:
            mask = (freq >= fbands[f][1][0]) & (freq <= fbands[f][1][1])
            ax.fill_between(freq, psd, where=mask, color=fbands[f][2],
                            alpha=0.8)
            ax.axvline(x=fbands[f][1][0],
                       linestyle='--',
                       color='gray')
        ax.set_xlim(0.003, 0.4)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('PSD [$s^2$/Hz]')
        ax.set_title('Power Spectral Density', fontweight='bold')

        return ax
    else:
        return freq, psd


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
       from systole.plotting import circular
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
    data : `pd.DataFrame`
        Angular data (rad.).
    y : str or list
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
       from systole.plotting import plot_circular
       x = np.random.normal(np.pi, 0.5, 100)
       y = np.random.uniform(0, np.pi*2, 100)
       data = pd.DataFrame(data={'x': x, 'y': y}).melt()
       plot_circular(data=data, y='value', hue='variable')
    """
    # Check data format
    if isinstance(data, pd.DataFrame):
        assert data.shape[0] > 0, 'Data must have at least 1 row.'

    palette = itertools.cycle(sns.color_palette('deep'))

    if hue is None:
        ax = circular(data[y].values, **kwargs)

    else:
        n_plot = data[hue].nunique()

        fig, axs = plt.subplots(1, n_plot, subplot_kw=dict(projection='polar'))

        for i, cond in enumerate(data[hue].unique()):

            x = data[y][data[hue] == cond]

            ax = circular(x, color=next(palette), ax=axs[i], **kwargs)

    return ax
