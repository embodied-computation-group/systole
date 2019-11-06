import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_events(events, sfreq=None, palette=None):
    """Given a peaks vector, returns frequency plots.

    Parameters
    ----------
    events : array like
        The events. Should be in the form: [0] times * sample, values different
        from 0 encoding events.

    Returns
    -------
    ax : Matplotlib instance
        Figure.
    """
    if isinstance(events, list):
        events = np.asarray(events)

    fig, ax = plt.subplots()
    if palette is None:
        current_palette = sns.color_palette()
    else:
        current_palette = palette

    for i in np.unique(events):
        if i != 0:
            trig = np.where(events == i)
            trig = np.append(trig, len(events))
            for ii in range(0, len(trig)-1):
                if ii % 2 == 0:
                    alpha = 0.5
                else:
                    alpha = 1
                plt.fill_between(x=np.arange(trig[ii], trig[ii+1]),
                                 y1=i,
                                 y2=i+1, alpha=alpha, color=current_palette[i])
    return ax

def plot_oximeter(oximeter):
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
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(oximeter.times, oximeter.threshold, linestyle='--', color='gray',
             label='Threshold')
    plt.fill_between(x=oximeter.times,
                     y1=oximeter.threshold,
                     y2=np.asarray(oximeter.recording).min(),
                     alpha=0.2,
                     color='gray')
    plt.plot(oximeter.times, oximeter.recording, label='Recording')
    plt.fill_between(x=oximeter.times,
                     y1=oximeter.recording,
                     y2=np.asarray(oximeter.recording).min(),
                     color='w')
    plt.plot(np.asarray(oximeter.times)[np.where(oximeter.peaks)[0]],
             np.asarray(oximeter.recording)[np.where(oximeter.peaks)[0]],
             'ro', label='Online estimation')
    plt.ylabel('PPG level', size=20)
    plt.xlabel('Time (s)', size=20)
    plt.title('PPG recording', size=25)
    plt.legend()

    return fig, ax


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


# def plot_psd(x):
#     """Plot power spectal density.
#
#     Parameters
#     ----------
#     freq : array
#         The frequencies array.
#     psd : array
#         Power spectral density.
#     Returns
#     -------
#     ax, fig : Matplotlib instances
#         The power spectral density plot.
#     """
#     freqs, psd = welch(x, sfreq=5000, low=0.01)
#
#     freq_range = [0.02, 0.1]
#     f_range = (freqs > freq_range[0]) & (freqs < freq_range[1])
#
#     freq_bands = [0.033, 0.066]
#     f_band = (freqs < freq_bands[1]) & (freqs > freq_bands[0])
#
#     plt.figure(figsize=(7, 4))
#     plt.plot(freqs[f_range], psd[f_range], lw=2, color='k')
#     plt.fill_between(freqs[f_band], psd[f_band],
#                      color='skyblue')
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Power spectral density (uV^2 / Hz)')
#     # plt.xlim(freq_range)
#     # plt.ylim(psd[freqs[f_band]])
#     plt.title("Welch's periodogram")
#     sns.despine()
