# Set of functions used to extract the frequency domain features of ECG
# recordings. Method available: Welch.

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import welch


def hrv_frequency(x, sfreq=1000, method='welch', fbands=None, low=0.003,
                  high=0.4, show=True):
    """Extract the frequency domain features of ECG signals.

    Parameters
    ----------
    x : list or numpy array
        Length of R-R intervals (default is in miliseconds).
    sfreq : int
        The sampling frequency.
    method : str
        The method used to extract freauency power. Default set to `'welch'`.
    fbands : None | dict, optional
        Dictionary containing the names of the frequency bands of interest
        (str), their range (tuples) and their color in the PSD plot. Default is
        {'vlf': ['Very low frequency', (0.003, 0.04), 'b'],
        'lf': ['Low frequency', (0.04, 0.15), 'g'],
        'hf': ['High frequency', (0.15, 0.4), 'r']}
    show : boolean
        Plot the power spectrum density. Default is `True`.

    Returns
    -------
    ax | freq, psd : Matplotlib instance | numpy array
        If `show=True`, return the PSD plot. If `show=False`, will return the
        frequencies and PSD level as arrays.
    """
    # Interpolate R-R interval
    time = np.cumsum(x)
    f = interpolate.interp1d(time, x)
    new_time = np.arange(time[0], time[-1], 1)
    x = f(new_time)

    if method == 'welch':

        # Define window length
        nperseg = (2 / low) * sfreq

        # Compute Power Spectral Density
        freq, psd = welch(x=x, fs=sfreq, nperseg=nperseg, nfft=None)

        psd = psd/1000000

    if method == 'AR':
        print('Not available yet')

    if fbands is None:
        fbands = {'vlf': ['Very low frequency', (0.003, 0.04), 'b'],
                  'lf':	['Low frequency', (0.04, 0.15), 'g'],
                  'hf':	['High frequency', (0.15, 0.4), 'r']}

    if show is True:
        # Plot the PSD
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(freq, psd, 'k')
        for f in ['vlf', 'lf', 'hf']:
            mask = (freq > fbands[f][1][0]) & (freq <= fbands[f][1][1])
            ax.fill_between(freq, psd, where=mask, alpha=0.5,
                            color=fbands[f][2])
            ax.axvline(x=fbands[f][1][0],
                       linestyle='--',
                       color='gray')
        ax.set_xlim(0.003, 0.4)
        ax.set_xlabel('Frequency [Hz]', size=15)
        ax.set_ylabel('PSD [V**2/Hz]', size=15)
        ax.set_title('Power Spectral Density', size=20)

        return ax
    else:
        return freq, psd
