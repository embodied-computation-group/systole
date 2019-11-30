# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import welch


def hrv_psd(x, sfreq=5, method='welch', fbands=None, low=0.003,
            high=0.4, show=True):
    """Plot PSD of heart rate variability.

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
    f = interpolate.interp1d(time, x, kind='cubic')
    new_time = np.arange(time[0], time[-1], 1000/sfreq)  # Sampling rate = 5 Hz
    x = f(new_time)

    if method == 'welch':

        # Define window length
        nperseg = 256 * sfreq

        # Compute Power Spectral Density
        freq, psd = welch(x=x, fs=sfreq, nperseg=nperseg, nfft=nperseg)

        psd = psd/1000000

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
        ax.set_ylabel('PSD [$s^2$/Hz]', size=15)
        ax.set_title('Power Spectral Density', size=20)

        return ax
    else:
        return freq, psd


def frequency_domain(x, sfreq=5, method='welch', fbands=None):
    """Extract the frequency domain features of heart rate variability.

    Parameters
    ----------
    x : list or numpy array
        Length of R-R intervals (in miliseconds).
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

    Returns
    -------
    stats : pandas DataFrame
        DataFrame of HRV parameters (frequency domain)
    """
    # Interpolate R-R interval
    time = np.cumsum(x)
    f = interpolate.interp1d(time, x, kind='cubic')
    new_time = np.arange(time[0], time[-1], 1000/sfreq)  # Sampling rate = 5 Hz
    x = f(new_time)

    if method == 'welch':

        # Define window length
        nperseg = 256 * sfreq

        # Compute Power Spectral Density
        freq, psd = welch(x=x, fs=sfreq, nperseg=nperseg, nfft=nperseg)

        psd = psd/1000000

    if fbands is None:
        fbands = {'vlf': ['Very low frequency', (0.003, 0.04), 'b'],
                  'lf':	['Low frequency', (0.04, 0.15), 'g'],
                  'hf':	['High frequency', (0.15, 0.4), 'r']}

    # Extract HRV parameters
    ########################
    stats = pd.DataFrame([])
    for band in fbands:
        this_psd = psd[
            (freq >= fbands[band][1][0]) & (freq < fbands[band][1][1])]
        this_freq = freq[
            (freq >= fbands[band][1][0]) & (freq < fbands[band][1][1])]

        # Peaks (Hz)
        peak = round(this_freq[np.argmax(this_psd)], 4)
        stats = stats.append({'Values': peak, 'Metric': band+'_peak'},
                             ignore_index=True)

        # Power (ms**2)
        power = np.trapz(x=this_freq, y=this_psd) * 1000000
        stats = stats.append({'Values': power, 'Metric': band+'_power'},
                             ignore_index=True)

    hf = stats.Values[stats.Metric == 'hf_power'].values[0]
    lf = stats.Values[stats.Metric == 'lf_power'].values[0]
    vlf = stats.Values[stats.Metric == 'vlf_power'].values[0]

    # Power (%)
    power_per_vlf = vlf/(vlf+lf+hf)*100
    power_per_lf = lf/(vlf+lf+hf)*100
    power_per_hf = hf/(vlf+lf+hf)*100

    # Power (n.u.)
    power_nu_hf = hf/(hf + lf)
    power_nu_lf = lf/(hf + lf)

    values = [power_per_vlf, power_per_lf, power_per_hf,
              power_nu_hf, power_nu_lf]
    metrics = ['pover_vlf_per', 'pover_lf_per', 'pover_hf_per',
               'pover_lf_nu', 'pover_hf_nu']

    stats = stats.append(pd.DataFrame({'Values': values, 'Metric': metrics}),
                         ignore_index=True, sort=False)

    return stats
