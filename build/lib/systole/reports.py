# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import matplotlib.pyplot as plt
from systole.detection import oxi_peaks, artefact_correction
from systole.hrv import hrv_psd
from systole.utils import heart_rate


def report_oxi(signal, file_name='report.png', dpi=600):
    """Generate HRV report.

    Parameters
    ----------
    signal : 1d array-like
        PPG singal.
    file_name : str
        Output file name.
    dpi : int
        Image quality.
    """
    # Find peaks
    signal, peaks = oxi_peaks(signal)

    # Artifacts removal
    clean_peaks, per = artefact_correction(peaks)

    # Extract instantaneous heartrate
    sfreq = 1000
    noisy_hr, time = heart_rate(peaks, sfreq=sfreq, unit='bpm', kind='cubic')
    hr, time = heart_rate(clean_peaks, sfreq=sfreq, unit='bpm', kind='cubic')

    time = np.arange(0, len(signal)/sfreq, 1/sfreq)

    fig = plt.figure(figsize=(8, 13))
    gs = fig.add_gridspec(4, 3)
    fig_ax1 = fig.add_subplot(gs[0, :])

    fig_ax1.plot(time, signal, linewidth=.2)
    fig_ax1.plot(time[np.where(clean_peaks)[0]],
                 signal[np.where(clean_peaks)[0]], 'ro', markersize=0.8)
    fig_ax1.set_ylabel('PPG level')
    fig_ax1.set_xlim(0, time[-1])
    fig_ax1.set_title('Signal', fontweight='bold')

    fig_ax2 = fig.add_subplot(gs[1, :])
    fig_ax2.plot(time, noisy_hr, linewidth=.8, color='r')
    fig_ax2.plot(time, hr, linewidth=.8, color='g')
    fig_ax2.set_ylabel('BPM')
    fig_ax2.set_xlabel('Time (s)')
    fig_ax2.set_xlim(0, time[-1])
    fig_ax2.set_title('RR time-course', fontweight='bold')

    # HRV
    rr = np.diff(np.where(peaks)[0])

    fig_ax3 = fig.add_subplot(gs[2, 0])
    fig_ax3.hist(rr, bins=30, alpha=.5)
    fig_ax3.hist(rr[(rr <= 400) | (rr >= 1500)], bins=30, color='r')
    fig_ax3.set_title('Distribution', fontweight='bold')
    fig_ax3.set_ylabel('Count')
    fig_ax3.set_xlabel('RR(s)')

    fig_ax4 = fig.add_subplot(gs[2, 1])
    hrv_psd(rr, show=True, ax=fig_ax4)

    fig_ax5 = fig.add_subplot(gs[2, 2])
    fig_ax5.plot(rr[:-1], rr[1:], color='gray', markersize=1, alpha=0.8)
    fig_ax5.set_title('Pointcare Plot', fontweight='bold')
    fig_ax5.set_ylabel('$RR_{n+1}$')
    fig_ax5.set_xlabel('$RR_n$')
    plt.tight_layout(h_pad=0.05)
    if file_name is not None:
        plt.savefig(file_name, dpi=dpi)
    plt.close()
