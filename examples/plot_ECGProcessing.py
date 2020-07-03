"""
ECG preprocessing and R wave detection
======================================

This example shows how to extract peaks from ECG recording. The peaks detection
algorithms are implemented by the py-ecg-detectors module:
https://github.com/berndporr/py-ecg-detectors
"""

# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
# Licence: GPL v3

import numpy as np
import matplotlib.pyplot as plt
from systole import import_dataset
from systole.detection import ecg_peaks
from systole.utils import heart_rate, to_neighbour

#%%
# Loading ECG dataset
# -------------------
data = import_dataset()


segment = data.ecg[-2000*5:]


#%%
# Finding R peaks
# ---------------
signal, peaks = ecg_peaks(segment, method='moving-average', sfreq=2000)
peaks = to_neighbour(signal, peaks, size=100)
time = np.arange(0, len(signal))/1000
plt.plot(time, signal)
plt.plot(time[peaks], signal[peaks], 'ro')


(60000/np.diff(np.where(peaks)[0])).mean()

hr, time = heart_rate(peaks)
plt.subplot(211)
plt.plot(time, hr)
plt.subplot(212)
plt.plot(time, signal)
plt.plot(time[peaks], signal[peaks], 'ro')


#%%
# Compare different methods
# -------------------------
# Here, we are going to estimate instantaneous heartrate from a 20 minutes
# recording and compare the output from different methods.
plt.figure(figsize=(13, 5))
for method in ['hamilton', 'christov', 'engelse-zeelenberg', 'pan-tompkins',
               'wavelet-transform', 'moving-average']:

    signal, peaks = ecg_peaks(segment, method=method, sfreq=2000)
    hr, time = heart_rate(peaks)
    plt.plot(time, hr, label=method)

plt.legend()
plt.ylim(400, 1200)
plt.xlim(600, 630)


#%%
# References
# ----------
# .. [#] Porr, B., & Howell, L. (2019). R-peak detector stress test with a new
# noisy ECG database reveals significant performance differences amongst
# popular detectors. Cold Spring Harbor Laboratory.
# https://doi.org/10.1101/722397
