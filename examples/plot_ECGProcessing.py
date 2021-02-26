"""
ECG preprocessing and R wave detection
======================================

This notebook describe ECG signal processing, from R wave detection to heart
rate variability and evoked heart rate activity.
"""

# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
# Licence: GPL v3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from systole import import_dataset
from systole.detection import ecg_peaks
from systole.utils import heart_rate, to_epochs

#%%
# Loading ECG dataset
# -------------------
signal_df = import_dataset()

#%%
# Finding R peaks
# ---------------
# The peaks detection algorithms are imported from the py-ecg-detectors module:
# https://github.com/berndporr/py-ecg-detectors
signal, peaks = ecg_peaks(signal_df.ecg, method="hamilton", sfreq=1000, find_local=True)

#%%
# Heart Rate Variability
# ----------------------

#%%
# Event related cardiac deceleration
# ----------------------------------

# Extract instantaneous heart rate
heartrate, new_time = heart_rate(peaks, kind="cubic", unit="bpm")

# Downsample the stim events channel
# to fit with the new sampling frequency (1000 Hz)
neutral, disgust = np.zeros(len(new_time)), np.zeros(len(new_time))

disgust[np.round(np.where(signal_df.stim.to_numpy() == 2)[0]).astype(int)] = 1
neutral[np.round(np.where(signal_df.stim.to_numpy() == 3)[0]).astype(int)] = 1

#%%
# Event related plot
# ------------------
sns.set_context("talk")
fig, ax = plt.subplots(figsize=(8, 5))
for cond, data, col in zip(
    ["Neutral", "Disgust"],
    [neutral, disgust],
    [sns.xkcd_rgb["denim blue"], sns.xkcd_rgb["pale red"]],
):

    # Epoch intantaneous heart rate
    # and downsample to 2 Hz to save memory
    epochs = to_epochs(heartrate, data, tmin=0, tmax=11)[:, ::500]

    # Plot
    df = pd.DataFrame(epochs).melt()
    df.variable /= 2
    sns.lineplot(data=df, x="variable", y="value", ci=68, label=cond, color=col, ax=ax)

ax.set_xlim(0, 10)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Heart Rate (BPM)")
ax.set_title("Instantaneous heart rate after neutral and disgusting images")
sns.despine()
plt.tight_layout()


#%%
# References
# ----------
# .. [#] Porr, B., & Howell, L. (2019). R-peak detector stress test with a new
# noisy ECG database reveals significant performance differences amongst
# popular detectors. Cold Spring Harbor Laboratory.
# https://doi.org/10.1101/722397
