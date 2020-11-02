#!/usr/bin/env python
# coding: utf-8

# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

# In[1]:


# Execute this cell to install systole when running an interactive session
#%%capture
#! pip install git+https://github.com/embodied-computation-group/systole.git@dev


# In[2]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from systole.detection import oxi_peaks, interpolate_clipping, ecg_peaks
from systole.plotly import plot_raw, plot_frequency, plot_subspaces, plot_timedomain, plot_nonlinear
from systole import import_dataset1, import_ppg
from systole.utils import heart_rate, to_epochs
from systole.hrv import frequency_domain

from IPython.display import Image
from IPython.core.display import HTML 

sns.set_context('talk')


# # Introduction to cardiac signal analysis for cognitive science

# ## Electrocardiography (ECG)

# In[3]:


Image(url='https://github.com/embodied-computation-group/systole/raw/dev/notebooks/images/ecg.png', width=800)


# First, we load a dataset containing ECG and respiratory data. These data were acquired using the paradigm described in Legrand et al. (2020). ECG, EDA and Respiration were acquired while the participant watched neutral and disgusting images. Here, we only need ECG respiration and the stim channel (encoding the presentation of the images), we then only provide these arguments in the `modalities` parameter to speed up download and save memory.

# In[4]:


ecg_df = import_dataset1(modalities=['ECG', 'Respiration', 'Stim'])


# ### R peaks detection

# Here, we will use the `ecg_peaks` function to perform R peak detection.  The peaks detection algorithms are imported from the [py-ecg-detectors module](https://github.com/berndporr/py-ecg-detectors) (Porr & Howell, 2019), and the method can be selected via the `ecg_method` parameter among the following: `hamilton`, `christov`, `engelse-zeelenberg`, `pan-tompkins`, `wavelet-transform`, `moving-average`. In this tutorial, we will use the [pan-tompkins algorithm](https://en.wikipedia.org/wiki/Pan%E2%80%93Tompkins_algorithm).

# In[5]:


signal = ecg_df[(ecg_df.time > 500) & (ecg_df.time < 530)].ecg.to_numpy()  # Select the first minute of recording
signal, peaks = ecg_peaks(signal, method='pan-tompkins', sfreq=1000)


# This function will output the resampled `signal` (only if the sampling frequency of the input signal was different from 1000 Hz), and a `peaks` vector. The peaks vector has the same size than the input signal and is a boolean array (only contain 0/False and 1/True). The R peaks are encoded as 1 and the rest is set to 0. This vector can then be used to plot the detected R peaks on the input signal (panel **1** below). We can also use this vector to compute the distance between each R peaks (the R-R interval, see panel **2** below), which is used to measure the instantaneous heart rate (see panel **3** below).

# In[6]:


time = np.arange(0, len(signal))/1000

fig, axs = plt.subplots(3, 1, figsize=(18, 9), sharex=True)

axs[0].plot(time, signal, color='#c44e52')
axs[0].scatter(time[peaks], signal[peaks], color='gray')
axs[0].set_ylabel('ECG (mV)')

axs[1].plot(time, peaks, color='g')
axs[1].set_ylabel('R peaks')

axs[2].plot(time[peaks][1:], np.diff(np.where(peaks)[0]), color='#4c72b0', linestyle='--')
axs[2].scatter(time[peaks][1:], np.diff(np.where(peaks)[0]), color='#4c72b0')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('R-R interval (ms)')
sns.despine()


# The R-R intervals are expressed in milliseconds, but can also be converted into beats per minutes (BPM) using the following formula:
# $$BPM = \frac{60000}{RR_{Intervals}} $$
# It is worth noting that the conversion from RR intervals to beats per minute involves a non-linear function of the form $\frac{1}{x}$. This should be taken into account when interpreting the amplitude of heart rate increase/decrease with different baseline. For example, increasing the heart rate from 40 to 45 bpm diminish RR interval by approximately 167ms, while increasing the heart rate from 120 to 125 bpm decrease RR intervals by only 20ms (see graph below). While these differences have only marginal influence in most of the circumstances, this should be taken into account when measuring heart rate variability based on absolute RR interval difference, which is the case of some time-domain metric (e.g. the pNN50 or pNN20).

# In[7]:


rr = np.arange(300, 2000, 2)
plt.figure(figsize=(8, 8))
plt.plot(60000/rr, rr)

plt.plot([25, 40], [60000/40, 60000/40], 'gray')
plt.plot([25, 45], [60000/45, 60000/45], 'gray')

plt.plot([25, 120], [60000/120, 60000/120], 'gray')
plt.plot([25, 125], [60000/125, 60000/125], 'gray')

plt.xlabel('BPM')
plt.ylabel('RR intervals (ms)')
plt.title('Converting RR intervals to BPM')
sns.despine()


# ### Time series visualization

# We can use the interactive plot to visualize this signal and automatically detected R peaks using the `plot_raw()` function.

# In[8]:


plot_raw(ecg_df[(ecg_df.time>500) & (ecg_df.time<550)], type='ecg', ecg_method='pan-tompkins')


# ## Photoplethysmography (PPG)

# [Photoplethysmography](https://en.wikipedia.org/wiki/Photoplethysmogram) is a non-invasive method used to measure the change of blood volume. The PPG signal is characterized by a main **systolic peak**, often (but not always) followed by a smaller **diastolic peaks** before the signal return to origin. The lower point between the systolic and diastolic peak is the **dicrotic notch**. The systolic peaks correspond to the moment where the volume of blood in the blood vessel suddenly increase due to the pressure of the heart contraction. The blood volume measured at the periphery does not change immediately after the cardiac systole, but rather with a delay varying depending on physiological parameters. For this reason, the systolic peak and the R peak are not concomitant but rather delayed, the systolic peak often occurring the T wave of the ECG. The delay between the R wave on the ECG and the systolic peak can vary between individuals and across time in the same individual.

# In[9]:


Image(url='https://github.com/embodied-computation-group/systole/raw/dev/notebooks/images/pulseOximeter.png', width=1200)


# First, we import an example signal. This time serie represent a PPG recording from pulse oximeter in a young health participant. The sampling frequecy is 75 Hz (75 data points/seconds)

# In[10]:


ppg = import_ppg()


# ### Systolic peak detection

# The main information of interest we can retrieve from the PPG signal is the timing of occurrence of the systolic peak. The timing of these events is indeed tightly linked to the occurrence of the R wave (although with a slightly variable delay), and we can use the peak vector resulting from this analysis the same way we analyse the RR interval time-series with an ECG signal. Because we are not measuring the heart rate strictly speaking, but the pulse at the periphery of the body, this approach is often termed **pulse rate variability** to distinguish from the heart rate variability that builds on the ECG signal.

# In[11]:


Image(url='https://github.com/embodied-computation-group/systole/raw/dev/notebooks/images/ppgRecording.png', width=1200)


# As before, you can plot the time serie and visualize peak detection and the inter beat interval time series using the `plot_raw` function.

# In[12]:


plot_raw(ppg)


# ### Clipping artefacts

# Clipping is a form of distortion that can limit signal when it exceeds a certain threshold [see the Wikipedia page](https://en.wikipedia.org/wiki/Clipping_(signal_processing)). Some device can produce clipping artefacts when recording the PPG signal. Here, we can see that some clipping artefacts are found between 100 and 150 seconds in the previous recording. The threshold values (here `255`), is often set by the device and can easily be found. These artefacts should be corrected before systolic peaks detection. One way to go is to remove the portion of the signal where clipping artefacts occurs and use cubic spline interpolation to reconstruct a plausible estimate of the *real* underlying signal. This is what the function `interpolate_clipping()` do.

# In[13]:


signal = ppg[(110 < ppg.time) & (ppg.time < 113)].ppg.to_numpy()  # Extract a portion of signal with clipping artefacts
clean_signal = interpolate_clipping(signal, threshold=255)  # Remove clipping segment and interpolate missing calues

sns.set_context('talk')
plt.figure(figsize=(13, 5))
plt.plot(np.arange(0, len(clean_signal))/75, clean_signal, label='Corrected PPG signal', linestyle= '--', color='#c44e52')
plt.plot(np.arange(0, len(signal))/75, signal, label='Raw PPG signal', color='#4c72b0')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('PPG level (a.u)')
sns.despine()
plt.tight_layout()


# ## Heart rate variability

# In[14]:


signal, peaks = ecg_peaks(ecg_df[(ecg_df.time>500) & (ecg_df.time<550)].ecg, method='pan-tompkins', sfreq=1000, find_local=True)
rr = np.diff(np.where(peaks)[0])


# ### Time domain

# In[15]:


plot_timedomain(rr)


# ### Frequency domain

# In[16]:


plot_frequency(rr)


# ### Non linear domain

# In[17]:


plot_nonlinear(rr)


# ## Instantaneous heart rate

# Finding R peaks.

# In[18]:


signal, peaks = ecg_peaks(ecg_df.ecg, method='pan-tompkins', sfreq=1000, find_local=True)


# Extract instantaneous heart rate

# In[19]:


heartrate, new_time = heart_rate(peaks, kind='previous', unit='bpm')


# Downsample the stim events channel to fit with the new sampling frequency (1000 Hz)

# In[20]:


neutral, disgust = np.zeros(len(new_time)), np.zeros(len(new_time))

disgust[np.round(np.where(ecg_df.stim.to_numpy() == 2)[0]).astype(int)] = 1
neutral[np.round(np.where(ecg_df.stim.to_numpy() == 1)[0]).astype(int)] = 1


# Event related plot.

# In[21]:


sns.set_context('talk')
fig, ax = plt.subplots(figsize=(8, 5))
for cond, data, col in zip(
        ['Neutral', 'Disgust'], [neutral, disgust],
        [sns.xkcd_rgb["denim blue"], sns.xkcd_rgb["pale red"]]):

    # Epoch intantaneous heart rate
    # and downsample to 2 Hz to save memory
    epochs = to_epochs(heartrate, data, tmin=0, tmax=10)[:, ::500]

    # Plot
    df = pd.DataFrame(epochs).melt()
    df.variable /= 2
    sns.lineplot(data=df, x='variable', y='value', ci=68, label=cond,
                 color=col, ax=ax, markers=['--', '-o'])

    plt.scatter(np.arange(0, 10, .5), epochs.mean(0), color=col)

#ax.set_xlim(0, 10)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Heart Rate (BPM)')
ax.set_title('Instantaneous heart rate during picture presentation')
sns.despine()
plt.tight_layout()


# ## Artefacts correction

# Cardiac signals ccan be noisy, either due to artefacts in the signal, invalid peaks detection or even ECG signal. We often distinguish between three kind of RR artefacts:
# * Missing R peaks / long beats
# * Extra R peaks or short beats
# * Ectopi beats forming negative positive negative (NPN) or positive negative positive (PNP) segments.

# Metrics of heart rate variability are highly influenced by RR artefacts, being missing, extra or ectopic beats. The figure bellow exemplify the addition of artefacts to time and frequency domain of heart rate variability.

# In[22]:


plot_subspaces(rr)


# # References

# van Gent, P., Farah, H., van Nes, N., & van Arem, B. (2019). HeartPy: A novel heart rate algorithm for the analysis of noisy signals. *Transportation Research Part F: Traffic Psychology and Behaviour, 66, 368–378*. https://doi.org/10.1016/j.trf.2019.09.015
# - **Description of a simple systolic peak detection algorithm and clipping artefacts correction**.
# 
# Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for heart rate variability time series artefact correction using novel beat classification. *Journal of Medical Engineering & Technology, 43(3), 173–181*. https://doi.org/10.1080/03091902.2019.1640306
# - **This paper describe the method for artefact detection and artefact correction based on the first and second derivative of RR intervals**.
# 
# Porr, B., & Howell, L. (2019). R-peak detector stress test with a new noisy ECG database reveals significant performance differences amongst popular detectors. Cold Spring Harbor Laboratory. https://doi.org/10.1101/722397
# - **Recent implementation of famous R peak detection algorithms tested against large dataset**.
# 

# In[ ]:




