---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(cardiac_cycles)=
# Detecting cardiac cycles
Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

```{code-cell} ipython3
:tags: [hide-input]

%%capture
import sys
if 'google.colab' in sys.modules:
    ! pip install systole
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from systole.detection import interpolate_clipping, ecg_peaks
from systole.plots import plot_raw
from systole import import_dataset1, import_ppg
from systole import serialSim
from systole.recording import Oximeter

from bokeh.io import output_notebook
from bokeh.plotting import show
output_notebook()

sns.set_context('talk')
```

This notebook focuses on the characterisation of cardiac cycles from the physiological signals we previously described (PPG and ECG). Here we only briefly mention different QRS-detection algorithms as related to the estimation of heart rate frequency, which is a commonly used measure in psychology and cognitive science. However, both the ECG and the PPG signals contain rich information beyond the beat to beat frequency that is relevant for cognitive and physiological modelling. All these approaches are not fully covered by [Systole](https://embodied-computation-group.github.io/systole/#), but are implemented in other software (e.g see {cite:p}`2021:makowski` for ECG components delineation).

In this notebook, we are going to review the peak detection algorithm, which future tutorials covering, for example, heart-rate variability will build upon. Here our intention is to provide a better intuition of what is happening in our peak detection algorithms, the corrections that can be applied, and the possible bias and artefacts that can emerge. However, you do not need a perfect understanding of all these steps if you want to apply peak detection to youre data, and you might also consider skipping this part to proceed directly to the next tutorial focusing on artefact detection and correction.

+++

## Electrocardiography

+++

Because we will ultimately be interested in heart rate and its variability, our first goal will be to detect the R peaks. We will use the `ecg_peaks()` function provided by Systole to perform R peak detection. This function is a simple wrapper for well-known peak detection algorithms that are implemented in the [py-ecg-detectors module](https://github.com/berndporr/py-ecg-detectors) {cite:p}`2019:porr`. The detection algorithm can be selected via the `ecg_method` parameter, it should be among the following: `hamilton`, `christov`, `engelse-zeelenberg`, `pan-tompkins`, `wavelet-transform`, `moving-average`. In this tutorial, we will use the [pan-tompkins algorithm](https://en.wikipedia.org/wiki/Pan%E2%80%93Tompkins_algorithm) {cite:p}`1985:pan` as it is a fast, well-perfoming, and commonly used algorithm for QRS detection.

+++

Let's first load an ECG recording. Here, we will select a 5 minute interval and compare the performances of the different algorithms supported by Systole.

```{code-cell} ipython3
# Import ECg recording
ecg_df = import_dataset1(modalities=['ECG'], disable=True)
signal = ecg_df[ecg_df.time.between(60, 360)].ecg.to_numpy()  # Select 5 minutes
```

### Detecting R peaks
The main feature that we can extract from the ECG recording is the R wave (see image in notebook 1). A large variety of algorithms have been proposed to extract the timing of R waves while controlling for signal noise and physiological variability. Reviewing and comparing all of the available methods is beyond the scope of this tutorial. Here, we are going to restrict our focus to some of the most popular methods, which also are available in the Python opensource ecosystem (`hamilton`, `christov`, `engelse-zeelenberg`, `pan-tompkins`, and `moving-average`. These methods were implemented originally in the [py-ecg-detectors module](https://github.com/berndporr/py-ecg-detectors) - Systole includes a modified version that runs with the [Numba](http://numba.pydata.org/) package for better performance, resulting in 7-30x faster estimation, depending on the algorithm.

```{code-cell} ipython3
from systole.detectors import pan_tompkins, hamilton, moving_average, christov, engelse_zeelenberg
```

#### Pan-Tompkins

The [Pan-Tompkins algorithm](https://en.wikipedia.org/wiki/Pan%E2%80%93Tompkins_algorithm) was introduced in 1985. It uses band-pass filtering and derivation to identify the QRS complex and apply an adaptive threshold to detect the R peaks on the filtered signal.

This is a very popular - maybe to most popular - method for R peaks detection. One of the advantages is that it can easily be applied for online detection (see [this implementation](https://github.com/c-labpl/qrs_detector) for example). As we can see in the timing report, this algorithm is also the fastest among those available in the Systole toolbox, making ideal for applications such as online peak detection and biofeedback. You can also see that the algorithm's peak detection is highly accurate as we do not see any artifacts in the estimated R-R time series. One notable exception is the first peak which has been estimated incorrectly. This is due to the filtering steps in the Pan-Tompkins algorithms, and can be corrected by using a slightly longer signal than the range of interest.

```{code-cell} ipython3
%%timeit
peaks_pt = pan_tompkins(signal, sfreq=1000)
```

```{code-cell} ipython3
show(
    plot_raw(signal, modality='ecg', ecg_method='pan-tompkins', backend='bokeh', show_heart_rate=True)
)
```

#### Moving average

```{code-cell} ipython3
%%timeit
peaks_wa = ecg_peaks(signal, sfreq=1000, method="moving-average")
```

```{code-cell} ipython3
show(
    plot_raw(signal, modality='ecg', ecg_method='moving-average', backend='bokeh', show_heart_rate=True)
)
```

#### Hamilton

```{code-cell} ipython3
%%timeit
peaks_ha = hamilton(signal, sfreq=1000)
```

```{code-cell} ipython3
show(
    plot_raw(signal, modality='ecg', ecg_method='hamilton', backend='bokeh', show_heart_rate=True)
)
```

#### Christov

```{code-cell} ipython3
%%timeit
peaks_ch = christov(signal, sfreq=1000)
```

```{code-cell} ipython3
show(
    plot_raw(signal, modality='ecg', ecg_method='christov', backend='bokeh', show_heart_rate=True)
)
```

#### Engelse-Zeelenberg

```{code-cell} ipython3
%%timeit
peaks_ew = engelse_zeelenberg(signal, sfreq=1000)
```

```{code-cell} ipython3
show(
    plot_raw(signal, modality='ecg', ecg_method='engelse-zeelenberg', backend='bokeh', show_heart_rate=True)
)
```

## Photoplethysmography

+++

### Systolic peaks detection

+++

#### Online

+++

Let's first simulate some PPG recording. Here, we will use the `serialSim` class from Systole in conjunction with the `Oximeter` recording class. This will *simulate* the presence of a pulse oximeter on the computer and provide the information from a pre-recorded signal in real time. Simulating synthetic cardiac data in this way can be a great way to test your data analysis pipelines before collecting data - avoiding painful mistakes that can occur!

```{code-cell} ipython3
# Pulse oximeter simulation
ser = serialSim()

# Create an Oxymeter instance, initialize recording and record for 10 seconds
oxi = Oximeter(serial=ser, sfreq=75).setup()
oxi.read(20);
```

```{code-cell} ipython3
# Retrieve the data from the oxi class
times = np.array(oxi.times)
threshold = np.array(oxi.threshold)
recording = np.array(oxi.recording)
peaks = np.array(oxi.peaks)
```

This method uses the derivative to find peaks in the signal and select them based on and adaptive threshold based on the rolling mean and rolling standard deviation in a given time window.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(15, 5))
ax.set_title("Oximeter recording")

ax.plot(times, threshold, linestyle="--", color="gray", label="Threshold", linewidth=1.5)
ax.fill_between(
    x=times, y1=threshold, y2=recording.min(), alpha=0.2, color="gray"
)
ax.plot(times, recording, label="Recording", color="#4c72b0", linewidth=1.5)
ax.fill_between(x=times, y1=recording, y2=recording.min(), color="w")
ax.scatter(x=times[np.where(peaks)[0]], y=recording[np.where(peaks)[0]],
           color="#c44e52", alpha=.6, label="Online estimation",
           edgecolors='k')
ax.set_ylabel("PPG level")
ax.set_xlabel("Time (s)")
ax.legend()
sns.despine()
```

#### Offline

+++

A simple online approach like the one we described is usually good enough to detect all the systolic peaks, provided that the subject is not moving too much.

```{code-cell} ipython3
ppg = import_ppg()
```

**Clipping artefacts**

+++

Clipping is a form of distortion that can limit signal when it exceeds a certain threshold [see the Wikipedia page](https://en.wikipedia.org/wiki/Clipping_(signal_processing)). Some PPG devices can produce clipping artefacts when recording the PPG signal, for example if a participant is pressing too hard on the fingerclip. Here, we can see that some clipping artefacts are found between 100 and 150 seconds in the previous recording. The threshold values (here `255`), is often set by the device and can easily be found depending on the manufacturer. These artefacts should be corrected before systolic peaks detection. One way to correct these artefacts is to remove the portion of the signal where clipping artefacts occurs and use cubic spline interpolation to reconstruct a plausible estimate of the *real* underlying signal. This is what the function `interpolate_clipping()` does {cite:p}`2019:vanGent`.

```{code-cell} ipython3
signal = ppg[ppg.time.between(110, 113)].ppg.to_numpy()  # Extract a portion of signal with clipping artefacts
clean_signal = interpolate_clipping(signal, min_threshold=0, max_threshold=255)  # Remove clipping segment and interpolate missing calues
```

```{code-cell} ipython3
plt.figure(figsize=(15, 5))
plt.plot(np.arange(0, len(clean_signal))/75, clean_signal, label='Corrected PPG signal', linestyle= '--', color='#c44e52')
plt.plot(np.arange(0, len(signal))/75, signal, label='Raw PPG signal', color='#4c72b0')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('PPG level (a.u)')
sns.despine()
plt.tight_layout()
```

This concludes the tutorial on using Systole to detect individual heartbeats and to reject common ECG and PPG artefacts. The next tutorial will go further into more nuanced issues in artefact detection and correction.

```{code-cell} ipython3

```
