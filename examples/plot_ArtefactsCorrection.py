"""
Outliers and ectobeats correction
=================================

Here, we describe two method for artefacts and outliers correction, after
detection using the method proposed by Lipponen & Tarvainen (2019) [1]_.
"""

# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
# Licence: GPL v3

# This example describe two approaches for RR artefacts correction:
# * `correct_rr()` will find and correct artefacts in the RR time series. The
# signal length will possibly change after the interpolation of long, short or
# ectopic beats. This method is more relevant for HRV analyse of long recording
# where the timing of experimental events is not important.
# * `correct_peaks()` will find and correct artefacts in a boolean peaks
# vector, thus ensuring the length of recording remain constant and corrected
# peaks fit the signal sampling rate. This method is more adapted to
# event-related cardiac activity designs.

#%%
import numpy as np
from systole import import_dataset1
from systole.detection import ecg_peaks
from systole.correction import correct_peaks, correct_rr
from systole.utils import input_conversion
from systole.plots import plot_rr
from bokeh.io import output_notebook
from bokeh.plotting import show
output_notebook()


#%% Import ECG recording and Stim channel
ecg_df = import_dataset1(modalities=['ECG', 'Stim'])

#%% Peak detection in the ECG signal using the Pan-Tompkins method
signal, peaks = ecg_peaks(ecg_df.ecg, method='pan-tompkins', sfreq=1000)

#%% Method 1 - RR correction
# #############################
# Let's first convert the R peaks previously detected to RR intervals.
# 
# import an example time series of RR interval. Here, `rr` is simplys a 1d Numpy array of intervals expressed in miliseconds (ms).
rr_ms = input_conversion(peaks, input_type="peaks", output_type="rr_ms")

#%%
# We can visualize this series using Systole's built in `plot_rr` function. Here we are using Bokeh as plotting backend.

show(
    plot_rr(rr_ms, input_type='rr_ms', backend='bokeh', figsize=300)
)
#%% RR correction
# #############################
rr_correction = correct_rr(rr_ms)

#%% Method 2 - Peaks correction
# #############################

#%% Create aretefacts in the peaks vector
# Here, we are going to add false extra and missed peaks (by adding and removing ones in the peaks vector).
corrupted_peaks = peaks

np.random.seed(123)
# Create 100 missed peaks
corrupted_peaks[
    np.random.choice(np.where(corrupted_peaks)[0], 50)
    ] = 0

# Create 100 extra peaks
corrupted_peaks[
    np.random.choice(len(peaks), 50)
    ] = 1

#%%
# We can visualize the corrupted vector (this time setting `show_artefacts` to `True`).
show(
    plot_rr(
        corrupted_peaks, input_type='peaks', backend='bokeh', figsize=300,
        show_artefacts=True)
)

#%%
# We can see that this procedure has successfully created 50 extra and 50 missed peaks
# in the signal. We can see that some of the extra peaks are even falling in the red
# area in the plot above. This area cover all values below 200ms and above 3000ms and 
# are considered as physiologically unlikely/impossible RR intervals for humans that
# should be discared automatically most of the time. We will try to correct that using 
# Systole's `correct_peaks` function. Contrarily to `correct_rr`, this function operate 
# at the peaks detection level and will keep the lenght of the signal constant. This
# can be especially usefull when we want to look at the evolution of RR interval in
# response to specific events.

peaks_correction = correct_peaks(corrupted_peaks)

#%%
# We can see that the function is doing a good job at detection and correcting peaks.
# 47/50 extra and 44/50 missed were corrected here.
show(
    plot_rr(
        peaks_correction["clean_peaks"], input_type='peaks', backend='bokeh', 
        figsize=300, show_artefacts=True)
)

#%%
# References
# ----------
# .. [1] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for
#   heart rate variability time series artefact correction using novel
#   beat classification. Journal of Medical Engineering & Technology,
#   43(3), 173–181. https://doi.org/10.1080/03091902.2019.1640306
