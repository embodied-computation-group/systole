"""
Outliers and ectobeats correction
=================================

Here, we describe two method for artefacts and outliers correction, after
detection using the method proposed by Lipponen & Tarvainen (2019) [#]_.
"""

# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
# Licence: GPL v3

# Two approaches for artefacts correction are proposed:
# * `correct_peaks()` will find and correct artefacts in a boolean peaks
# vector, thus ensuring the length of recording remain constant and corrected
# peaks fit the signal sampling rate. This method is more adapted to
# event-related cardiac activity designs.

# * `correct_rr()` will find and correct artefacts in the RR time series. The
# signal length will possibly change after the interpolation of long, short or
# ectopic beats. This method is more relevant for HRV analyse of long recording
# where the timing of experimental events is not important.

#%%
import numpy as np
import matplotlib.pyplot as plt
from systole import simulate_rr
from systole.plotting import plot_subspaces
from systole.correction import correct_peaks, correct_rr


#%% Method 1 - Peaks correction
# #############################

peaks = simulate_rr(as_peaks=True)
peaks_correction = correct_peaks(peaks)
peaks_correction

#%% Method 2 - RR correction
# #############################
rr = simulate_rr()
rr_correction = correct_rr(rr)

#%%
# References
# ----------
# .. [#] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for
#   heart rate variability time series artefact correction using novel
#   beat classification. Journal of Medical Engineering & Technology,
#   43(3), 173–181. https://doi.org/10.1080/03091902.2019.1640306
