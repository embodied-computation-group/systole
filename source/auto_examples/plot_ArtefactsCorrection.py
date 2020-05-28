"""
Outliers and ectobeats correction
=================================

This example shows how to detect extra ectobeats, missed and ectobeats from RR
time series using the method proposed by Lipponen & Tarvainen (2019) [#]_.
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
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# rr = rr_artifacts()
#
# # Method 1
# ##########

#%%
# References
# ----------
# .. [#] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for
#   heart rate variability time series artefact correction using novel
#   beat classification. Journal of Medical Engineering & Technology,
#   43(3), 173–181. https://doi.org/10.1080/03091902.2019.1640306
