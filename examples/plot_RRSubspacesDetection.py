"""
Outliers and ectobeats detection
================================

This example shows how to detect extra ectobeats, missed and ectobeats from RR
time series using the method proposed by Lipponen & Tarvainen (2019) [#]_.
"""

# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
# Licence: GPL v3

#%%
from systole.detection import rr_outliers
from systole.plotting import plot_subspaces, plot_hr
from systole import import_rr

#%%
# Simulate RR time series
# -----------------------

rr = import_rr().rr[:100]

#%%
# Add artefacts
# -------------

# Add missed beat
rr[20] = 1600

# Add extra beat
rr[40] = 400

# Add ectobeat (type 1)
rr[60] = 1100
rr[61] = 500

# Add ectobeat (type 2)
rr[80] = 500
rr[81] = 1100

#%%
# Artefact detection
# ------------------
# You can visualize the two main subspaces and spot outliers.
# Here we can see that two intervals have been labelled as probable ectobeats
# (left pannel), and a total of 6 datapoints are considered as outliers, being
# too long or too short (right pannel).

plot_subspaces(rr)

#%%
# Plotting
# --------
# We can then plot back the labelled outliers in the RR interval time course

ectobeats, outliers = rr_outliers(rr)
plot_hr(rr.values, kind='linear', outliers=(ectobeats | outliers))

#%%
# References
# ----------
# .. [#] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for
#   heart rate variability time series artefact correction using novel
#   beat classification. Journal of Medical Engineering & Technology,
#   43(3), 173–181. https://doi.org/10.1080/03091902.2019.1640306
