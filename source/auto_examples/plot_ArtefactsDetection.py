"""
Outliers and artefacts detection
================================

This example shows how to detect ectopic, missed, extra, slow and long long
from RR or pulse rate interval time series using the method proposed by
Lipponen & Tarvainen (2019) [#]_.
"""

# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
# Licence: GPL v3

#%%
from systole.detection import rr_artefacts
from systole.plots import plot_subspaces
from systole.utils import simulate_rr

#%%
# RR artefacts
# ------------
# The proposed method will detect 4 kinds of artefacts in an RR time series:
# Missed R peaks, when an existing R component was erroneously NOT detected by
# the algorithm.
# * Extra R peaks, when an R peak was detected but does not exist in the
# signal.
# * Long or short interval intervals, when R peaks are correctly detected but
# the resulting interval has extreme value in the overall time-series.
# * Ectopic beats, due to disturbance of the cardiac rhythm when the heart
# either skip or add an extra beat.
# * The category in which the artefact belongs will have an influence on the
# correction procedure (see Artefact correction).

#%%
# Simulate RR time series
# -----------------------
# This function will simulate RR time series containing ectopic, extra, missed,
# long and short artefacts.

rr = simulate_rr()

#%%
# Artefact detection
# ------------------

outliers = rr_artefacts(rr)

#%%
# Subspaces visualization
# -----------------------
# You can visualize the two main subspaces and spot outliers. The left pamel
# plot subspaces that are more sensitive to ectopic beats detection. The right
# panel plot subspaces that will be more sensitive to long or short beats,
# comprizing the extra and missed beats.

plot_subspaces(rr)

#%%
# References
# ----------
# .. [#] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for
#   heart rate variability time series artefact correction using novel
#   beat classification. Journal of Medical Engineering & Technology,
#   43(3), 173–181. https://doi.org/10.1080/03091902.2019.1640306
