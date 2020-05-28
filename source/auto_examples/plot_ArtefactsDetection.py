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
import plotly
from systole.detection import rr_artefacts
from systole.utils import simulate_rr
from systole.plotly import plot_subspaces

#%%
# Simulate RR time series
# -----------------------
# This function will simulate RR time series containing ectopic, extra, missed,
# long and short artefacts.

rr = simulate_rr()

#%%
# Artefact detection
# ------------------

artefacts = rr_artefacts(rr)

#%%
# Time series visualization
# -------------------------
# You can visualize the RR time series. Providing and `artefact` dictionary
# will also highlight the detected artefactsa and outliers. Here, we can see
# that all the simulated artefacts were correctly labelled by the algorythm.

fig = plot_subspaces(rr)
plotly.io.show(fig)

#%%
# Subspaces visualization
# -----------------------
# You can visualize the two main subspaces and spot outliers. The left pamel
# plot subspaces that are more sensitive to ectopic beats detection. The right
# panel plot subspaces that will be more sensitive to long or short beats,
# comprizing the extra and missed beats.

fig = plot_subspaces(rr)
plotly.io.show(fig)

#%%
# References
# ----------
# .. [#] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for
#   heart rate variability time series artefact correction using novel
#   beat classification. Journal of Medical Engineering & Technology,
#   43(3), 173–181. https://doi.org/10.1080/03091902.2019.1640306
