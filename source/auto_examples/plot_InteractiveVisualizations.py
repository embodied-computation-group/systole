"""
Interactive visualization
=========================

The pipeline of physiological recording can often benefiit from
interactive data visualization and exploration to guide data analysis. Systole
include functions  build on the top of Plotly (https://plotly.com/) for
interactive data visualization and dashboard integration
(https://plotly.com/dash/).
"""

# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
# Licence: GPL v3

#%%
import plotly
from systole.detection import rr_artefacts
from systole import import_rr, import_ppg
from systole.plotly import plot_subspaces, plot_frequency, plot_raw,\
    plot_timedomain, plot_nonlinear

#%%
# Raw data
# --------
#
ppg = import_ppg()[0]
plot_raw(ppg)

#%%
# HRV analyses
# ------------

rr = import_rr().rr.values

#%%
# Frequency domain
# ----------------
plot1 = plot_timedomain(rr)
plotly.io.show(plot1)

#%%
# Frequency domain
# ----------------
plot2 = plot_frequency(rr)
plotly.io.show(plot2)

#%%
# Nonlinear domain
# ----------------
plot3 = plot_nonlinear(rr)
plotly.io.show(plot3)

#%%
# Artefact detection
# ------------------

artefacts = rr_artefacts(rr)

#%%
# Subspaces visualization
# -----------------------
# You can visualize the two main subspaces and spot outliers. The left pamel
# plot subspaces that are more sensitive to ectopic beats detection. The right
# panel plot subspaces that will be more sensitive to long or short beats,
# comprizing the extra and missed beats.

plot_subspaces(rr)
