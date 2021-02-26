"""
Estimation of respiratory signal from PPG
=========================================

This script shows how respration can be estimated from PPG signal.
"""

# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
# Licence: GPL v3

#%%
import plotly

from systole.detection import rr_artefacts

#%%
# Subspaces visualization
# -----------------------
# You can visualize the two main subspaces and spot outliers. The left pamel
# plot subspaces that are more sensitive to ectopic beats detection. The right
# panel plot subspaces that will be more sensitive to long or short beats,
# comprizing the extra and missed beats.

plot_subspaces(rr)
