"""

Plot raw physiological signal
=============================

"""

# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
# Licence: GPL v3


from bokeh.io import output_notebook
from bokeh.plotting import show
from systole.plots import plot_raw

from systole import import_dataset1, import_ppg

#%%
# Plot raw ECG signal
# --------------------

# Import PPG recording as pandas data frame
physio_df = import_dataset1(modalities=['ECG', 'Respiration'])

# Only use the first 60 seconds for demonstration
ecg = physio_df[physio_df.time.between(60, 90)].ecg
plot_raw(ecg, modality='ecg', sfreq=1000, ecg_method='sleepecg')
#%%
# Plot raw PPG signal
# -------------------
# Import Respiration recording as pandas data frame
rsp = import_dataset1(modalities=['Respiration'])

# Only use the first 90 seconds for demonstration
rsp = physio_df[physio_df.time.between(500, 600)].respiration
plot_raw(rsp, sfreq=1000, modality="respiration")
#%%
# Plot raw respiratory signal
# ---------------------------

# Import PPG recording as pandas data frame
ppg = import_ppg()

# Only use the first 60 seconds for demonstration
plot_raw(ppg[ppg.time<60], sfreq=75)

#%%
# Using Bokeh as plotting backend
# -------------------------------

output_notebook()

show(
    plot_raw(ppg, backend="bokeh", show_heart_rate=True, show_artefacts=True)
 )