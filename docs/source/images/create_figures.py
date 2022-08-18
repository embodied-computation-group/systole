
"""Create figures presented in the README.
"""

from systole import import_dataset1
from systole.detection import ecg_peaks
from systole.plots import plot_frequency, plot_poincare, plot_raw, plot_subspaces
from bokeh.io import export_png, save
from bokeh.layouts import row
# Import ECg recording
signal = import_dataset1(modalities=['ECG']).ecg.to_numpy()

# R peaks detection
signal, peaks = ecg_peaks(signal, method='pan-tompkins', sfreq=1000)

##########
# Raw plot
##########

#%% As PNG
export_png(
    plot_raw(signal[60000 : 120000], modality="ecg", backend="bokeh", 
             show_heart_rate=True, show_artefacts=True, figsize=300),
    filename="raw.png", width=1400
)

#%% As HTML
save(
    plot_raw(signal[60000 : 120000], modality="ecg", backend="bokeh", 
             show_heart_rate=True, show_artefacts=True, figsize=300),
    filename="raw.html", 
)

########################
# Heart Rate Variability
########################

#%% As PNG
export_png(
    row(
        plot_frequency(peaks, input_type="peaks", backend="bokeh", figsize=(600, 400)),
        plot_poincare(peaks, input_type="peaks", backend="bokeh", figsize=(400, 400)),
    ),filename="hrv.png"
)

#%% As HTML
save(
    row(
        plot_frequency(peaks, input_type="peaks", backend="bokeh", figsize=(300, 200)),
        plot_poincare(peaks, input_type="peaks", backend="bokeh", figsize=(200, 200)),
    ),filename="hrv.html"
)

#####################
# Artefacts detection
#####################

#%% As PNG
export_png(
    plot_subspaces(peaks, input_type="peaks", backend="bokeh"),
    filename="subspaces.png"
)

#%% As HTML
save(
    plot_subspaces(peaks, input_type="peaks", backend="bokeh"),
    filename="subspaces.html"
)