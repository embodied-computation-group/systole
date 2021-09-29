from systole import import_dataset1
from systole.detection import ecg_peaks
from systole.plots import plot_frequency, plot_pointcare
from bokeh.io import export_png, save
from bokeh.layouts import row

# Import ECg recording
signal = import_dataset1(modalities=['ECG']).ecg.to_numpy()

# R peaks detection
signal, peaks = ecg_peaks(signal, method='pan-tompkins', sfreq=1000)

#%% As PNG
export_png(
    row(
        plot_frequency(peaks, input_type="peaks", backend="bokeh", figsize=(600, 400)),
        plot_pointcare(peaks, input_type="peaks", backend="bokeh", figsize=(400, 400)),
    ),filename="hrv.png"
)

#%% As HTML
save(
    row(
        plot_frequency(peaks, input_type="peaks", backend="bokeh", figsize=(600, 400)),
        plot_pointcare(peaks, input_type="peaks", backend="bokeh", figsize=(400, 400)),
    ),filename="hrv.html"
)