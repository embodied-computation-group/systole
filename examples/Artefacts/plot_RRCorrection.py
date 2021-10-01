"""
Detecting and correcting artefacts in RR time series
====================================================

This example describes artefacts correction in RR time series.

The function `correct_rr()` automatically detect artefacts using the method proposed
by Lipponen & Tarvainen (2019) [#]_. At each iteration, shorts, extra, long, missed 
and ectopic beats are corrected using interpolation of the RR time series, and the
detection procedure is run again using cleaned intervals. Importantly, when using 
this method the signal length can be altered after the interpolation, introducing 
misalignement with eg. triggers from the experiment. For this reason, it is only 
recommended to use it in the context of "bloc design" study or heart rate variability.

"""

# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
# Licence: GPL v3

#%%
import numpy as np
import pandas as pd
from systole import import_dataset1
from systole.detection import ecg_peaks
from systole.correction import correct_rr
from systole.utils import input_conversion
from systole.plots import plot_rr, plot_frequency
from systole.hrv import frequency_domain
import matplotlib.pyplot as plt
import seaborn as sns

#%% Import ECG recording and events triggers
ecg_df = import_dataset1(modalities=['ECG', 'Stim'])

#%% Detecting R peaks in the ECG signal using the Pan-Tompkins method
signal, peaks = ecg_peaks(ecg_df.ecg, method='pan-tompkins', sfreq=1000)

#%% First, we are going to convert the peaks vector into RR intervals time series.
rr_ms = input_conversion(peaks, input_type="peaks", output_type="rr_ms")

#%% We can visualize this series using Systole's built in `plot_rr` function. Here we
# are using Matplotlib as plotting backend.
plot_rr(rr_ms, input_type='rr_ms', figsize=(13, 5))

#%% For now, this time series it not really artefacted. We can easily simulate missed
# peaks and extra peaks by manually increasing or decreasing the length of some RR 
# intervals.

np.random.seed(123)  # For result reproductibility

corrupted_rr = rr_ms.copy()  # Create a new RR intervals vector

# Randomly select 50 intervals in the time series and multiply them by 2 (missed peaks)
corrupted_rr[np.random.choice(len(corrupted_rr), 50)] *= 2

# Randomly select 50 intervals in the time series and divide them by 3 (extra peaks)
corrupted_rr[np.random.choice(len(corrupted_rr), 50)] /= 3

#%% Lets see if the artefact we created are correctly detected. Note that here, we are
# using `show_artefacts=True` so the artefacts detection runs automatically and shows
# in the plot.
plot_rr(
    corrupted_rr, input_type='rr_ms', show_artefacts=True, line=False, figsize=(13, 5)
    )
plt.show()
#%% The artefacts simulation seems to work fine so far. We have created abnormal long 
# and short RR intervals and they are later correctly detected. We can now apply the RR
# time series correction method. This function will automatically detect possible
# artefacts in the RR intervals and reconstruct the most probable value using time
# series interpolation. The number of iteration is set to `2` by default, we add it 
# here for clarity.
rr_correction = correct_rr(corrupted_rr, n_iterations=2)

#%% The num
plot_rr(rr_correction["clean_rr"], input_type='rr_ms', show_artefacts=True,
        line=False, figsize=(13, 5))
plt.show()
#%% We can see that after two iterations, most/all of the artefacts have been corrected.
# This does not means that the new values match exactly the RR intervals, and the new 
# corrected time series will always slightly differs from the original one. However, we
# can estimate how large this difference is by comparing the true, corrupted and 
# corrected time series a posteriori. Here, instead of comparing the time series side 
# by side, we can have a look at some HRV metrics that are known to be affected by RR
# artefacts, like the high frequency HRV.
_, axs = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
for i, rr, lab in zip(range(3), 
                 [rr_ms, corrupted_rr, rr_correction["clean_rr"]],
                 ["Original", "Corrupted", "Corrected"]):
    plot_frequency(rr, input_type="rr_ms", ax=axs[i])
    axs[i].set_title(lab)


#%% Looking at the PSD plots, it seems that the corrected RR time series has highly 
# similar frequency dynamic. To get a more quantitative view of this resultm e can
# simply repeat this process of RR corruption-correction many time and check the HF-HRV
# parameters estimated at each steps.

# Clean the RR time series before simulation
initial_rr = correct_rr(rr_ms.copy())["clean_rr"]

simulation_df = pd.DataFrame([])
for i in range(20):
    
    # Measure HF-HRV for corrupted RR intervals time series
    corrupted_rr = initial_rr.copy()
    corrupted_rr[np.random.choice(len(corrupted_rr), 50)] *= 2
    corrupted_rr[np.random.choice(len(corrupted_rr), 50)] /= 3
    corrupted_hrv = frequency_domain(corrupted_rr, input_type="rr_ms")
    corrupted_hf = corrupted_hrv[corrupted_hrv.Metric == "power_hf_nu"].Values.iloc[0]
    
    # Measure HF-HRV for corrected RR intervals time series
    corrected = correct_rr(corrupted_rr, n_iterations=2, verbose=False)["clean_rr"]
    corrected_hrv = frequency_domain(corrected, input_type="rr_ms")
    corrected_hf = corrected_hrv[corrected_hrv.Metric == "power_hf_nu"].Values.iloc[0]

    simulation_df = simulation_df.append(
        pd.DataFrame({"HF-HRV (n.u.)": [corrupted_hf, corrected_hf],
                      "Data Quality": ["Corrupted", "Corrected"]
                      })
        )

initial_hrv = frequency_domain(initial_rr, input_type="rr_ms")
initial_hf = initial_hrv[initial_hrv.Metric == "power_hf_nu"].Values.iloc[0]

#%% Simulation results
plt.figure(figsize=(5, 8))
sns.boxplot(data=simulation_df, x="Data Quality", y="HF-HRV (n.u.)", palette="vlag")
sns.stripplot(data=simulation_df, x="Data Quality", y="HF-HRV (n.u.)",
              size=8, color=".3", linewidth=0)
plt.axhline(y=initial_hf, linestyle="--", color="gray")
plt.title("HF-HV recovery \n after RR artefacts correction")
plt.annotate("True HF-HRV", xy=(0, initial_hf), xytext=(-0.4, initial_hf - 0.05),
             arrowprops = dict(facecolor ='grey', shrink = 0.05))
sns.despine()
plt.show()

#%% As we can see in the figure above, while the estimate of the HF-HRV is largely 
# affected by the presence of simulated artefacts, the proposed correction methods
# allows to recover the true parameter with some precision.

#%%
# References
# ----------
# .. [#] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for
#   heart rate variability time series artefact correction using novel
#   beat classification. Journal of Medical Engineering & Technology,
#   43(3), 173–181. https://doi.org/10.1080/03091902.2019.1640306
