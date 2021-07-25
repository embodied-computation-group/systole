"""
Instantaneous Heart Rate
========================

This example show how to record PPG signals using the `Nonin 3012LP
Xpod USB pulse oximeter <https://www.nonin.com/products/xpod/>`_ and the `Nonin
8000SM 'soft-clip' fingertip sensors <https://www.nonin.com/products/8000s/>`_.
Peaks are automatically labelled online and the instantaneous heart rate is
plotted.
"""

# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
# Licence: GPL v3

import matplotlib.pyplot as plt
import pandas as pd
from systole import serialSim
from systole.recording import Oximeter
from systole.plots import plot_raw, plot_rr


#%%
# Recording
# ---------
# For the demonstration purpose, here we simulate data acquisition through
# the pulse oximeter using pre-recorded signal.

ser = serialSim()

#%%
# If you want to enable online data acquisition, you should uncomment the
# following lines and provide the reference of the COM port where the pulse
# oximeter is plugged in.

###############################################################################
# .. code-block:: python
#
#   import serial
#   ser = serial.Serial('COM4')  # Change this value according to your setup

# Create an Oxymeter instance, initialize recording and record for 10 seconds
oxi = Oximeter(serial=ser, sfreq=75).setup()
oxi.read(30)

#%%
# Plotting
# --------
fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True)

plot_raw(oxi.recording, sfreq=75, show_heart_rate=False, ax=ax[0])

times = pd.to_datetime(oxi.times, unit="s", origin="unix")

ax[1].plot(times, oxi.peaks, "#55a868")
ax[1].set_title("Peaks vector")
ax[1].set_ylabel("Peak\n detection")

plot_rr(oxi.peaks, input_type='peaks', ax=ax[2])

plt.tight_layout()