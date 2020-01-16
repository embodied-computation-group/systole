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

from systole import serialSim
from systole.utils import heart_rate
from systole.recording import Oximeter
import matplotlib.pyplot as plt
import numpy as np

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
oxi.plot_recording(ax=ax[0])

ax[1].plot(oxi.times, oxi.peaks, 'k')
ax[1].set_title('Peaks vector', fontweight='bold')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Peak\n detection')


hr, time = heart_rate(oxi.peaks, sfreq=75, unit='rr', kind='cubic')
ax[2].plot(time, hr, label='Interpolated HR', linestyle='--', color='gray')
ax[2].plot(np.array(oxi.times)[np.where(oxi.peaks)[0]],
           hr[np.where(oxi.peaks)[0]], 'ro', label='Instantaneous HR')
ax[2].set_xlabel('Time (s)')
ax[2].set_title('Instantaneous Heart Rate', fontweight='bold')
ax[2].set_ylabel('RR intervals (ms)')

plt.tight_layout()
