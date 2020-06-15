"""
Heartbeat Evoked Arpeggios
============================

This tutorial illustrates how to use the ``Oximeter`` class to trigger stimuli
at different phases of the cardiac cycle using the [Psychopy](https://www.psychopy.org/)
toolbox. The PPG signal is recorded for 30 seconds and peaks are detected
online. Four notes ('C', 'E', 'G', 'Bfl') are played in synch with peak
detection with various delays: no delay,  1/4, 2/4 or 3/4 of the previous
cardiac cycle length. While R-R intervals are prone to large changes over longer
timescales, such changes are physiologically limited from one heartbeat to the next,
limiting variance in the onset synchrony between the tones and the cardiac cycle.
On this basis, each presentation time is calibrated based on the previous RR-interval.
This procedure can easily be adapted to create a standard interoception task, e.g. by either presenting
tones at no delay (systole, s+) or at a fixed offset (diastole, s-).
"""

# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
# Licence: GPL v3

import time
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from psychopy.sound import Sound

from systole.utils import norm_triggers
from systole import serialSim
from systole.utils import to_angles
from systole.plotting import circular
from systole.recording import Oximeter

#%%
# Recording
# ---------
# For the purpose of demonstration, here we simulate data acquisition through
# the pulse oximeter using pre-recorded signal.

ser = serialSim()

#%%
# If you want to allow online data acquisition, you should uncomment the
# following lines and provide the reference of the COM port where the pulse
# oximeter is plugged in.

###############################################################################
# .. code-block:: python
#
#   import serial
#   ser = serial.Serial('COM4')  # Change this value according to your setup

#%%
# Create an Oximeter instance, initialize recording and record for 10 seconds

oxi = Oximeter(serial=ser, sfreq=75, add_channels=4).setup()

#%%
# Create an Oxymeter instance, initialize recording and record for 10 seconds

beat = Sound('C', secs=0.1)
diastole1 = Sound('E', secs=0.1)
diastole2 = Sound('G', secs=0.1)
diastole3 = Sound('Bfl', secs=0.1)

systoleTime1, systoleTime2, systoleTime3 = None, None, None
tstart = time.time()
while time.time() - tstart < 30:

    # Check if there are new data to read
    while oxi.serial.inWaiting() >= 5:

        # Convert bytes into list of int
        paquet = list(oxi.serial.read(5))

        if oxi.check(paquet):  # Data consistency
            oxi.add_paquet(paquet[2])  # Add new data point

        # T + 0
        if oxi.peaks[-1] == 1:
            beat = Sound('C', secs=0.1)
            beat.play()
            systoleTime1 = time.time()
            systoleTime2 = time.time()
            systoleTime3 = time.time()

        # T + 1/4
        if systoleTime1 is not None:
            if time.time() - systoleTime1 >= ((oxi.instant_rr[-1]/4)/1000):
                diastole1 = Sound('E', secs=0.1)
                diastole1.play()
                systoleTime1 = None

        # T + 2/4
        if systoleTime2 is not None:
            if time.time() - systoleTime2 >= (
                                    ((oxi.instant_rr[-1]/4) * 2)/1000):
                diastole2 = Sound('G', secs=0.1)
                diastole2.play()
                systoleTime2 = None

        # T + 3/4
        if systoleTime3 is not None:
            if time.time() - systoleTime3 >= (
                                    ((oxi.instant_rr[-1]/4) * 3)/1000):
                diastole3 = Sound('A', secs=0.1)
                diastole3.play()
                systoleTime3 = None

        # Track the note status
        oxi.channels['Channel_0'][-1] = beat.status
        oxi.channels['Channel_1'][-1] = diastole1.status
        oxi.channels['Channel_2'][-1] = diastole2.status
        oxi.channels['Channel_3'][-1] = diastole3.status

#%%
# Events
# --------

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
oxi.plot_recording(ax=ax1)
oxi.plot_events(ax=ax2)
plt.tight_layout()

#%%
# Cardiac cycle
# -------------

angles = []
x = np.asarray(oxi.peaks)
for ev in oxi.channels:
    events = norm_triggers(np.asarray(oxi.channels[ev]), threshold=1, n=40,
                           direction='higher')
    angles.append(to_angles(np.where(x)[0], np.where(events)[0]))

palette = itertools.cycle(sns.color_palette('deep'))
ax = plt.subplot(111, polar=True)
for i in angles:
    circular(i, color=next(palette), ax=ax)
