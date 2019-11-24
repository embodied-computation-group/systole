# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import serial
import time
from ecgrecording import Oximeter
from ecgcircular import to_angles, circular, plot_circular
from psychopy.sound import Sound
import matplotlib.pyplot as plt
import numpy as np

ser = serial.Serial('COM4')

# Open seral port for Oximeter
oxi = Oximeter(serial=ser, sfreq=75, add_channels=4).setup()

systole = Sound('C', secs=0.1)
diastole1 = Sound('E', secs=0.1)
diastole2 = Sound('G', secs=0.1)
diastole3 = Sound('A', secs=0.1)


systoleTime1, systoleTime2, systoleTime3 = None, None, None
tstart = time.time()
while time.time() - tstart < 10:

    # Check if there are new data to read
    while oxi.serial.inWaiting() >= 5:

        # Convert bytes into list of int
        paquet = list(oxi.serial.read(5))
        if oxi.check(paquet):  # Data consistency
            oxi.add_paquet(paquet[2])  # Add new data point

        # T + 0
        if oxi.peaks[-1] == 1:
            systole = Sound('C', secs=0.1)
            systole.play()
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
            if time.time() - systoleTime2 >= (((oxi.instant_rr[-1]/4) * 2)/1000):
                diastole2 = Sound('G', secs=0.1)
                diastole2.play()
                systoleTime2 = None

        # T + 3/4
        if systoleTime3 is not None:
            if time.time() - systoleTime3 >= (((oxi.instant_rr[-1]/4) * 3)/1000):
                diastole3 = Sound('A', secs=0.1)
                diastole3.play()
                systoleTime3 = None

        # Track the note status
        oxi.channels['Channel_0'][-1] = systole.status
        oxi.channels['Channel_1'][-1] = diastole1.status
        oxi.channels['Channel_2'][-1] = diastole2.status
        oxi.channels['Channel_3'][-1] = diastole3.status

# Plot recording
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
oxi.plot_recording(ax=ax1)
oxi.plot_events(ax=ax2)
oxi.plot_hr(ax=ax3)
plt.tight_layout()

#
angles = []
x = np.asarray(oxi.peaks)
for ev in oxi.channels:
    events = np.asarray(oxi.channels[ev])
    for i in range(len(events)):
        if events[i] == 1:
            events[i+1:i+10] = 0
    angles.append(to_angles(x, events))


circular(angles[0], color='gray')
circular(angles[1], color='r')
circular(angles[2], color='g')
circular(angles[3], color='b')
plt.savefig('circular.png')
