"""
Recording PPG signal
====================
"""

# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
# Licence: GPL v3

# The py:class:systole.recording.Oximeter class can be used to read incoming
# PPG signal from `Nonin 3012LP Xpod USB pulse oximeter
# <https://www.nonin.com/products/xpod/>`_ together with the `Nonin 8000SM
# 'soft-clip' fingertip sensors <https://www.nonin.com/products/8000s/>`_.
# This function can easily be integrated with other stimulus presentation
# software lie `PsychoPy <https://www.psychopy.org/>`_ to record cardiac
# activity during psychological experiments, or to synchronize stimulus
# delivery with cardiac phases (e.g. systole or diastole).


#%%
# Reading
# -------
# Recording and plotting your first time-series will only require 5 lines
# of code:

import time

import serial

from systole.recording import Oximeter

ser = serial.Serial("COM4")  # Add your USB port here

# Open serial port, initialize and plot recording for Oximeter
oxi = Oximeter(serial=ser).setup().read(duration=10)

# The signal can be directly plotted using built-in functions.
oxi.plot_oximeter()

##############################################################################
# .. figure::  https://github.com/embodied-computation-group/systole/raw/master/Images/recording.png
#    :align:   center
##############################################################################

#%%
# Interfacing with PsychoPy
# -------------------------

# * The ``read()`` method will record for a predefined amount of time
# (specified by the ``duration`` parameter, in seconds). This 'serial mode'
# is the easiest and most robust method, but it does not allow the execution
# of other instructions in the meantime.

# Code 1 {}
oximeter.read(duration=10)
# Code 2 {}

# * The ``readInWaiting()`` method will only read the bytes temporally stored
# in the USB buffer. For the Nonin device, this represents up to 10 seconds of
# recording (this procedure should be executed at least one time every 10
# seconds for a continuous recording). When inserted into a while loop, it can
# record PPG signal in parallel with other commands.


tstart = time.time()
while time.time() - tstart < 10:
    oximeter.readInWaiting()
    # Insert code here {...}

#%%
# Online detection
# ----------------
# Online heart beat detection, for cardiac-stimulus synchrony


# Open serial port
ser = serial.Serial("COM4")  # Change this value according to your setup

# Create an Oxymeter instance and initialize recording
oxi = Oximeter(serial=ser, sfreq=75, add_channels=4).setup()

# Online peak detection for 10 seconds
tstart = time.time()
while time.time() - tstart < 10:
    while oxi.serial.inWaiting() >= 5:
        paquet = list(oxi.serial.read(5))
        oxi.add_paquet(paquet[2])  # Add new data point
        if oxi.peaks[-1] == 1:
            print("Heartbeat detected")
