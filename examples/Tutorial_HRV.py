"""
Recording
=========


"""

# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
# Licence: GPL v3

# It can easily interface with `PsychoPy <https://www.psychopy.org/>`_ to
# record PPG signal during psychological experiments, and to synchronize
# stimulus deliver to e.g., systole or diastole.

# For example, you can record and plot data in less than 6 lines of code:


#%%
# Event related cardiac deceleration
# ----------------------------------
import serial
from systole.recording import Oximeter
ser = serial.Serial('COM4')  # Add your USB port here

# Open serial port, initialize and plot recording for Oximeter
oxi = Oximeter(serial=ser).setup().read(duration=10)


Interfacing with PsychoPy
-------------------------

The ``Oximeter`` class can be used together with a stimulus presentation software to record cardiac activity during psychological experiments.

* The ``read()`` method

will record for a predefined amount of time (specified by the ``duration`` parameter, in seconds). This 'serial mode' is the easiest and most robust method, but it does not allow the execution of other instructions in the meantime.

.. code-block:: python

  # Code 1 {}
  oximeter.read(duration=10)
  # Code 2 {}

* The ``readInWaiting()`` method

will only read the bytes temporally stored in the USB buffer. For the Nonin device, this represents up to 10 seconds of recording (this procedure should be executed at least one time every 10 seconds for a continuous recording). When inserted into a while loop, it can record PPG signal in parallel with other commands.

.. code-block:: python

  import time
  tstart = time.time()
  while time.time() - tstart < 10:
      oximeter.readInWaiting()
      # Insert code here {...}
