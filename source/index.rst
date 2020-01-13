.. image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
  :target: https://github.com/LegrandNico/systole/blob/master/LICENSE

================

.. figure::  images/banner.png
   :align:   center

================

**Systole** is an open-source Python package providing simple tools to record and analyze body signal for psychophysiology.

.. toctree::
   :maxdepth: 2

   api.rst
   auto_examples/index.rst

Installation
============

Download the zip file, extract the folder and run from the terminal:

.. code-block:: shell

  python setup.py install

Dependencies
============
numpy
pandas
scipy
matplotlib
seaborn

Quick start
===========

Systole support the recording of PPG signal through the `Nonin 3012LP Xpod USB pulse oximeter <https://www.nonin.com/products/xpod/>`_ together with the `Nonin 8000SM 'soft-clip' fingertip sensors <https://www.nonin.com/products/8000s/>`_.
It can easily interface with `PsychoPy <https://www.psychopy.org/>`_ to record PPG signal during psychological experiments.

Record and plot data with less than 6 lines of code.

.. code-block:: python

  import serial
  from systole.recording import Oximeter
  ser = serial.Serial('COM4')  # Add your USB port here

  # Open serial port, initialize and plot recording for Oximeter
  oxi = Oximeter(serial=ser).setup().read(duration=10)

  # Plot data
  oxi.plot()

.. figure::  images/recording.png
   :align:   center

Recording
=========

2 methods are available to record PPG signal:

* The `read()` method

Will continuously record for certain amount of time (specified by the
`duration` parameter, in seconds). This is the easiest and most robust method,
but it is not possible to run instructions in the meantime (serial mode).

.. code-block:: python

  # Code 1 {}
  oximeter.read(duration=10)
  # Code 2 {}

* The `readInWaiting()` method

Will read all the availlable bytes (up to 10 seconds of recording). When
inserted into a while loop, it allows to record PPG signal in parallel with
other commands.

.. code-block:: python

  import time
  tstart = time.time()
  while time.time() - tstart < 10:
      oximeter.readInWaiting()
      # Insert code here {...}

Online detection
================

Set an online peak detection algorithm in less than 10 lines of code.

.. code-block:: python

  import serial
  import time
  from systole.recording import Oximeter

  # Open serial port
  ser = serial.Serial('COM4')  # Change this value according to your setup

  # Create an Oxymeter instance and initialize recording
  oxi = Oximeter(serial=ser, sfreq=75, add_channels=4).setup()

  # Online peak detection for 10 seconds
  tstart = time.time()
  while time.time() - tstart < 10:
      while oxi.serial.inWaiting() >= 5:
          paquet = list(oxi.serial.read(5))
          oxi.add_paquet(paquet[2])  # Add new data point
          if oxi.peaks[-1] == 1:
            print('Heartbeat detected')

See also a complete tutorial here: <https://github.com/LegrandNico/systole/tree/master/notebooks/HeartBeatEvokedTone.rst>

Peaks detection
===============
Methods from clipping correction and peak detection algorithm is adapted from [#]_.

Artifact removal
================
It is possible to detect and correct outliers from RR time course following the method described in [#]_.

Heart rate variability
======================
Import RR time-serie.

.. code-block:: python

  from systole import import_rr
  rr = import_rr().rr.values

Time-domain
-----------

Extract summary of time-domain indexes.

.. code-block:: python

  from systole.hrv import time_domain

  stats = time_domain(rr)
  stats

.. table:: Output
   :widths: auto

   +-------+-----------+
   |*Value*|*Metric*   |
   +-------+-----------+

Frequency-domain
----------------
.. code-block:: python

  from systole.hrv import hrv_psd1

  hrv_psd(rr)

.. figure::  images/psd.png
   :align:   center

Extract summary of frequency-domain indexes.

.. code-block:: python

  from systole.hrv import frequency_domain

  frequency_domain(rr)

.. table:: Output
   :widths: auto

   +-----------+---------------+
   | *Metric*  | *Value*       |
   +-----------+---------------+


Non-linear
----------

.. code-block:: python

  from systole.hrv import nonlinear

  nonlinear(rr)

.. table:: Output
   :widths: auto

   +-----------+---------------+
   | *Metric*  | *Value*       |
   +-----------+---------------+


All the results have been tested against Kubios HVR 2.2 (<https://www.kubios.com>).
Some variability can be observed with frequency-domain outputs.


Development
===========

This module was created and is maintained by Nicolas Legrand and Micah Allen (ECG group, https://the-ecg.org/). If you want to contribute, feel free to contact one of the contributor, open an issue or submit a pull request.

This program is provided with NO WARRANTY OF ANY KIND.

Acknowledgement
===============
Systole was largely inspired by preexisting toolboxes for heart rate variability and signal analysis.

HeartPy: https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/

hrv: https://github.com/rhenanbartels/hrv

ECG-detector: https://github.com/berndporr/py-ecg-detectors

Pingouin: https://pingouin-stats.org/

References
==========

**Peak detection (PPG signal)**

.. [#] van Gent, P., Farah, H., van Nes, N., & van Arem, B. (2019). HeartPy: A novel heart rate algorithm for the analysis of noisy signals. Transportation Research Part F: Traffic Psychology and Behaviour, 66, 368–378. https://doi.org/10.1016/j.trf.2019.09.015

**Artefact detection and correction:**

.. [#] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for heart rate variability time series artefact correction using novel beat classification. *Journal of Medical Engineering & Technology, 43(3), 173–181*. https://doi.org/10.1080/03091902.2019.1640306
