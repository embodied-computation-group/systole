
.. image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
  :target: https://github.com/embodied-computation-group/systole/blob/master/LICENSE

.. image:: https://badge.fury.io/py/systole.svg
    :target: https://badge.fury.io/py/systole

.. image:: https://zenodo.org/badge/219720901.svg
   :target: https://zenodo.org/badge/latestdoi/219720901

.. image:: https://travis-ci.org/embodied-computation-group/systole.svg?branch=master
   :target: https://travis-ci.org/embodied-computation-group/systole

.. image:: https://codecov.io/gh/embodied-computation-group/systole/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/embodied-computation-group/systole

================

.. figure::  https://github.com/embodied-computation-group/systole/raw/master/source/images/banner.png
   :align:   center

================

**Systole** is an open-source Python package providing simple tools to record and analyze, cardiac signals for psychophysiology.
In particular, the package provides tools to pre-process, analyze, and synchronize cardiac data from psychophysiology research.
This includes tools for data epoching, heart-rate variability, and synchronizing stimulus presentation with different cardiac phases via psychopy.

The documentation can be found under the following `link <https://systole-docs.github.io/>`_.

Installation
============

Systole can be installed using pip:

.. code-block:: shell

  pip install systole

The following packages are required to use Systole:

* Numpy (>=1.15)
* SciPy (>=1.3.0)
* Pandas (>=0.24)
* Matplotlib (>=3.0.2)
* Seaborn (>=0.9.0)

Recording
=========

Systole natively supports the recording of PPG signals through the `Nonin 3012LP Xpod USB pulse oximeter <https://www.nonin.com/products/xpod/>`_ together with the `Nonin 8000SM 'soft-clip' fingertip sensors <https://www.nonin.com/products/8000s/>`_.
It can easily interface with `PsychoPy <https://www.psychopy.org/>`_ to record PPG signal during psychological experiments, and to synchronize stimulus deliver to e.g., systole or diastole.

For example, you can record and plot data in less than 6 lines of code:

.. code-block:: python

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

Online detection
----------------

Online heart beat detection, for cardiac-stimulus synchrony:

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

Peaks detection
===============

Heartbeats can be detected in the PPG signal either online or offline.

Methods from clipping correction and peak detection algorithm is adapted from [#]_.

.. code-block:: python

  # Plot data
  oxi.plot_oximeter()

.. figure::  https://github.com/embodied-computation-group/systole/raw/master/Images/recording.png
   :align:   center

Artefact correction
===================

Systole implements the artefact rejection method recently proposed by Lipponen & Tarvainen (2019) [#]_.

.. code-block:: python

  from systole import simulate_rr
  from systole.plotting import plot_subspaces

  rr = simulate_rr()
  plot_subspaces(rr)

.. figure::  https://github.com/embodied-computation-group/systole/raw/master/Images/subspaces.png
   :align:   center

Interactive visualization
=========================

Systole integrates a set of functions for interactive data visualization based on `Plotly <https://plotly.com/>`_.

.. figure::  https://github.com/embodied-computation-group/systole/raw/master/Images/systole.gif
   :align:   center

Heartrate variability
======================

Systole supports basic time-domain, frequency-domain and non-linear extraction indices.

All time-domain and non-linear indices have been tested against Kubios HVR 2.2 (<https://www.kubios.com>). The frequency-domain indices can slightly differ. We recommend to always check your results against another software.

.. code-block:: python

  from systole.plotting import plot_psd

  plot_psd(rr)

.. figure::  https://github.com/embodied-computation-group/systole/raw/master/Images/psd.png
   :align:   center

Development
===========

This module was created and is maintained by Nicolas Legrand and Micah Allen (ECG group, https://the-ecg.org/). If you want to contribute, feel free to contact one of the developers, open an issue or submit a pull request.

This program is provided with NO WARRANTY OF ANY KIND.

Contributors
============

- Jan C. Brammer (jan.c.brammer@gmail.com)

Acknowledgements
================

This software and the ECG are supported by a Lundbeckfonden Fellowship (R272-2017-4345), and the AIAS-COFUND II fellowship programme that is supported by the Marie Skłodowska-Curie actions under the European Union’s Horizon 2020 (Grant agreement no 754513), and the Aarhus University Research Foundation.

Systole was largely inspired by pre-existing toolboxes dedicated to heartrate variability and signal analysis.

* HeartPy: https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/

* hrv: https://github.com/rhenanbartels/hrv

* pyHVR: https://pyhrv.readthedocs.io/en/latest/index.html

* ECG-detector: https://github.com/berndporr/py-ecg-detectors

* Pingouin: https://pingouin-stats.org/

References
==========

**Peak detection (PPG signal)**

.. [#] van Gent, P., Farah, H., van Nes, N., & van Arem, B. (2019). HeartPy: A novel heart rate algorithm for the analysis of noisy signals. Transportation Research Part F: Traffic Psychology and Behaviour, 66, 368–378. https://doi.org/10.1016/j.trf.2019.09.015

**Artefact detection and correction:**

.. [#] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for heart rate variability time series artefact correction using novel beat classification. *Journal of Medical Engineering & Technology, 43(3), 173–181*. https://doi.org/10.1080/03091902.2019.1640306
