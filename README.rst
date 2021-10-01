
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

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :target: https://github.com/psf/black

.. image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
  :target: https://pycqa.github.io/isort/

.. image:: http://www.mypy-lang.org/static/mypy_badge.svg
  :target: http://mypy-lang.org/

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
  :target: https://github.com/pre-commit/pre-commit

================

.. figure::  https://github.com/embodied-computation-group/systole/raw/master/source/images/banner.png
   :align:   center

================

**Systole** is an open-source Python package implementing simple tools for working with cardiac signals for 
psychophysiology research. In particular, the package provides tools to pre-process, visualize, and analyze cardiac data. 
This includes tools for data epoching, artefact detection, artefact correction, evoked heart-rate analyses, heart-rate 
variability analyses, circular statistical approaches to analysing cardiac cycles, and synchronising stimulus 
presentation with different cardiac phases via the psychopy.


The documentation can be found under the following `link <https://systole-docs.github.io/>`_.

Installation
++++++++++++

Systole can be installed using pip:

.. code-block:: shell

  pip install systole

The following packages are required to use Systole:

* `Numpy <https://numpy.org/>`_ (>=1.15)
* `SciPy <https://www.scipy.org/>`_ (>=1.3.0)
* `Pandas <https://pandas.pydata.org/>`_ (>=0.24)
* `Numba <http://numba.pydata.org/>`_ (>=0.51.2)
* `Seaborn <https://seaborn.pydata.org/>`_ (>=0.9.0)

Required when using `Matplotlib` plotting backend:
* `Matplotlib <https://matplotlib.org/>`_ (>=3.0.2)
Required when using `Bokeh` plotting backend:
* `Bokeh <https://docs.bokeh.org/en/latest/index.html#>`_ (>=2.3.3)


Tutorials
=========

For an introduction to Systole and cardiac signal analysis, you can refer to the following tutorial:

1. Cardiac signal analysis - |Colab badge 1|
2. Detecting cardiac cycles - |Colab badge 2|
3. Detecting and correcting artefats - |Colab badge 3|
4. Heart rate variability - |Colab badge 4|
5. Instantaneous and evoked heart rate - |Colab badge 5|

.. |Colab badge 1| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/embodied-computation-group/systole/blob/dev/source/notebooks/1-PhysiologicalSignals.ipynb

.. |Colab badge 2| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/embodied-computation-group/systole/blob/dev/source/notebooks/2-DetectingCycles.ipynb

.. |Colab badge 3| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/embodied-computation-group/systole/blob/dev/source/notebooks/3-DetectingAndCorrectingArtefacts.ipynb

.. |Colab badge 4| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/embodied-computation-group/systole/blob/dev/source/notebooks/4-HeartRateVariability.ipynb

.. |Colab badge 5| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/embodied-computation-group/systole/blob/dev/source/notebooks/5-InstantaneousHeartRate.ipynb


Getting started
+++++++++++++++

.. code-block:: python

  from systole import import_dataset1

  # Import ECg recording
  signal = import_dataset1(modalities=['ECG']).ecg.to_numpy()


Signal extraction and interactive plotting
==========================================
The package integrates a set of functions for interactive or non interactive data visualization based on `Matplotlib <https://matplotlib.org/>`_ and `Bokeh <https://docs.bokeh.org/en/latest/index.html#>`_.

.. code-block:: python

  from systole.plots plot_raw

  plot_raw(signal[60000 : 120000], modality="ecg", backend="bokeh", 
              show_heart_rate=True, show_artefacts=True, figsize=300)

.. figure::  https://github.com/embodied-computation-group/systole/raw/master/source/images/raw.png
   :align:   center


Artefacts detection and rejection
=================================
Artefacts can be detected and corrected in the RR interval time series or the peaks vector following the algorythm proposed by Lipponen & Tarvainen (2019).

.. code-block:: python

  from systole.detection import ecg_peaks
  from systole.plots plot_subspaces

  # R peaks detection
  signal, peaks = ecg_peaks(signal, method='pan-tompkins', sfreq=1000)

  plot_subspaces(peaks, input_type="peaks", backend="bokeh")

.. figure::  https://github.com/embodied-computation-group/systole/raw/master/source/images/subspaces.png
   :align:   center

Instantaneous and evoked heart rate
===================================


Heart rate variability analysis
===============================
Systole implemetns basic time-domain, frequency-domain and non-linear HRV indices.

.. code-block:: python

  from bokeh.layouts import row
  from systole.plots plot_frequency, plot_pointcare

  row(
      plot_frequency(peaks, input_type="peaks", backend="bokeh", figsize=(600, 400)),
      plot_pointcare(peaks, input_type="peaks", backend="bokeh", figsize=(400, 400)),
      )

.. figure::  https://github.com/embodied-computation-group/systole/raw/master/source/images/hrv.png
   :align:   center


Online systolic peak detection, cardiac-stimulus synchrony, and cardiac circular analysis
=========================================================================================

Systole natively supports recording of physiological signals from the following setups:
- `Nonin 3012LP Xpod USB pulse oximeter <https://www.nonin.com/products/xpod/>`_ together with the `Nonin 8000SM 'soft-clip' fingertip sensors <https://www.nonin.com/products/8000s/>`_ (USB).
- Remote Data Access (RDA) via BrainVision Recorder together with `Brain product ExG amplifier <https://www.brainproducts.com/>`_ (Ethernet).


Development
+++++++++++

This module was created and is maintained by Nicolas Legrand and Micah Allen (ECG group, https://the-ecg.org/). If you want to contribute, feel free to contact one of the developers, open an issue or submit a pull request.

This program is provided with NO WARRANTY OF ANY KIND.

Contributors
++++++++++++

- Jan C. Brammer (jan.c.brammer@gmail.com)
- Gidon Levakov (gidonlevakov@gmail.com)
- Peter Doggart (peter.doggart@pulseai.io)

Acknowledgements
++++++++++++++++

This software and the ECG are supported by a Lundbeckfonden Fellowship (R272-2017-4345), and the AIAS-COFUND II fellowship programme that is supported by the Marie Skłodowska-Curie actions under the European Union’s Horizon 2020 (Grant agreement no 754513), and the Aarhus University Research Foundation.

Systole was largely inspired by pre-existing toolboxes dedicated to heartrate variability and signal analysis.

* HeartPy: https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/

* hrv: https://github.com/rhenanbartels/hrv

* pyHVR: https://pyhrv.readthedocs.io/en/latest/index.html

* ECG-detector: https://github.com/berndporr/py-ecg-detectors

* Pingouin: https://pingouin-stats.org/

* NeuroKit2: https://github.com/neuropsychology/NeuroKit

================

|AU| |lundbeck| |lab|

.. |AU| image::  https://github.com/embodied-computation-group/systole/raw/dev/Images/au_clinisk_logo.png
   :width: 100%

.. |lundbeck| image::  https://github.com/embodied-computation-group/systole/raw/dev/Images/lundbeckfonden_logo.png
   :width: 10%

.. |lab| image::  https://github.com/embodied-computation-group/systole/raw/dev/Images/LabLogo.png
   :width: 20%