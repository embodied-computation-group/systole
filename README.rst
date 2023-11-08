
.. image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
  :target: https://github.com/embodied-computation-group/systole/blob/master/LICENSE

.. image:: https://badge.fury.io/py/systole.svg
    :target: https://badge.fury.io/py/systole

.. image:: https://joss.theoj.org/papers/10.21105/joss.03832/status.svg
   :target: https://doi.org/10.21105/joss.03832

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

.. image:: https://badges.gitter.im/ecg-systole/community.svg
   :target: https://gitter.im/ecg-systole/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge

================

.. image:: https://github.com/embodied-computation-group/systole/blob/dev/docs/source/images/logo.png
   :align: center

================

**Systole** is an open-source Python package implementing simple tools for working with cardiac signals for psychophysiology research. In particular, the package provides tools to pre-process, visualize, and analyze cardiac data. 
This includes tools for data epoching, artefact detection, artefact correction, evoked heart rate analyses, heart rate 
variability analyses, circular statistical approaches to analysing cardiac cycles, and synchronising stimulus 
presentation with different cardiac phases via Psychopy.

The documentation can be found under the following `link <https://embodied-computation-group.github.io/systole/#>`_.

If you have questions, you can ask them in the `Gitter chat <https://gitter.im/ecg-systole/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge>`_.

How to cite?
++++++++++++

If you are using **Systole** in a publication we ask you to cite the following paper::

  Legrand N., & Allen, M., (2022). Systole: A python package for cardiac signal synchrony and analysis. Journal of Open Source Software, 7(69), 3832, https://doi.org/10.21105/joss.03832


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
* `Matplotlib <https://matplotlib.org/>`_ (>=3.0.2)
* `Bokeh <https://docs.bokeh.org/en/latest/index.html#>`_ (>=2.3.3)
* `pyserial <https://pyserial.readthedocs.io/en/latest/pyserial.html>`_ (>=3.4)
* `setuptools <https://setuptools.pypa.io/en/latest/>`_ (>=38.4)
* `requests <https://docs.python-requests.org/en/latest/>`_ (>=2.26.0)
* `tabulate <https://github.com/astanin/python-tabulate/>`_ (>=0.8.9)


The Python version should be 3.7 or higher.

Tutorials
=========

For an introduction to Systole and cardiac signal analysis, you can refer to the following tutorial:

.. list-table::
   :widths: 60 40
   :header-rows: 0
   :align: center

   * - Cardiac signal analysis 
     - |Colab badge 1|
   * - Detecting cardiac cycles 
     - |Colab badge 2|
   * - Detecting and correcting artefats 
     - |Colab badge 3|
   * - Heart rate variability 
     - |Colab badge 4|
   * - Instantaneous and evoked heart rate 
     - |Colab badge 5|
   * - Working with BIDS folders
     - |Colab badge 6|

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

.. |Colab badge 6| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/embodied-computation-group/systole/blob/dev/source/notebooks/6-WorkingWithBIDSFolders.ipynb


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

  from systole.plots import plot_raw

  plot_raw(signal[60000 : 120000], modality="ecg", backend="bokeh", 
              show_heart_rate=True, show_artefacts=True, figsize=300)

.. image:: https://github.com/embodied-computation-group/systole/blob/dev/docs/source/images/raw.png
   :align: center


Artefacts detection and rejection
=================================
Artefacts can be detected and corrected in the RR interval time series or the peaks vector using the method proposed by Lipponen & Tarvainen (2019).

.. code-block:: python

  from systole.detection import ecg_peaks
  from systole.plots import plot_subspaces

  # R peaks detection
  signal, peaks = ecg_peaks(signal, method='pan-tompkins', sfreq=1000)

  plot_subspaces(peaks, input_type="peaks", backend="bokeh")

.. image:: https://github.com/embodied-computation-group/systole/blob/dev/docs/source/images/subspaces.png
   :align: center


Heart rate variability analysis
===============================
Systole implements time-domain, frequency-domain and non-linear HRV indices, as well as tools for evoked heart rate analysis.

.. code-block:: python

  from bokeh.layouts import row
  from systole.plots import plot_frequency, plot_poincare

  row(
      plot_frequency(peaks, input_type="peaks", backend="bokeh", figsize=(300, 200)),
      plot_poincare(peaks, input_type="peaks", backend="bokeh", figsize=(200, 200)),
      )

.. image:: https://github.com/embodied-computation-group/systole/blob/dev/docs/source/images/hrv.png
   :align: center


Online systolic peak detection, cardiac-stimulus synchrony, and cardiac circular analysis
=========================================================================================

The package natively supports recording of physiological signals from the following setups:
- `Nonin 3012LP Xpod USB pulse oximeter <https://www.nonin.com/products/xpod/>`_ together with the `Nonin 8000SM 'soft-clip' fingertip sensors <https://www.nonin.com/products/8000s/>`_ (USB).
- Remote Data Access (RDA) via BrainVision Recorder together with `Brain product ExG amplifier <https://www.brainproducts.com/>`_ (Ethernet).

Interactive visualization of BIDS structured datasets
=====================================================

.. code-block:: python

  from systole.viewer import Viewer

  view = Viewer(
      input_folder="/BIDS/folder/path/",
      pattern="task-mytask",
      modality="beh",
      signal_type="ECG"
  )

.. image:: https://github.com/embodied-computation-group/systole/blob/dev/docs/source/images/editor.gif
   :align: center

Inserting and removing peaks
============================

.. image:: https://github.com/embodied-computation-group/systole/blob/dev/docs/source/images/peaks.gif
   :align: center

Annotating bad segments
=======================

.. image:: https://github.com/embodied-computation-group/systole/blob/dev/docs/source/images/segments.gif
   :align: center

Development
+++++++++++

This module was created and is maintained by Nicolas Legrand and Micah Allen (ECG group, https://the-ecg.org/). If you want to contribute, feel free to contact one of the developers, open an issue or submit a pull request.

This program is provided with NO WARRANTY OF ANY KIND.

Acknowledgements
++++++++++++++++

This software and the ECG are supported by a Lundbeckfonden Fellowship (R272-2017-4345), and the AIAS-COFUND II fellowship programme that is supported by the Marie Skłodowska-Curie actions under the European Union’s Horizon 2020 (Grant agreement no 754513), and the Aarhus University Research Foundation.

Systole was largely inspired by pre-existing toolboxes dedicated to heartrate variability and signal analysis.

* HeartPy: https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/

* hrv: https://github.com/rhenanbartels/hrv

* pyHRV: https://pyhrv.readthedocs.io/en/latest/index.html

* ECG-detector: https://github.com/berndporr/py-ecg-detectors

* Pingouin: https://pingouin-stats.org/

* NeuroKit2: https://github.com/neuropsychology/NeuroKit

================

|AU| |lundbeck| |lab|

.. |AU| image::  https://github.com/embodied-computation-group/systole/blob/dev/docs/source/images/au_clinisk_logo.png
   :width: 100%

.. |lundbeck| image::  https://github.com/embodied-computation-group/systole/blob/dev/docs/source/images/lundbeckfonden_logo.png
   :width: 10%

.. |lab| image::  https://github.com/embodied-computation-group/systole/blob/dev/docs/source/images/LabLogo.png
   :width: 20%
