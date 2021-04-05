
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

* `Numpy <https://numpy.org/>`_ (>=1.15)
* `SciPy <https://www.scipy.org/>`_ (>=1.3.0)
* `Pandas <https://pandas.pydata.org/>`_ (>=0.24)
* `Matplotlib <https://matplotlib.org/>`_ (>=3.0.2)
* `Seaborn <https://seaborn.pydata.org/>`_ (>=0.9.0)
* `py-ecg-detectors <https://github.com/berndporr/py-ecg-detectors>`_ (>=1.0.2)

Interactive plotting functions and reports generation will also require the following packages to be installed:

* `Plotly <https://plotly.com/>`_ (>=4.8.0)

Tutorial
========

For an overview of all the recording functionalities, you can refer to the following examples:

* Recording
* Artefacts detection and artefacts correction
* Heart rate variability

For an introduction to Systole and cardiac signal analysis, you can refer to the following tutorial:

* Introduction to cardiac signal analysis - |Colab badge| - `Jupyter Book <https://legrandnico.github.io/Notebooks/IntroductionCardiacSignalAnalysis.html>`_ 

.. |Colab badge| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/LegrandNico/Notebooks/blob/main/IntroductionCardiacSignalAnalysis.ipynb

Recording
=========

Systole natively supports recording of physiological signals from the following setups:
* `Nonin 3012LP Xpod USB pulse oximeter <https://www.nonin.com/products/xpod/>`_ together with the `Nonin 8000SM 'soft-clip' fingertip sensors <https://www.nonin.com/products/8000s/>`_ (USB).
* Remote Data Access (RDA) via BrainVision Recorder together with `Brain product ExG amplifier <https://www.brainproducts.com/>`_ (Ethernet).

Artefact correction
===================

Systole implements systolic peak detection inspired by van Gent et al. (2019) [#]_ and the artefact rejection method recently proposed by Lipponen & Tarvainen (2019) [#]_.

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

.. [#] van Gent, P., Farah, H., van Nes, N., & van Arem, B. (2019). HeartPy: A novel heart rate algorithm for the analysis of noisy signals. *Transportation Research Part F: Traffic Psychology and Behaviour, 66, 368–378*. https://doi.org/10.1016/j.trf.2019.09.015

**Artefact detection and correction:**

.. [#] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for heart rate variability time series artefact correction using novel beat classification. *Journal of Medical Engineering & Technology, 43(3), 173–181*. https://doi.org/10.1080/03091902.2019.1640306
