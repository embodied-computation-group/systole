
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
* `Numba <http://numba.pydata.org/>`_ (>=0.51.2)

Interactive plotting functions and reports generation will also require the following packages to be installed:

* `Bokeh <https://docs.bokeh.org/en/latest/index.html#>`_ (>=2.3.3)

Recording
=========

Systole natively supports recording of physiological signals from the following setups:
* `Nonin 3012LP Xpod USB pulse oximeter <https://www.nonin.com/products/xpod/>`_ together with the `Nonin 8000SM 'soft-clip' fingertip sensors <https://www.nonin.com/products/8000s/>`_ (USB).
* Remote Data Access (RDA) via BrainVision Recorder together with `Brain product ExG amplifier <https://www.brainproducts.com/>`_ (Ethernet).

Features extraction
===================

Cardiac cycles
--------------

Systole implements fast systolic peak detection for both photoplesthymography (PPG) and electrocardiography (ECG).

.. code-block:: python

  from systole import simulate_rr
  from systole.plots import plot_subspaces

  rr = simulate_rr()
  plot_subspaces(rr)

.. figure::  https://github.com/embodied-computation-group/systole/raw/master/Images/subspaces.png
   :align:   center

Artefacts correction
--------------------
Artefacts can be detected and corrected in the RR interval time serie or the peaks vector following the algorythm proposed by Lipponen & Tarvainen (2019).


Heartrate variability
======================

Systole implemetns basic time-domain, frequency-domain and non-linear HRV indices.

Interactive reports
===================

Systole integrates a set of functions for interactive data visualization based on `Bokeh <https://docs.bokeh.org/en/latest/index.html#>`_.

.. code-block:: python

  from systole.plots import plot_frequency

  plot_psd(rr)

.. figure::  https://github.com/embodied-computation-group/systole/raw/master/Images/psd.png
   :align:   center