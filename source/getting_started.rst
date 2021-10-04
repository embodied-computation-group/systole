
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

.. raw:: html
   :file: ./images/raw.html


Artefacts detection and rejection
=================================
Artefacts can be detected and corrected in the RR interval time series or the peaks vector using the method proposed by Lipponen & Tarvainen (2019).

.. code-block:: python

  from systole.detection import ecg_peaks
  from systole.plots plot_subspaces

  # R peaks detection
  signal, peaks = ecg_peaks(signal, method='pan-tompkins', sfreq=1000)

  plot_subspaces(peaks, input_type="peaks", backend="bokeh")

.. raw:: html
   :file: ./images/subspaces.html


Heart rate variability analysis
===============================
Systole implements time-domain, frequency-domain and non-linear HRV indices, as well as tools for evoked heart rate analysis.

.. code-block:: python

  from bokeh.layouts import row
  from systole.plots plot_frequency, plot_pointcare

  row(
      plot_frequency(peaks, input_type="peaks", backend="bokeh", figsize=(300, 200)),
      plot_pointcare(peaks, input_type="peaks", backend="bokeh", figsize=(200, 200)),
      )

.. raw:: html
   :file: ./images/hrv.html


Online systolic peak detection, cardiac-stimulus synchrony, and cardiac circular analysis
=========================================================================================

The package natively supports recording of physiological signals from the following setups:
- `Nonin 3012LP Xpod USB pulse oximeter <https://www.nonin.com/products/xpod/>`_ together with the `Nonin 8000SM 'soft-clip' fingertip sensors <https://www.nonin.com/products/8000s/>`_ (USB).
- Remote Data Access (RDA) via BrainVision Recorder together with `Brain product ExG amplifier <https://www.brainproducts.com/>`_ (Ethernet).
