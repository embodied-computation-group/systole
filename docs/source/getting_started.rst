.. meta::
   :description: Install Systole and run your first cardiac signal analysis in Python, from
      loading a recording to detecting peaks and computing heart rate variability.
   :keywords: install systole, systole-core, python cardiac analysis, peak detection,
      heart rate variability, getting started


Installation
++++++++++++

The last stable version of Systole can be installed using pip:

.. code-block:: shell

  pip install systole-core

.. note::

   The distribution is named ``systole-core`` on PyPI, but the package is still
   imported as ``systole``::

     import systole

If you want to download the `dev` branch instead and try the last features that are currently under development (and probably a bit unstable), use:

.. code-block:: shell

  pip install "git+https://github.com/embodied-computation-group/systole.git@dev"

The following packages are required to use Systole:

* `Numpy <https://numpy.org/>`_ (>=1.21)
* `SciPy <https://www.scipy.org/>`_ (>=1.3.0)
* `Pandas <https://pandas.pydata.org/>`_ (>=0.24)
* `Numba <http://numba.pydata.org/>`_ (>=0.58.0)
* `Seaborn <https://seaborn.pydata.org/>`_ (>=0.9.0)
* `Matplotlib <https://matplotlib.org/>`_ (>=3.0.2)
* `Bokeh <https://docs.bokeh.org/en/latest/index.html#>`_ (>=3.0.0)
* `Jinja2 <https://jinja.palletsprojects.com/>`_ (>=3.0)
* `pyserial <https://pyserial.readthedocs.io/en/latest/pyserial.html>`_ (>=3.4)
* `setuptools <https://setuptools.pypa.io/en/latest/>`_ (>=38.4)
* `requests <https://docs.python-requests.org/en/latest/>`_ (>=2.26.0)
* `tabulate <https://github.com/astanin/python-tabulate/>`_ (>=0.8.9)
* `sleepecg <https://github.com/cbrnr/sleepecg>`_ (>=0.5.1)
* `joblib <https://joblib.readthedocs.io/>`_ (>=1.3.2)
* `tqdm <https://tqdm.github.io/>`_
* `packaging <https://packaging.pypa.io/>`_


The Python version should be 3.9 or higher.


Getting started
+++++++++++++++

.. code-block:: python

  from systole import import_dataset1

  # Import ECg recording
  signal = import_dataset1(modalities=['ECG']).ecg.to_numpy()


Signal extraction and interactive plotting
==========================================
The package integrates a set of functions for interactive or non interactive data visualization based on `Matplotlib <https://matplotlib.org/>`_ and `Bokeh <https://docs.bokeh.org/en/latest/index.html#>`_.

.. note::

   To display the Bokeh figures inside a Jupyter notebook, Bokeh's own
   ``output_notebook()`` has to be called once per session, before plotting::

     from bokeh.io import output_notebook
     output_notebook()

   Without it the figures are created but nothing is rendered.

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
  from systole.plots plot_frequency, plot_poincare

  row(
      plot_frequency(peaks, input_type="peaks", backend="bokeh", figsize=(300, 200)),
      plot_poincare(peaks, input_type="peaks", backend="bokeh", figsize=(200, 200)),
      )

.. raw:: html
   :file: ./images/hrv.html


Online systolic peak detection, cardiac-stimulus synchrony, and cardiac circular analysis
=========================================================================================

The package natively supports recording of physiological signals from the following setups:
- `Nonin 3012LP Xpod USB pulse oximeter <https://www.nonin.com/products/xpod/>`_ together with the `Nonin 8000SM 'soft-clip' fingertip sensors <https://www.nonin.com/products/8000s/>`_ (USB).
- Remote Data Access (RDA) via BrainVision Recorder together with `Brain product ExG amplifier <https://www.brainproducts.com/>`_ (Ethernet).
