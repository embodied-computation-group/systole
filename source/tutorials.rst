.. _api_ref:

Tutorials
=========

Systole_ is a Python package oriented to physiological
signal analysis, with the main focus on cardiac activity. The notion that cognition
and metacognition are influenced by body states and physiological signals - heart
rate, and heart rate variability being the most prominent ones - is extensively
investigated by cognitive neuroscience. Systole_ was first developed to help
interfacing psychological tasks (e.g using Psychopy) 
with physiological recording in the context of psychophysiology experiments. It 
has progressively grown and integrates more algorithms and metrics, and embeds 
several static and dynamic plotting utilities that we think make it useful to 
generate reports, helps to produce clear and transparent research outputs, and 
participates in creating more robust and open science overall.

These tutorials introduce both the basic concepts of cardiac signal analysis and their
implementation with examples of analysis using Systole. The focus and the illustrations
are largely derived from the cognitive science approach and interest, but it can be
of interest to anyone willing to use Systole or learn about cardiac signal analysis
in general.

.. _Systole: https://systole-docs.github.io/

.. panels::

   - 1 - Cardiac signals
   ^^^^^^^^^^^^^^^^^^^^^

   .. toctree::
      :maxdepth: 4

      notebooks/1-PhysiologicalSignals.ipynb

   +++
   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/embodied-computation-group/systole/blob/dev/source/notebooks/1-PhysiologicalSignals.ipynb
      :align: center

   ---

   - 2 - Cardicac cycles
   ^^^^^^^^^^^^^^^^^^^^^

   .. toctree::
      :maxdepth: 4

      notebooks/2-DetectingCycles.ipynb

   +++
   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/embodied-computation-group/systole/blob/dev/source/notebooks/2-DetectingCycles.ipynb
      :align: center

   ---

   - 3 - Artefacts
   ^^^^^^^^^^^^^^^

   .. toctree::
      :maxdepth: 4

      notebooks/3-DetectingAndCorrectingArtefacts.ipynb

   +++
   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/embodied-computation-group/systole/blob/dev/source/notebooks/3-DetectingAndCorrectingArtefacts.ipynb
      :align: center
      
   ---

   - 4 - HRV
   ^^^^^^^^^

   .. toctree::
      :maxdepth: 4

      notebooks/4-HeartRateVariability.ipynb

   +++
   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/embodied-computation-group/systole/blob/dev/source/notebooks/4-HeartRateVariability.ipynb
      :align: center

   ---

   - 5 - Evoked HRV
   ^^^^^^^^^^^^^^^^

   .. toctree::
      :maxdepth: 4

      notebooks/5-InstantaneousHeartRate.ipynb

   +++
   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/embodied-computation-group/systole/blob/dev/source/notebooks/5-InstantaneousHeartRate.ipynb
      :align: center
