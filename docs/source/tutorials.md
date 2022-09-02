# Tutorials

Systole is oriented to physiological signal analysis, with the main focus on cardiac activity. The notion that cognition and metacognition are influenced by body states and physiological signals - heart rate, and heart rate variability being the most prominent ones - is extensively investigated by cognitive neuroscience. Systole_ was first developed to help
interfacing psychological tasks (e.g using Psychopy) with physiological recording in the context of psychophysiology experiments. It has progressively grown and integrates more algorithms and metrics, and embeds several static and dynamic plotting utilities that we think make it useful to generate reports, helps to produce clear and transparent research outputs, and participates in creating more robust and open science overall.

These tutorials introduce both the basic concepts of cardiac signal analysis and their implementation with examples of analysis using Systole. The focus and the illustrations are largely derived from the cognitive science approach and interest, but it can be of interest to anyone willing to use Systole or learn about cardiac signal analysis in general.

```{toctree}
---
hidden:
glob:
---

notebooks/*

```

| Notebook | Colab |
| --- | ---|
| {ref}`physiological_signals` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/systole/blob/dev/docs/source/notebooks/1-PhysiologicalSignals.ipynb)
| {ref}`cardiac_cycles` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/systole/blob/dev/docs/source/notebooks/2-DetectingCycles.ipynb)
| {ref}`correcting_artefacts` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/systole/blob/dev/docs/source/notebooks/3-DetectingAndCorrectingArtefacts.ipynb)
| {ref}`hrv` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/systole/blob/dev/docs/source/notebooks/4-HeartRateVariability.ipynb)
| {ref}`instantaneous_heart_rate` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/systole/blob/dev/docs/source/notebooks/5-InstantaneousHeartRate.ipynb)
| {ref}`bids_folders` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/systole/blob/dev/docs/source/notebooks/6-WorkingWithBIDSFolders.ipynb)
