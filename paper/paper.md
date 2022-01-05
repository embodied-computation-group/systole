---
title: 'Systole: A python package for cardiac signal synchrony and analysis'
tags:
  - python
  - heart rate variability
  - psychology
  - electrocardiography
  - photoplethysmography
authors:
  - name: Nicolas Legrand
    orcid: 0000-0002-2651-4929
    affiliation: "1"
  - name: Micah Allen
    orcid:  0000-0001-9399-4179
    affiliation: "1, 2, 3"
affiliations:
 - name: Center of Functionally Integrative Neuroscience, Aarhus University Hospital, Denmark
   index: 1
 - name: Aarhus Institute of Advanced Studies, Aarhus University, Denmark
   index: 2
 - name: Cambridge Psychiatry, University of Cambridge, United Kingdom
   index: 3
date: 5 January 2022
bibliography: paper.bib
---

# Summary

Systole is a package for cardiac signal analysis in Python. It provides an interface 
for recording cardiac signals via electrocardiography (ECG) or photoplethysmography 
(PPG), as well as both online and offline data analysis methods extracting cardiac 
features, synchronizing experimental stimuli with different phases of the heart, 
removing artefacts at different levels and generating plots for data quality check and 
publication. Systole is built on the top of Numpy [@harris:2020], Pandas [@reback2020pandas; @mckinney-proc-scipy-2010] and Scipy [@SciPy:2020], and can use both 
Matplotlib [@hunter:2007] and Bokeh [@bokeh] to generate visualisations. It is designed to build modular 
pipelines that can interface easily with other signal processing or heart rate 
variability toolboxes, with a focus on data quality checks. Several parts of the 
toolbox utilize Numba [@numba] to offer more competitive performances with classic processing
 algorithms.

# Statement of need

Analysis of cardiac data remains a major component of medical, physiological, 
neuroscientific, and psychological research. In psychology, for example, rising 
interest in interoception (i.e., sensing of cardiac signals) has led to a 
proliferation of studies synchronizing the onset or timing of experimental stimuli to 
different phases of the heart. Similarly, there is rising interest in understanding how 
heart-rate variability relates to mental illness, cognitive function, and physical 
wellbeing. This diverse interest calls for more open-source tools designed to work 
with cardiac data in the context of psychology research, but to date, only limited 
options exist. To improve the reproducibility and accessibility of advanced forms of 
cardiac physiological analysis, such as online peak detection and artefact control, 
we here introduce a fully documented Python package, Systole. Systole introduces core 
functionalities to interface with pulse oximeters devices, accelerated peaks detection 
and artefacts rejection algorithms as well as a complete suite of interactive and 
non-interactive plotting functions to improve quality checks and reports creation in 
the context of physiological signal analysis. 


# Overview
The package focuses on 5 core functional elements. The documentation of the package 
includes extensive tutorial notebooks and examples vignettes illustrating these points. 
It can be found at [https://systole-docs.github.io/](https://systole-docs.github.io/).
The package has already been used in two publications that also describe some example 
uses [@legrand:2020; @legrand:2021].

Core functionalities: 

1. Signal extraction and interactive plotting.
Systole uses adapted versions of ECG [@luis:2019] and PPG [@van_gent:2019] 
peaks detectors accelerated with Numba [@numba] for increased performances. 
It also implements plotting functionalities for RR time series and heart rate 
variability visualization both on Matplotlib [@hunter:2007] and Bokeh [@bokeh]. This 
API is designed and developed to facilitate the automated generation of interactive 
reports and dashboards from large datasets of physiological data.

1. Artefact detection and rejection.
The current artefact detection relies on the Python adaptation of the method proposed 
by @lipponen:2019. The correcting of the artefacts is modular and can be 
adapted to specific uses. Options are provided to control for signal length and events 
synchronization while correcting.

3. Instantaneous and evoked heart-rate analysis.
Systole includes utilities and plotting functions for instantaneous and evoked heart 
rate analysis from raw signal or RR time series data.

4. Heart-rate variability analysis.
The package integrates functions for heart rate variability analysis in the time, 
frequency and non-linear domain. This includes the most widely used metrics of heart 
rate variability in cognitive science. Other metric or feature extractions can be 
performed by interfacing with other packages that are more specialized in this domain 
[@gomes:2019; @makowski:2021].

5. Online systolic peak detection, cardiac-stimulus synchrony, and cardiac circular analysis. 
The package currently supports recording and synchronization with experiment software 
like Psychopy [@peirce:2019] from Nonin 3012LP Xpod USB pulse oximeter together 
with the Nonin 8000SM ‘soft-clip’ fingertip sensors (USB) as well as Remote Data Access 
(RDA) via BrainVision Recorder together with Brain product ExG amplifier (Ethernet).

# References
