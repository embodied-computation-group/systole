
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

.. image:: https://badges.gitter.im/ecg-systole/community.svg
   :target: https://gitter.im/ecg-systole/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge

================

.. figure:: https://github.com/embodied-computation-group/systole/blob/dev/source/images/logo.png
   :align: center

================

**Systole** is an open-source Python package implementing simple tools for working with cardiac signals for psychophysiology research. In particular, the package provides tools to pre-process, visualize, and analyze cardiac data. 
This includes tools for data epoching, artefact detection, artefact correction, evoked heart rate analyses, heart rate 
variability analyses, circular statistical approaches to analysing cardiac cycles, and synchronising stimulus 
presentation with different cardiac phases via Psychopy.

If you have questions, you can ask them in the `Gitter chat <https://gitter.im/ecg-systole/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge>`_.

Systole documentation
=====================

.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex

    ---
    :img-top: images/forward-fast-solid.png

    Getting started
    ^^^^^^^^^^^^^^^

    New to *Systole*? Check out the getting started guides. They contain an
    introduction to *Systole'* main concepts and links to additional tutorials.

    +++

    .. link-button:: getting_started
            :type: ref
            :text: To the getting started guides
            :classes: btn btn-light btn-block stretched-link

    ---
    :img-top: images/table-cells-large-solid.png

    Example gallery
    ^^^^^^^^^^^^^^^

    See this section for examples of using Systole in different ways.

    +++

    .. link-button:: auto_examples/index
            :type: ref
            :text: To the example gallery
            :classes: btn btn-light btn-block stretched-link

    ---
    :img-top: images/tutorials.png

    Tutorials
    ^^^^^^^^^

    New to cardiac signal analysis? Want to see how you can use *Systole* when dealing
    with real-world problems? Check out the tutorial notebooks for an introduction to
    theoretical and practical aspects of physiological signal analysis for cognitive
    neuroscience.

    +++

    .. link-button:: tutorials
            :type: ref
            :text: To the tutorial notebooks
            :classes: btn btn-light btn-block stretched-link

    ---
    :img-top: images/code-solid.png

    API reference
    ^^^^^^^^^^^^^

    The reference guide contains a detailed description of the Systole API. The
    reference describes how the methods work and which parameters can be used.

    +++

    .. link-button:: api
            :type: ref
            :text: To the reference guide
            :classes: btn btn-light btn-block stretched-link


Acknowledgements
================

This software and the ECG are supported by a Lundbeckfonden Fellowship (R272-2017-4345), and the AIAS-COFUND II fellowship programme that is supported by the Marie Skłodowska-Curie actions under the European Union’s Horizon 2020 (Grant agreement no 754513), and the Aarhus University Research Foundation.

Systole was largely inspired by pre-existing toolboxes dedicated to heartrate variability and signal analysis.

* HeartPy: https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/

* hrv: https://github.com/rhenanbartels/hrv

* pyHVR: https://pyhrv.readthedocs.io/en/latest/index.html

* ECG-detector: https://github.com/berndporr/py-ecg-detectors

* Pingouin: https://pingouin-stats.org/

* NeuroKit2: https://github.com/neuropsychology/NeuroKit


Development
===========

This module was created and is maintained by Nicolas Legrand and Micah Allen (ECG group, https://the-ecg.org/). If you want to contribute, feel free to contact one of the developers, open an issue or submit a pull request.

This program is provided with NO WARRANTY OF ANY KIND.

Contributors
============

- Jan C. Brammer (jan.c.brammer@gmail.com)
- Gidon Levakov (gidonlevakov@gmail.com)
- Peter Doggart (peter.doggart@pulseai.io)

================

|AU| |lundbeck| |lab|

.. |AU| image::  https://github.com/embodied-computation-group/systole/blob/dev/source/images/au_clinisk_logo.png
   :width: 100%

.. |lundbeck| image::  https://github.com/embodied-computation-group/systole/blob/dev/source/images/lundbeckfonden_logo.png
   :width: 10%

.. |lab| image::  https://github.com/embodied-computation-group/systole/blob/dev/source/images/LabLogo.png
   :width: 20%


.. toctree::
   :maxdepth: 3
   :hidden:

   Getting started <getting_started>
   Gallery <auto_examples/index.rst>
   Tutorials <tutorials.rst>
   API reference <api.rst>
   Release notes <changelog.rst>