.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_plot_HeartBeatEvokedArpeggios.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_HeartBeatEvokedArpeggios.py:


Heart Beats Evoked Arpeggios
============================

This tutorial illustrates how to use the ``Oximeter`` class to triggers stimuli
at different cardiac cycles using the [Psychopy](https://www.psychopy.org/)
toolbox. The PPG signal is recorded for 30 seconds and peaks are detected
online. Four notes ('C', 'E', 'G', 'Bfl') are played in synch with peak
detection with various delay: no delay,  1/4, 2/4 or 3/4 of the previous
cardiac cycle length. While R-R intervals are prone to large changes in the
long term, such changes are physiologically limited for heartbeat, thus
limiting the variability of phase in which the note is played.


.. code-block:: default


    # Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
    # Licence: GPL v3

    import time
    from systole import serialSim
    from psychopy.sound import Sound
    from systole.circular import to_angles, circular
    from systole.recording import Oximeter
    import matplotlib.pyplot as plt
    import numpy as np








Recording
---------
For the purpose of demonstration, here we simulate data acquisition through
the pulse oximeter using pre-recorded signal.


.. code-block:: default


    ser = serialSim()








If you want to allow online data acquisition, you should uncomment the
following lines and provide the reference of the COM port where the pulse
oximeter is plugged.

.. code-block:: python

  import serial
  ser = serial.Serial('COM4')  # Change this value according to your setup

Create an Oxymeter instance, initialize recording and record for 10 seconds


.. code-block:: default


    oxi = Oximeter(serial=ser, sfreq=75, add_channels=4).setup()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Reset input buffer




Create an Oxymeter instance, initialize recording and record for 10 seconds


.. code-block:: default


    systole = Sound('C', secs=0.1)
    diastole1 = Sound('E', secs=0.1)
    diastole2 = Sound('G', secs=0.1)
    diastole3 = Sound('Bfl', secs=0.1)

    systoleTime1, systoleTime2, systoleTime3 = None, None, None
    tstart = time.time()
    while time.time() - tstart < 10:

        # Check if there are new data to read
        while oxi.serial.inWaiting() >= 5:

            # Convert bytes into list of int
            paquet = list(oxi.serial.read(5))

            if oxi.check(paquet):  # Data consistency
                oxi.add_paquet(paquet[2])  # Add new data point

            # T + 0
            if oxi.peaks[-1] == 1:
                systole = Sound('C', secs=0.1)
                systole.play()
                systoleTime1 = time.time()
                systoleTime2 = time.time()
                systoleTime3 = time.time()

            # T + 1/4
            if systoleTime1 is not None:
                if time.time() - systoleTime1 >= ((oxi.instant_rr[-1]/4)/1000):
                    diastole1 = Sound('E', secs=0.1)
                    diastole1.play()
                    systoleTime1 = None

            # T + 2/4
            if systoleTime2 is not None:
                if time.time() - systoleTime2 >= (((oxi.instant_rr[-1]/4) * 2)/1000):
                    diastole2 = Sound('G', secs=0.1)
                    diastole2.play()
                    systoleTime2 = None

            # T + 3/4
            if systoleTime3 is not None:
                if time.time() - systoleTime3 >= (((oxi.instant_rr[-1]/4) * 3)/1000):
                    diastole3 = Sound('A', secs=0.1)
                    diastole3.play()
                    systoleTime3 = None

            # Track the note status
            oxi.channels['Channel_0'][-1] = systole.status
            oxi.channels['Channel_1'][-1] = diastole1.status
            oxi.channels['Channel_2'][-1] = diastole2.status
            oxi.channels['Channel_3'][-1] = diastole3.status








Events
--------
The


.. code-block:: default

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    oxi.plot_recording(ax=ax1)
    oxi.plot_events(ax=ax2)
    plt.tight_layout()





.. image:: /auto_examples/images/sphx_glr_plot_HeartBeatEvokedArpeggios_001.png
    :class: sphx-glr-single-img





Cardiac cycle
-------------


.. code-block:: default

    angles = []
    x = np.asarray(oxi.peaks)
    for ev in oxi.channels:
        events = np.asarray(oxi.channels[ev])
        for i in range(len(events)):
            if events[i] == 1:
                events[i+1:i+10] = 0
        angles.append(to_angles(x, events))

    circular(angles[0], color='gray')
    circular(angles[1], color='r')
    circular(angles[2], color='g')
    circular(angles[3], color='b')



.. image:: /auto_examples/images/sphx_glr_plot_HeartBeatEvokedArpeggios_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    C:\ProgramData\Anaconda3\lib\site-packages\systole-0.0.1-py3.7.egg\systole\circular.py:69: MatplotlibDeprecationWarning: Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.
      ax = plt.subplot(111, polar=True)

    <matplotlib.axes._subplots.PolarAxesSubplot object at 0x0000026794B6E5C0>




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  16.941 seconds)


.. _sphx_glr_download_auto_examples_plot_HeartBeatEvokedArpeggios.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_HeartBeatEvokedArpeggios.py <plot_HeartBeatEvokedArpeggios.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_HeartBeatEvokedArpeggios.ipynb <plot_HeartBeatEvokedArpeggios.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
