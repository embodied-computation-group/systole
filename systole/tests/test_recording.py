# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
import matplotlib
import numpy as np
from unittest import TestCase
from systole import serialSim
from systole.recording import Oximeter


serial = serialSim()
oxi = Oximeter(serial=serial, add_channels=1)


class TestRecording(TestCase):

    def test_oximeter(self):
        oxi.setup()
        oxi.read(10)
        oxi.find_peaks()

        # Simulate events in recording
        for idx in np.random.choice(len(oxi.recording), 5):
            oxi.channels['Channel_0'][idx] = 1
        ax = oxi.plot_events()
        assert isinstance(ax, matplotlib.axes.Axes)

        ax = oxi.plot_hr()
        assert isinstance(ax, matplotlib.axes.Axes)

        ax = oxi.plot_recording()
        assert isinstance(ax, matplotlib.axes.Axes)

        oxi.readInWaiting()
        oxi.waitBeat()


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
