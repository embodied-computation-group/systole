# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import os
import unittest
import matplotlib
import numpy as np
from unittest import TestCase
from systole import serialSim
from systole.recording import Oximeter


class TestRecording(TestCase):

    def test_oximeter(self):

        serial = serialSim()
        oxi = Oximeter(serial=serial, add_channels=1)
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
        oxi.find_peaks()

        oxi.peaks = []
        oxi.instant_rr = []
        oxi.times = []
        oxi.threshold = []

        oxi.save('test')
        assert os.path.exists("test.npy")
        os.remove("test.npy")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
