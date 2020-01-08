# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
import time
import matplotlib
import numpy as np
from unittest import TestCase
from systole import import_ppg
from systole.recording import Oximeter


# Simulate serial inputs from ppg recording
# =========================================
class serial():

    def __init__(self):
        self.sfreq = 75
        self.ppg = import_ppg()[0]
        self.start = time.time()

    def inWaiting(self):
        if time.time() - self.start > 1 / self.sfreq:
            self.start = time.time()
            lenInWating = 5
        else:
            lenInWating = 0

        return lenInWating

    def read(self, length):

        if len(self.ppg) == 0:
            self.ppg = import_ppg()[0]

        # Read 1rst item of ppg signal
        rec = self.ppg[:1]
        self.ppg = self.ppg[1:]

        # Build valid paquet
        paquet = [1, 255, rec[0], 127]
        paquet.append(sum(paquet) % 256)

        return paquet[0], paquet[1], paquet[2], paquet[3], paquet[4]

    def reset_input_buffer(self):
        print('Reset input buffer')


serial = serial()
oxi = Oximeter(serial=serial, add_channels=1)


class TestRecording(TestCase):

    def test_oximeter(self):
        oxi.setup()
        oxi.read(5)
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
