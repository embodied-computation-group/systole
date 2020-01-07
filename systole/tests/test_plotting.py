# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import unittest
import matplotlib
from unittest import TestCase
from systole.plotting import plot_hr, plot_events, plot_oximeter, plot_peaks,\
    plot_subspaces
from systole import import_ppg, import_rr
from systole.recording import Oximeter
from systole.detection import hrv_subspaces

# Simulate oximeter instance from recorded signal
ppg = import_ppg()
oxi = Oximeter(serial=None, add_channels=1)
oxi.threshold = [0] * 75
oxi.peaks = [0] * 75
oxi.instant_rr = [0] * 75
oxi.recording = list(ppg[0, :75])
for i in range(len(ppg[0, 75:750])):
    oxi.add_paquet(ppg[0, 75+i])
oxi.channels['Channel_0'] = np.zeros(750, dtype=int)
oxi.channels['Channel_0'][np.random.choice(np.arange(0, 750), 5)] = 1
oxi.times = list(np.arange(0, 10, 1/75))


class TestPlotting(TestCase):

    def test_plot_hr(self):
        ax = plot_hr(oxi)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_events(self):
        ax = plot_events(oxi)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_oximeter(self):
        ax = plot_oximeter(oxi)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_peaks(self):
        ax = plot_peaks(oxi.peaks)
        assert isinstance(ax, matplotlib.axes.Axes)

    def plot_subspaces(self):
        rr = import_rr()
        s1, s2, s3 = hrv_subspaces(rr)
        ax = plot_subspaces(s1, s2, s3)
        assert isinstance(ax[0], matplotlib.axes.Axes)
        assert isinstance(ax[1], matplotlib.axes.Axes)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
