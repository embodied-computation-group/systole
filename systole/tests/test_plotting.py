# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import pandas as pd
import unittest
import pytest
import matplotlib
from unittest import TestCase
from systole.plotting import (
    plot_raw,
    plot_events,
    plot_oximeter,
    plot_subspaces,
    circular,
    plot_circular,
    plot_psd,
)
from systole import import_ppg, import_rr, serialSim
from systole.recording import Oximeter


serial = serialSim()
oxi = Oximeter(serial=serial, add_channels=1).setup().read(10)
oxi.channels["Channel_0"][100] = 1

# Simulate oximeter instance from recorded signal
ppg = import_ppg().ppg.to_numpy()
oxi = Oximeter(serial=None, add_channels=1)
oxi.threshold = [0] * 75
oxi.peaks = [0] * 75
oxi.instant_rr = [0] * 75
oxi.recording = list(ppg[:75])
for i in range(len(ppg[75:750])):
    oxi.add_paquet(ppg[75 + i])
oxi.channels["Channel_0"] = np.zeros(750, dtype=int)
oxi.channels["Channel_0"][np.random.choice(np.arange(0, 750), 5)] = 1
oxi.times = list(np.arange(0, 10, 1 / 75))

# Create angular data
x = np.random.normal(np.pi, 0.5, 100)
y = np.random.uniform(0, np.pi * 2, 100)
z = np.concatenate(
    [np.random.normal(np.pi / 2, 0.5, 50), np.random.normal(np.pi + np.pi / 2, 0.5, 50)]
)


class TestPlotting(TestCase):
    def test_plot_raw(self):
        fig, ax = plot_raw(ppg)
        assert isinstance(ax[0], matplotlib.axes.Axes)

    def test_plot_events(self):
        ax = plot_events(oxi)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_oximeter(self):
        ax = plot_oximeter(oxi)
        ax = plot_oximeter(ppg[75:])
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_subspaces(self):
        rr = import_rr().rr
        ax = plot_subspaces(rr)
        assert isinstance(ax[0], matplotlib.axes.Axes)
        assert isinstance(ax[1], matplotlib.axes.Axes)

    def test_plot_psd(self):
        """Test plot_psd function"""
        rr = import_rr().rr.values
        ax = plot_psd(rr)
        assert isinstance(ax, matplotlib.axes.Axes)
        freq, psd = plot_psd(rr, show=False)
        assert freq.mean() == 1.25
        assert psd.mean().round(4) == 0.003

    def test_circular(self):
        """Tests _circular function"""
        ax = circular(list(x))
        assert isinstance(ax, matplotlib.axes.Axes)
        for dens in ["area", "heigth", "alpha"]:
            ax = circular(x, density="alpha", offset=np.pi, ax=None)
            assert isinstance(ax, matplotlib.axes.Axes)
        ax = circular(x, density="height", mean=True, units="degree", color="r")
        assert isinstance(ax, matplotlib.axes.Axes)
        with pytest.raises(ValueError):
            ax = circular(x, density="xx")

    def test_plot_circular(self):
        """Test plot_circular function"""
        data = pd.DataFrame(data={"x": x, "y": y, "z": z}).melt()
        ax = plot_circular(data=data, y="value", hue="variable")
        assert isinstance(ax, matplotlib.axes.Axes)
        ax = plot_circular(data=data, y="value", hue=None)
        assert isinstance(ax, matplotlib.axes.Axes)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
