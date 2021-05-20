# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import numpy as np
import pandas as pd

from systole import import_ppg, import_rr, serialSim
from systole.recording import Oximeter

from systole.plots import (  # plot_events,; plot_evoked,; plot_timevarying,
    plot_circular,
    plot_ectopic,
    plot_frequency,
    plot_pointcare,
    plot_raw,
    plot_rr,
    plot_shortLong,
    plot_subspaces,
)

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


class TestPlots(TestCase):
    def test_plot_circular(self):
        """Test plot_circular function"""
        data = pd.DataFrame(data={"x": x, "y": y, "z": z}).melt()
        for backend in ["matplotlib", "bokeh"]:
            plot_circular(data=data, y="value", hue="variable", backend=backend)
            plot_circular(data=data, y="value", hue=None, backend=backend)

    def test_plot_ectopic(self):
        """Test plot_ectopic function"""
        rr = import_rr().rr
        for backend in ["matplotlib", "bokeh"]:
            plot_ectopic(rr, backend=backend)

    # def test_plot_events(self):
    #    """Test plot_events function"""
    #    for backend in ["matplotlib", "bokeh"]:
    #        plot_events(oxi, backend=backend)

    # def test_plot_evoked(self):
    #    """Test plot_evoked function"""
    #    for backend in ["matplotlib", "bokeh"]:
    #        plot_evoked(oxi, backend=backend)

    def test_plot_frequency(self):
        """Test plot_frequency function"""
        rr = import_rr().rr
        for backend in ["matplotlib", "bokeh"]:
            plot_frequency(rr, backend=backend, input_type="rr_ms")

    def test_plot_pointcare(self):
        """Test plot_pointcare function"""
        rr = import_rr().rr
        for backend in ["matplotlib", "bokeh"]:
            plot_pointcare(rr, backend=backend, input_type="rr_ms")

    def test_plot_raw(self):
        """Test plot_raw function"""
        for backend in ["matplotlib", "bokeh"]:
            plot_raw(ppg, backend=backend)
            plot_raw(ppg, backend=backend, show_heart_rate=True)

    def test_plot_rr(self):
        """Test plot_rr function"""
        rr = import_rr().rr
        for backend in ["matplotlib", "bokeh"]:
            plot_rr(rr, backend=backend, input_type="rr_ms")

    def test_plot_shortLong(self):
        """Test plot_shortLong function"""
        rr = import_rr().rr
        for backend in ["matplotlib", "bokeh"]:
            plot_shortLong(rr, backend=backend, input_type="rr_ms")

    def test_plot_subspaces(self):
        """Test plot_subspaces function"""
        rr = import_rr().rr
        for backend in ["matplotlib", "bokeh"]:
            plot_subspaces(rr, backend=backend)

    # def test_plot_timevarying(self):
    #    """Test plot_timevarying function"""
    #    rr = import_rr().rr
    #    for backend in ["matplotlib", "bokeh"]:
    #        plot_timevarying(rr, backend=backend)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
