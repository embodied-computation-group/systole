# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import numpy as np
import pandas as pd

from systole import import_dataset1, import_ppg
from systole.plotly import (
    plot_ectopic,
    plot_frequency,
    plot_nonlinear,
    plot_raw,
    plot_shortLong,
    plot_subspaces,
    plot_timedomain,
)
from systole.utils import simulate_rr

rr = simulate_rr()
ppg = import_ppg().ppg.to_numpy()
signal_df = pd.DataFrame({"time": np.arange(0, len(ppg)) / 75, "ppg": ppg})


class TestInteractive(TestCase):
    def test_plot_raw(self):
        """Test plot_raw function"""
        plot_raw(signal_df)
        plot_raw(signal_df.ppg.to_numpy())
        ecg_df = import_dataset1()[: 20 * 2000]
        plot_raw(ecg_df, type="ecg")
        plot_raw(ecg_df.ecg.to_numpy(), type="ecg", sfreq=2000)

    def test_plot_shortLong(self):
        """Test plot_shortLong function"""
        plot_shortLong(rr)
        with self.assertRaises(ValueError):
            plot_shortLong(rr=None, artefacts=None)

    def test_plot_ectopic(self):
        """Test plot_ectopic function"""
        plot_ectopic(rr)
        with self.assertRaises(ValueError):
            plot_ectopic(rr=None, artefacts=None)

    def test_plot_subspaces(self):
        """Test nnX function"""
        plot_subspaces(rr)

    def test_plot_frequency(self):
        """Test nnX function"""
        plot_frequency(rr)

    def test_plot_nonlinear(self):
        """Test nnX function"""
        plot_nonlinear(rr)

    def test_plot_timedomain(self):
        """Test nnX function"""
        plot_timedomain(rr)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
