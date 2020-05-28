# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import pandas as pd
import numpy as np
import unittest
from unittest import TestCase
from systole.plotly import plot_raw, plot_shortLong, plot_ectopic, \
    plot_subspaces
from systole import import_ppg
from systole.utils import simulate_rr

rr = simulate_rr()
ppg = import_ppg()
signal_df = pd.DataFrame({'time': np.arange(0, len(ppg[0]))/75,
                          'signal': ppg[0]})


class TestInteractive(TestCase):

    def test_plot_raw(self):
        """Test plot_raw function"""
        plot_raw(signal_df)

    def test_plot_shortLong(self):
        """Test plot_shortLong function"""
        plot_shortLong(rr)

    def test_plot_ectopic(self):
        """Test plot_ectopic function"""
        plot_ectopic(rr)

    def test_plot_subspaces(self):
        """Test nnX function"""
        plot_subspaces(rr)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
