# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import pandas as pd
import matplotlib
import unittest
from unittest import TestCase
from systole.hrv_frequency import hrv_psd, frequency_domain
from systole import import_rr

rr = import_rr().rr.values


class TestHrvFrequency(TestCase):

    def test_hrv_psd(self):
        """Test hrv_psd function"""
        ax = hrv_psd(rr)
        assert isinstance(ax, matplotlib.axes.Axes)
        freq, psd = hrv_psd(rr, show=False)
        assert len(freq) == len(psd)

    def test_frequency_domain(self):
        """Test frequency_domain function"""
        stats = frequency_domain(rr)
        assert isinstance(stats, pd.DataFrame)
        assert stats.size == 22


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
