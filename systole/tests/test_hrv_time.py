# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import pandas as pd
import unittest
from unittest import TestCase
from systole.hrv_time import nnX, pnnX, rmssd, time_domain
from systole import import_rr

rr = import_rr().rr.values


class TestHrvTime(TestCase):

    def test_nnX(self):
        """Test nnX function"""
        nn = nnX(rr)
        assert nn == 64

    def test_pnnX(self):
        """Test pnnX function"""
        pnn = pnnX(rr)
        assert round(pnn, 2) == 26.23

    def test_rmssd(self):
        """Test rmssd function"""
        rms = rmssd(rr)
        assert round(rms, 2) == 45.55

    def test_time_domain(self):
        """Test time_domain function"""
        stats = time_domain(rr)
        assert isinstance(stats, pd.DataFrame)
        assert stats.size == 24


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
