# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import pandas as pd
import unittest
from unittest import TestCase
from systole.hrv import nnX, pnnX, rmssd, time_domain,\
    frequency_domain, nonlinear
from systole import import_rr

rr = import_rr().rr.values


class TestHrv(TestCase):

    def test_nnX(self):
        """Test nnX function"""
        nn = nnX(list(rr))
        assert nn == 64

    def test_pnnX(self):
        """Test pnnX function"""
        pnn = pnnX(list(rr))
        assert round(pnn, 2) == 26.23

    def test_rmssd(self):
        """Test rmssd function"""
        rms = rmssd(list(rr))
        assert round(rms, 2) == 45.55

    def test_time_domain(self):
        """Test time_domain function"""
        stats = time_domain(list(rr))
        assert isinstance(stats, pd.DataFrame)
        assert stats.size == 24

    def test_frequency_domain(self):
        """Test frequency_domain function"""
        stats = frequency_domain(list(rr))
        assert isinstance(stats, pd.DataFrame)
        assert stats.size == 22

    def test_nonlinear(self):
        """Test nonlinear_domain function"""
        stats = nonlinear(list(rr))
        assert isinstance(stats, pd.DataFrame)
        self.assertEqual(stats.size, 4)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
