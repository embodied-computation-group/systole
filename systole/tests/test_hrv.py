# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

from systole import import_rr
from systole.hrv import (
    frequency_domain,
    nnX,
    nonlinear_domain,
    pnnX,
    poincare,
    psd,
    recurrence,
    rmssd,
    time_domain,
)

rr = import_rr().rr.values


class TestHrv(TestCase):
    def test_nnX(self):
        """Test nnX function"""
        nn = nnX(list(rr), input_type="rr_ms")
        assert nn == 64
        nn = nnX(rr / 1000, input_type="rr_s")
        assert nn == 64
        with pytest.raises(ValueError):
            nnX(np.array([[1, 1], [1, 1]]))

    def test_pnnX(self):
        """Test pnnX function"""
        pnn = pnnX(list(rr), input_type="rr_ms")
        assert round(pnn, 2) == 26.23
        pnn = pnnX(rr / 1000, input_type="rr_s")
        assert round(pnn, 2) == 26.23
        with pytest.raises(ValueError):
            pnnX(np.array([[1, 1], [1, 1]]))

    def test_rmssd(self):
        """Test rmssd function"""
        rms = rmssd(list(rr))
        assert round(rms, 2) == 45.55
        rms = rmssd(rr / 1000, input_type="rr_s")
        assert round(rms, 2) == 45.55
        with pytest.raises(ValueError):
            rmssd(np.array([[1, 1], [1, 1]]))

    def test_time_domain(self):
        """Test time_domain function"""
        stats = time_domain(list(rr))
        assert isinstance(stats, pd.DataFrame)
        assert stats.size == 26
        with pytest.raises(ValueError):
            time_domain(np.array([[1, 1], [1, 1]]))
        stats = time_domain(rr / 1000, input_type="rr_s")

    def test_psd(self):
        """Test frequency_domain function"""
        freq, pwr = psd(rr=list(rr))
        freq2, pwr2 = psd(rr=rr / 1000, input_type="rr_s")
        assert (freq - freq2).sum() == 0.0
        assert (pwr - pwr2).sum() < 1e-10

    def test_frequency_domain(self):
        """Test frequency_domain function"""
        stats = frequency_domain(rr=list(rr))
        assert isinstance(stats, pd.DataFrame)
        assert stats.size == 22
        stats = frequency_domain(rr=rr / 1000, input_type="rr_s")

    def test_nonlinear_domain(self):
        """Test nonlinear_domain function"""
        stats = nonlinear_domain(list(rr))
        assert isinstance(stats, pd.DataFrame)
        self.assertEqual(stats.size, 14)
        stats = nonlinear_domain(rr / 1000, input_type="rr_s")

    def test_poincare(self):
        """Test poincare function"""
        sd1, sd2 = poincare(list(rr))
        assert np.isclose(sd1, 32.205379429718406)
        assert np.isclose(sd2, 115.10533841926389)

    def test_recurrence(self):
        """Test recurrence function"""
        recurrence_rate, l_max, l_mean, determinism, shan_entr = recurrence(list(rr))
        assert np.isclose(recurrence_rate, 0.42067652973283537)
        assert np.isclose(l_max, 235)
        assert np.isclose(l_mean, 13.069634703196346)
        assert np.isclose(determinism, 0.9772940674349125)
        assert np.isclose(shan_entr, 3.2551017150244004)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
