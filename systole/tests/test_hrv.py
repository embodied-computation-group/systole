# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

from systole import import_rr
from systole.hrv import (
    all_domain,
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

        # Kubios 2.2: 883.24
        assert np.isclose(stats[stats.Metric == "MeanRR"].Values, 883.238095)

        # Kubios 2.2: 69578
        assert np.isclose(stats[stats.Metric == "MeanBPM"].Values, 68.577834)

        # Kubios 2.2: 84.691
        assert np.isclose(stats[stats.Metric == "SDNN"].Values, 84.690544)

        # Kubios 2.2: 64
        assert np.isclose(stats[stats.Metric == "nn50"].Values, 64.000000)

        # Kubios 2.2: 26.230
        assert np.isclose(stats[stats.Metric == "pnn50"].Values, 26.229508)

        # Kubios 2.2: 45.545
        assert np.isclose(stats[stats.Metric == "RMSSD"].Values, 45.546669)

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

        # Kubios 2.2: 0.031250
        assert np.isclose(stats[stats.Metric == "vlf_peak"].Values, 0.032468)

        # Kubios 2.2: 4504.4
        assert np.isclose(stats[stats.Metric == "vlf_power"].Values, 4063.165990)

        # Kubios 2.2: 60.341
        assert np.isclose(stats[stats.Metric == "vlf_power_per"].Values, 58.424644)

        # Kubios 2.2: 0.066406
        assert np.isclose(stats[stats.Metric == "lf_peak"].Values, 0.064935)

        # Kubios 2.2: 2393.9
        assert np.isclose(stats[stats.Metric == "lf_power"].Values, 2339.544469)

        # Kubios 2.2: 32.068
        assert np.isclose(stats[stats.Metric == "lf_power_per"].Values, 33.640529)

        # Kubios 2.2: 80.860
        assert np.isclose(stats[stats.Metric == "lf_power_nu"].Values, 80.914590)

        # Kubios 2.2: 0.31250
        assert np.isclose(stats[stats.Metric == "hf_peak"].Values, 0.310761)

        # Kubios 2.2: 566.30
        assert np.isclose(stats[stats.Metric == "hf_power"].Values, 551.830831)

        # Kubios 2.2: 7.5861
        assert np.isclose(stats[stats.Metric == "hf_power_per"].Values, 7.934827)

        # Kubios 2.2: 19.129
        assert np.isclose(stats[stats.Metric == "hf_power_nu"].Values, 19.085410)

        # Kubios 2.2: 7465
        assert np.isclose(stats[stats.Metric == "total_power"].Values, 6954.541289)

        # Kubios 2.2: 4.227
        assert np.isclose(stats[stats.Metric == "lf_hf_ratio"].Values, 4.239604)

        assert isinstance(stats, pd.DataFrame)
        assert stats.size == 26
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

        # Kubios 2.2: 32.273
        assert np.isclose(sd1, 32.205379429718406)

        # Kubios 2.2: 115.38
        assert np.isclose(sd2, 115.10533841926389)

    def test_recurrence(self):
        """Test recurrence function"""
        recurrence_rate, l_max, l_mean, determinism, shan_entr = recurrence(list(rr))

        # Kubios 2.2: 37.231
        assert np.isclose(recurrence_rate, 38.08510638297872)

        # Kubios 2.2: 235
        assert np.isclose(l_max, 235)

        # Kubios 2.2: 11.501
        assert np.isclose(l_mean, 12.69109947643979)

        # Kubios 2.2: 99.055
        assert np.isclose(determinism, 70.43099273607749)

        # Kubios 2.2: 3.2120
        assert np.isclose(shan_entr, 3.1961134772254334)

    def test_all_domain(self):
        """Test all_domain function"""
        all_df = all_domain(list(rr))

        metrics = [
            "MeanRR",
            "MeanBPM",
            "MedianRR",
            "MedianBPM",
            "MinRR",
            "MinBPM",
            "MaxRR",
            "MaxBPM",
            "SDNN",
            "SDSD",
            "RMSSD",
            "nn50",
            "pnn50",
            "vlf_peak",
            "vlf_power",
            "lf_peak",
            "lf_power",
            "hf_peak",
            "hf_power",
            "vlf_power_per",
            "lf_power_per",
            "hf_power_per",
            "lf_power_nu",
            "hf_power_nu",
            "total_power",
            "lf_hf_ratio",
            "SD1",
            "SD2",
            "recurrence_rate",
            "l_max",
            "l_mean",
            "determinism_rate",
            "shannon_entropy",
        ]

        assert np.all([metric in metrics for metric in all_df.Metric.unique()])


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
