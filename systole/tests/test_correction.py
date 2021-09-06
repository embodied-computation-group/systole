# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import numpy as np

from systole import import_rr
from systole.correction import (
    correct_extra,
    correct_missed,
    correct_missed_peaks,
    correct_peaks,
    correct_rr,
    interpolate_bads,
)
from systole.utils import simulate_rr


class TestDetection(TestCase):
    def test_correct_extra(self):
        """Test ppg_peaks function"""
        rr = import_rr().rr.values  # Import RR time series
        rr[20] = 200
        clean_rr = correct_extra(rr, 20)
        clean_rr = correct_extra(rr, len(rr))
        clean_rr = correct_extra(list(rr), 20)
        assert len(clean_rr) == len(rr) - 1
        assert rr.std() > clean_rr.std()

    def test_correct_missed(self):
        """Test correct_missed function"""
        rr = import_rr().rr.values  # Import RR time series
        rr[20] = 1600
        clean_rr = correct_missed(rr, 20)
        clean_rr = correct_missed(list(rr), 20)
        assert len(clean_rr) == len(rr) + 1

    def test_interpolate_bads(self):
        """Test interpolate_bads function"""
        rr = import_rr().rr.values  # Import RR time series
        clean_rr = interpolate_bads(rr, [20, 21, 22])
        clean_rr = interpolate_bads(list(rr), [20, 21, 22])
        assert len(rr) == len(rr)
        assert np.abs(rr[20:23] - clean_rr[20:23]).mean() < 50

    def test_correct_rr(self):
        """Test correct_rr function"""
        rr = simulate_rr()  # Import RR time series
        correction = correct_rr(rr)
        correction = correct_rr(list(rr))
        assert len(correction["clean_rr"]) == 350
        assert correction["ectopic"] == 5
        assert correction["missed"] == 1
        assert correction["extra"] == 1
        assert correction["long"] == 1
        assert correction["short"] == 2

    def test_correct_peaks(self):
        """Test correct_peaks function"""
        peaks = simulate_rr(as_peaks=True)
        peaks_correction = correct_peaks(peaks)
        peaks_correction = correct_peaks(list(peaks))
        assert len(peaks_correction["clean_peaks"]) == 280154
        assert peaks_correction["missed"] == 1
        assert peaks_correction["extra"] == 1

    def test_correct_missed_peaks(self):
        """Test correct_missed_peaks function"""
        np.random.seed(123)
        rr = np.random.normal(1000, 200, 10).astype("int")
        peaks = np.zeros(10000)
        peaks[np.cumsum(rr)] = 1
        assert np.where(peaks)[0].sum() == 52029

        peaks[3735] = 0
        peaks = correct_missed_peaks(peaks, idx=4619)
        assert np.where(peaks)[0].sum() == 52122

        with self.assertRaises(ValueError):
            correct_missed_peaks(peaks, idx=4610)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
