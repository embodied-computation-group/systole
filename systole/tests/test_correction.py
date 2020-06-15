# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import unittest
from unittest import TestCase
from systole.correction import correct_extra, correct_missed, \
    correct_rr, interpolate_bads, correct_peaks, correct_extra_peaks, \
    correct_missed_peaks
from systole import import_rr
from systole.utils import simulate_rr


class TestDetection(TestCase):

    def test_correct_extra(self):
        """Test oxi_peaks function"""
        rr = import_rr().rr.values  # Import RR time series
        rr[20] = 200
        clean_rr = correct_extra(rr, 20)
        clean_rr = correct_extra(rr, len(rr))
        clean_rr = correct_extra(list(rr), 20)
        assert len(clean_rr) == len(rr)-1
        assert rr.std() > clean_rr.std()

    def test_correct_missed(self):
        """Test correct_missed function"""
        rr = import_rr().rr.values  # Import RR time series
        rr[20] = 1600
        clean_rr = correct_missed(rr, 20)
        clean_rr = correct_missed(list(rr), 20)
        assert len(clean_rr) == len(rr)+1

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
        assert len(correction['clean_rr']) == 352
        assert correction['ectopic'] == 5
        assert correction['missed'] == 1
        assert correction['extra'] == 1
        assert correction['long'] == 1
        assert correction['short'] == 3

    def test_correct_peaks(self):
        """Test correct_peaks function"""
        peaks = simulate_rr(as_peaks=True)
        peaks_correction = correct_peaks(peaks)
        peaks_correction = correct_peaks(list(peaks))
        assert len(peaks_correction['clean_peaks']) == 280154
        assert peaks_correction['missed'] == 1
        assert peaks_correction['extra'] == 1
        assert peaks_correction['ectopic'] == 0
        assert peaks_correction['long'] == 0
        assert peaks_correction['short'] == 0

    def test_correct_extra_peaks(self):
        """Test correct_extra_peaks function"""
        peaks = simulate_rr(as_peaks=True)
        # RR time series to peaks boolean vector
        peaks_correction = correct_extra_peaks(peaks, 20)
        peaks_correction = correct_extra_peaks(list(peaks), 20)
        assert len(peaks_correction) == len(peaks)
        assert sum(peaks_correction) == sum(peaks)-1

    def test_correct_missed_peaks(self):
        """Test correct_missed_peaks function"""
        peaks = simulate_rr(as_peaks=True)
        # RR time series to peaks boolean vector
        peaks_correction = correct_missed_peaks(peaks, 20)
        peaks_correction = correct_missed_peaks(list(peaks), 20)
        assert len(peaks_correction) == len(peaks)
        assert sum(peaks_correction) == sum(peaks)+1


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
