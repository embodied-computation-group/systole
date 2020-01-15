# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import unittest
from unittest import TestCase
from systole.detection import oxi_peaks, artefact_correction, missed_beat,\
    hrv_subspaces, rr_outliers, interpolate_clipping
from systole import import_ppg, import_rr


class TestDetection(TestCase):

    def test_oxi_peaks(self):
        """Test oxi_peaks function"""
        ppg = import_ppg('1')[0, :]  # Import PPG recording
        signal, peaks = oxi_peaks(ppg)
        assert len(signal) == len(peaks)
        assert np.all(np.unique(peaks) == [0, 1])

    def test_hrv_subspaces(self):
        rr = import_rr().rr  # Import PPG recording
        s1, s2, s3 = hrv_subspaces(rr)
        assert round(np.mean(s1), 3) == -0.004
        assert round(np.mean(s2), 3) == 0.076
        assert round(np.mean(s3), 3) == -0.082

    def test_interpolate_clipping(self):
        ppg = import_ppg('1')[0]
        clean_signal = interpolate_clipping(ppg)
        assert clean_signal.mean().round() == 100
        clean_signal = interpolate_clipping(list(ppg))
        assert clean_signal.mean().round() == 100

    def test_rr_outliers(self):
        rr = import_rr().rr  # Import PPG recording
        ectobeats, outliers = rr_outliers(rr)
        assert np.where(ectobeats)[0][0] == 122
        assert np.where(outliers)[0][0] == 120


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
