# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import unittest
from unittest import TestCase
from systole.correction import correct_extra, correct_missed, \
    correct_artefacts, interpolate_bads
from systole import import_rr


class TestDetection(TestCase):

    def test_correct_extra(self):
        """Test oxi_peaks function"""
        rr = import_rr().rr.values  # Import RR time series
        rr[20] = 200
        clean_rr = correct_extra(rr, 20)
        assert len(clean_rr) == len(rr)-1
        assert rr.std() > clean_rr.std()

    def test_correct_missed(self):
        rr = import_rr().rr.values  # Import RR time series
        rr[20] = 1600
        clean_rr = correct_missed(rr, 20)
        assert len(clean_rr) == len(rr)+1
        assert rr.std() > clean_rr.std()

    def test_interpolate_bads(self):
        rr = import_rr().rr.values  # Import RR time series
        clean_rr = interpolate_bads(rr, [20, 21, 22])
        assert len(rr) == len(rr)
        assert np.abs(rr[20:23] - clean_rr[20:23]).mean() < 50

    def test_correct_artefacts(self):
        rr = import_rr().rr.values  # Import RR time series
        rr[20] = 1600  # Missed
        rr[40] = 100  # Extra
        rr[60] = 1300  # Long
        rr[80] = 500  # Short
        rr[90], rr[91] = 400, 1600  # Extra
        correction = correct_artefacts(rr)
        assert len(correction['clean_rr']) == 245


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
