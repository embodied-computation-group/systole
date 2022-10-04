# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

from systole import import_ppg
from systole.detectors import mstpd


class TestDetectors(TestCase):
    def test_mstpd(self):
        """Test mstpd function"""
        ppg = import_ppg().ppg.to_numpy()
        peaks = mstpd(signal=ppg, sfreq=75, kind="peaks")
        onsets = mstpd(signal=ppg, sfreq=75, kind="onsets")
        peaks_onsets = mstpd(signal=ppg, sfreq=75, kind="both")
        assert (peaks_onsets[0] == peaks).all()
        assert (peaks_onsets[1] == onsets).all()


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
