# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

from systole import import_ppg
from systole.detectors import msptd


class TestDetectors(TestCase):
    def test_msptd(self):
        """Test msptd function"""
        ppg = import_ppg().ppg.to_numpy()
        peaks = msptd(signal=ppg, sfreq=75, kind="peaks")
        onsets = msptd(signal=ppg, sfreq=75, kind="onsets")
        peaks_onsets = msptd(signal=ppg, sfreq=75, kind="peaks-onsets")
        assert (peaks_onsets[0] == peaks).all()
        assert (peaks_onsets[1] == onsets).all()


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
