# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
import unittest
from systole.utils import heart_rate
from systole.detection import oxi_peaks
from unittest import TestCase
from systole import import_ppg

ppg = import_ppg('1')[0, :]  # Import PPG recording
signal, peaks = oxi_peaks(ppg)


class TestUtils(TestCase):

    def test_heart_rate(self):
        """Test heart_rate function"""
        heartrate, time = heart_rate(peaks)
        assert len(heartrate) == len(time)
        heartrate, time = heart_rate(
            peaks, unit='bpm', kind='cubic', sfreq=500)
        assert len(heartrate) == len(time)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
