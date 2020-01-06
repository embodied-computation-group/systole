# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase
from systole.epochs import to_epochs
from systole import import_ppg


class TestEpochs(TestCase):

    def test_to_epochs(self):
        """Test oxi_peaks function"""
        ppg = import_ppg('1')[0, :]  # Import PPG recording
        events = import_ppg('1')[1, :]  # Import events
        epochs = to_epochs(ppg, events, sfreq=75)
        assert epochs.ndim == 2


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
