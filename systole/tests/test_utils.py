# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
import numpy as np
from systole.utils import norm_triggers, heart_rate, to_angles, to_epochs,\
        time_shift
from systole.detection import oxi_peaks
from unittest import TestCase
from systole import import_ppg, import_rr


class TestUtils(TestCase):

    def test_norm_triggers(self):
        ppg = import_ppg('1')[0, :]  # Import PPG recording
        signal, peaks = oxi_peaks(ppg)
        peaks[np.where(peaks)[0]+1] = 1
        peaks[np.where(peaks)[0]+2] = 1
        y = norm_triggers(peaks)
        assert sum(y) == 378
        peaks = -peaks
        y = norm_triggers(peaks, threshold=-1, direction='lower')
        assert sum(y) == 378

    def test_heart_rate(self):
        """Test heart_rate function"""
        ppg = import_ppg('1')[0, :]  # Import PPG recording
        signal, peaks = oxi_peaks(ppg)
        heartrate, time = heart_rate(peaks)
        assert len(heartrate) == len(time)
        heartrate, time = heart_rate(
            peaks, unit='bpm', kind='cubic', sfreq=500)
        assert len(heartrate) == len(time)

    def test_time_shift(self):
        """Test time_shift function"""
        lag = time_shift([40, 50, 60], [45, 52])
        assert lag == [5, 2]

    def test_to_angle(self):
        """Test to_angles function"""
        rr = import_rr().rr.values
        # Create event vector
        events = rr + np.random.normal(500, 100, len(rr))
        ang = to_angles(np.cumsum(rr), np.cumsum(events))
        assert ~np.any(np.asarray(ang) < 0)
        assert ~np.any(np.asarray(ang) > np.pi * 2)

    def test_to_epochs(self):
        """Test oxi_peaks function"""
        ppg = import_ppg('1')[0, :]  # Import PPG recording
        events = import_ppg('1')[1, :]  # Import events
        epochs = to_epochs(ppg, events, sfreq=75)
        assert epochs.ndim == 2


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
