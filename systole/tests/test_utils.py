# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import pytest
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
        peaks[-1:] = 1
        y = norm_triggers(peaks)
        assert sum(y) == 378
        peaks = - peaks.astype(int)
        y = norm_triggers(peaks, threshold=-1, direction='lower')
        assert sum(y) == 378
        with pytest.raises(ValueError):
            norm_triggers(None)
            norm_triggers(peaks, threshold=-1, direction='invalid')

    def test_heart_rate(self):
        """Test heart_rate function"""
        ppg = import_ppg('1')[0, :]  # Import PPG recording
        signal, peaks = oxi_peaks(ppg)
        heartrate, time = heart_rate(peaks)
        assert len(heartrate) == len(time)
        heartrate, time = heart_rate(list(peaks))
        assert len(heartrate) == len(time)
        heartrate, time = heart_rate(
            peaks, unit='bpm', kind='cubic', sfreq=500)
        assert len(heartrate) == len(time)
        with pytest.raises(ValueError):
            heartrate, time = heart_rate([1, 2, 3])

    def test_time_shift(self):
        """Test time_shift function"""
        lag = time_shift([40, 50, 60], [45, 52])
        assert lag == [5, 2]

    def test_to_angle(self):
        """Test to_angles function"""
        rr = import_rr().rr.values
        # Create event vector
        events = rr + np.random.normal(500, 100, len(rr))
        ang = to_angles(list(np.cumsum(rr)), list(np.cumsum(events)))
        assert ~np.any(np.asarray(ang) < 0)
        assert ~np.any(np.asarray(ang) > np.pi * 2)
        ppg = import_ppg('1')[0, :]  # Import PPG recording
        signal, peaks = oxi_peaks(ppg)
        ang = to_angles(peaks, peaks)

    def test_to_epochs(self):
        """Test oxi_peaks function"""
        ppg = import_ppg('1')[0, :]  # Import PPG recording
        events = import_ppg('1')[1, :]  # Import events
        events[2] = 1
        epochs = to_epochs(ppg, events, sfreq=75, verbose=True,
                           apply_baseline=(-1, 0))
        assert epochs.ndim == 2
        epochs = to_epochs(ppg, events, sfreq=75, apply_baseline=None)
        epochs = to_epochs(ppg, events, sfreq=75, apply_baseline=-1)
        with pytest.raises(ValueError):
            epochs = to_epochs(ppg[1:], events, sfreq=75)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
