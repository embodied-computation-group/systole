# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import unittest
from unittest import TestCase
from systole.detection import oxi_peaks, rr_artefacts, interpolate_clipping,\
    ecg_peaks
from systole import import_ppg, import_dataset
from systole.utils import simulate_rr


class TestDetection(TestCase):

    def test_oxi_peaks(self):
        """Test oxi_peaks function"""
        ppg = import_ppg('1')[0, :]  # Import PPG recording
        signal, peaks = oxi_peaks(ppg)
        assert len(signal) == len(peaks)
        assert np.all(np.unique(peaks) == [0, 1])

    def test_rr_artefacts(self):
        rr = simulate_rr()  # Import PPG recording
        artefacts = rr_artefacts(rr)
        artefacts = rr_artefacts(list(rr))
        assert all(
            350 == x for x in [len(artefacts[k]) for k in artefacts.keys()])

    def test_interpolate_clipping(self):
        ppg = import_ppg('1')[0]
        clean_signal = interpolate_clipping(ppg)
        assert clean_signal.mean().round() == 100
        clean_signal = interpolate_clipping(list(ppg))
        assert clean_signal.mean().round() == 100
        clean_signal = interpolate_clipping(ppg)
        ppg[0], ppg[-1] = 255, 255
        clean_signal = interpolate_clipping(ppg)

    def test_ecg_peaks(self):
        signal_df = import_dataset()[:20*2000]
        signal, peaks = ecg_peaks(signal_df.ecg.to_numpy(), method='hamilton',
                                  sfreq=2000, find_local=True)
        for method in ['christov', 'engelse-zeelenberg', 'pan-tompkins',
                       'wavelet-transform', 'moving-average']:
            signal, peaks = ecg_peaks(signal_df.ecg, method=method, sfreq=2000,
                                      find_local=True)
            assert not np.any(peaks > 1)

        with self.assertRaises(ValueError):
            signal, peaks = ecg_peaks(signal_df.ecg.to_numpy(), method='error',
                                      sfreq=2000, find_local=True)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
