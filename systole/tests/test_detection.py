# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import unittest
from unittest import TestCase
from systole.detection import oxi_peaks, rr_artefacts, interpolate_clipping, ecg_peaks
from systole import import_ppg, import_dataset1
from systole.utils import simulate_rr


class TestDetection(TestCase):
    def test_oxi_peaks(self):
        """Test oxi_peaks function"""
        df = import_ppg()  # Import PPG recording
        signal, peaks = oxi_peaks(df.ppg.to_numpy(), clean_extra=True)
        assert len(signal) == len(peaks)
        assert np.all(np.unique(peaks) == [0, 1])
        assert np.mean(np.where(peaks)[0]) == 165778.0

    def test_rr_artefacts(self):
        rr = simulate_rr()  # Simulate RR time series
        artefacts = rr_artefacts(rr)
        artefacts = rr_artefacts(list(rr))
        assert all(350 == x for x in [len(artefacts[k]) for k in artefacts.keys()])

    def test_interpolate_clipping(self):
        df = import_ppg()
        clean_signal = interpolate_clipping(df.ppg.to_numpy())
        assert clean_signal.mean().round() == 100
        clean_signal = interpolate_clipping(list(df.ppg.to_numpy()))
        assert clean_signal.mean().round() == 100
        clean_signal = interpolate_clipping(df.ppg.to_numpy())
        df.ppg.iloc[0], df.ppg.iloc[-1] = 255, 255
        clean_signal = interpolate_clipping(df.ppg.to_numpy())

    def test_ecg_peaks(self):
        signal_df = import_dataset1()[: 20 * 2000]
        signal, peaks = ecg_peaks(
            signal_df.ecg.to_numpy(), method="hamilton", sfreq=2000, find_local=True
        )
        for method in [
            "christov",
            "engelse-zeelenberg",
            "pan-tompkins",
            "wavelet-transform",
            "moving-average",
        ]:
            signal, peaks = ecg_peaks(
                signal_df.ecg, method=method, sfreq=2000, find_local=True
            )
            assert not np.any(peaks > 1)

        with self.assertRaises(ValueError):
            signal, peaks = ecg_peaks(
                signal_df.ecg.to_numpy(), method="error", sfreq=2000, find_local=True
            )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
