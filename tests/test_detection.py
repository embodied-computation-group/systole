# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import numpy as np

from systole import import_dataset1, import_ppg
from systole.detection import ppg_peaks, ecg_peaks, rsp_peaks, interpolate_clipping, rr_artefacts


class TestDetection(TestCase):

    def test_ppg_peaks(self):
        """Test ppg_peaks function"""
        ppg = import_ppg().ppg.to_numpy().copy()  # Import PPG recording
        rolling_average_signal, rolling_average_peaks = ppg_peaks(ppg, sfreq=75, method="rolling_average")
        msptd_signal, msptd_peaks = ppg_peaks(ppg, sfreq=75, method="msptd")

        assert np.all(rolling_average_signal == msptd_signal)

        # mean RR intervals
        # Pinning these means to full float precision is brittle: the MSPTD
        # detector shifts the odd peak by a sample or two between SciPy
        # releases. A relative tolerance still catches a real regression, which
        # would move the mean interval by far more than a fraction of a sample.
        assert np.isclose(
            np.diff(np.where(rolling_average_peaks)[0]).mean(), 874.21, rtol=1e-3
        )
        assert np.isclose(
            np.diff(np.where(msptd_peaks)[0]).mean(), 867.30, rtol=1e-3
        )

        # with nan removal and clipping correction
        rolling_average_signal2, rolling_average_peaks2 = ppg_peaks(
            ppg, clipping_thresholds=(0, 255), clean_nan=True, sfreq=75
        )
        assert (rolling_average_signal == rolling_average_signal2).all()
        assert np.isclose(np.diff(np.where(rolling_average_peaks2)[0]).mean(), 874.2068965517242)

    def test_ecg_peaks(self):
        signal_df = import_dataset1(modalities=["ECG"])[: 20 * 2000]
        for method in [
            "sleepecg",
            "christov",
            "engelse-zeelenberg",
            "hamilton",
            "pan-tompkins",
            "moving-average",
        ]:
            _, peaks = ecg_peaks(
                signal_df.ecg, method=method, sfreq=2000, find_local=True
            )
            assert not np.any(peaks > 1)

        with self.assertRaises(ValueError):
            _, peaks = ecg_peaks(
                signal_df.ecg.to_numpy(), method="error", sfreq=2000, find_local=True
            )

    def test_resp_peaks(self):
        """Test resp_peaks function"""
        # Import respiration recording
        resp = import_dataset1(modalities=["Respiration"])[: 200 * 2000].respiration.to_numpy()

        rolling_average_signal, _ = rsp_peaks(
            resp, sfreq=2000, kind="peaks", method="rolling_average"
            )
        msptd_signal, _ = rsp_peaks(
            resp, sfreq=2000, kind="peaks", method="msptd"
            )

        assert np.all(rolling_average_signal == msptd_signal)

        # with nan removal
        rolling_average_signal2, _ = rsp_peaks(
            resp, clean_nan=True, sfreq=2000, kind="peaks", method="rolling_average"
        )
        assert (rolling_average_signal == rolling_average_signal2).all()

    def test_interpolate_clipping_does_not_modify_input(self):
        """Clipping correction must not write into the signal it was given.

        Regression test for #72. The working array was built with `np.asarray`,
        which does not copy an input that is already an array, and the edge
        corrections then assigned into it. That rewrote the caller's signal in
        place, and raised "assignment destination is read-only" outright for
        the read-only arrays pandas returns under copy-on-write.

        The edges have to sit at the clipping threshold for the corrections to
        fire at all, which is why this uses a hand-built signal.
        """
        signal = np.array(
            [255.0, 255.0, 200.0, 150.0, 100.0, 150.0, 200.0, 255.0, 255.0,
             255.0, 200.0, 150.0, 100.0, 150.0, 200.0, 255.0, 255.0]
        )

        # A writable input must come back untouched
        before = signal.copy()
        interpolate_clipping(signal, min_threshold=None, max_threshold=255.0)
        np.testing.assert_array_equal(signal, before)

        # A read-only input must be accepted rather than raising
        read_only = signal.copy()
        read_only.flags.writeable = False
        clean = interpolate_clipping(
            read_only, min_threshold=None, max_threshold=255.0
        )
        assert len(clean) == len(read_only)

    def test_rr_artefacts(self):
        ppg = import_ppg().ppg.to_numpy().copy()
        _, peaks = ppg_peaks(ppg, sfreq=75)
        rr_ms = np.diff(np.where(peaks)[0])
        artefacts_ms = rr_artefacts(rr_ms, input_type="rr_ms")
        artefacts_peaks = rr_artefacts(peaks, input_type="peaks")
        assert all(
            377 == x for x in [len(artefacts_ms[k]) for k in artefacts_ms.keys()]
        )
        assert all(
            377 == x for x in [len(artefacts_peaks[k]) for k in artefacts_peaks.keys()]
        )

    def test_interpolate_clipping(self):
        df = import_ppg()
        clean_signal = interpolate_clipping(df.ppg.to_numpy())
        assert clean_signal.mean().round() == 100
        clean_signal = interpolate_clipping(list(df.ppg.to_numpy()))
        assert clean_signal.mean().round() == 100
        clean_signal = interpolate_clipping(df.ppg.to_numpy())

        # Test with out of bound values as first and last
        df.ppg.iloc[0], df.ppg.iloc[-1] = 255, 255
        clean_signal = interpolate_clipping(df.ppg.to_numpy())
        df.ppg.iloc[0], df.ppg.iloc[-1] = 0, 0
        clean_signal = interpolate_clipping(df.ppg.to_numpy())


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
