# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
import seaborn as sns

from systole import import_dataset1, import_ppg, import_rr
from systole.detection import ecg_peaks
from systole.plots import (
    plot_circular,
    plot_ectopic,
    plot_events,
    plot_evoked,
    plot_frequency,
    plot_pointcare,
    plot_raw,
    plot_rr,
    plot_shortlong,
    plot_subspaces,
)


class TestPlots(TestCase):
    def test_plot_circular(self):
        """Test plot_circular function"""
        for backend in ["matplotlib"]:

            # Single array as input
            data = np.random.normal(np.pi, 0.5, 100)
            plot_circular(data=data, backend=backend)

            # List of arrays as input
            data = [
                np.random.normal(np.pi, 0.5, 100),
                np.random.uniform(0, np.pi * 2, 100),
            ]
            plot_circular(data=data, hue=None, backend=backend)

            # DataFrame as input
            x = np.random.normal(np.pi, 0.5, 100)
            y = np.random.uniform(0, np.pi * 2, 100)
            data = pd.DataFrame(data={"x": x, "y": y}).melt()
            plot_circular(data=data, y="value", hue="variable", backend=backend)

    def test_plot_ectopic(self):
        """Test plot_ectopic function"""
        rr = import_rr().rr
        for backend in ["matplotlib", "bokeh"]:
            plot_ectopic(rr, backend=backend)

    def test_plot_evoked(self):
        """Test plot_evoked function"""
        # Import ECG recording and Stim channel
        ecg_df = import_dataset1(modalities=["ECG", "Stim"])

        # Peak detection in the ECG signal using the Pan-Tompkins method
        _, peaks = ecg_peaks(ecg_df.ecg, method="pan-tompkins", sfreq=1000)

        triggers_idx = [
            np.where(ecg_df.stim.to_numpy() == 2)[0],
            np.where(ecg_df.stim.to_numpy() == 1)[0],
        ]

        for backend in ["matplotlib", "bokeh"]:
            plot_evoked(
                rr=peaks,
                triggers_idx=triggers_idx,
                input_type="peaks",
                backend=backend,
                palette=[sns.xkcd_rgb["denim blue"], sns.xkcd_rgb["pale red"]],
            )

    def test_plot_events(self):
        """Test plot_events function"""
        # Import ECG recording and Stim channel
        ecg_df = import_dataset1(modalities=["ECG", "Stim"])

        triggers_idx = [
            np.where(ecg_df.stim.to_numpy() == 2)[0],
            np.where(ecg_df.stim.to_numpy() == 1)[0],
        ]

        for backend in ["matplotlib", "bokeh"]:
            plot_events(
                triggers_idx=triggers_idx,
                backend=backend,
                events_labels=["Disgust", "Neutral"],
                tmin=-0.5,
                tmax=10.0,
            )

    def test_plot_frequency(self):
        """Test plot_frequency function"""
        rr = import_rr().rr
        for backend in ["matplotlib", "bokeh"]:
            plot_frequency(rr, backend=backend, input_type="rr_ms")

    def test_plot_pointcare(self):
        """Test plot_pointcare function"""
        rr = import_rr().rr
        for backend in ["matplotlib", "bokeh"]:
            plot_pointcare(rr, backend=backend, input_type="rr_ms")

    def test_plot_raw(self):
        """Test plot_raw function"""
        for backend in ["matplotlib", "bokeh"]:

            # Using ppg signal
            ppg = import_ppg().ppg.to_numpy()
            plot_raw(
                ppg,
                backend=backend,
                show_heart_rate=True,
                show_artefacts=True,
                modality="ppg",
                sfreq=75,
            )

            # Using ecg signal
            ecg_df = import_dataset1(modalities=["ECG"])
            plot_raw(
                ecg_df.ecg,
                backend=backend,
                show_heart_rate=True,
                show_artefacts=True,
                modality="ecg",
                sfreq=1000,
            )

    def test_plot_rr(self):
        """Test plot_rr function"""
        rr = import_rr().rr
        for backend in ["matplotlib", "bokeh"]:
            plot_rr(
                rr,
                backend=backend,
                input_type="rr_ms",
                show_artefacts=True,
                slider=True,
            )
            plot_rr(rr, backend=backend, input_type="rr_ms", points=False)
            plot_rr(rr, backend=backend, input_type="rr_ms", line=False)

    def test_plot_shortlong(self):
        """Test plot_shortlong function"""
        rr = import_rr().rr
        for backend in ["matplotlib", "bokeh"]:
            plot_shortlong(rr, backend=backend, input_type="rr_ms")

    def test_plot_subspaces(self):
        """Test plot_subspaces function"""
        rr = import_rr().rr
        for backend in ["matplotlib", "bokeh"]:
            plot_subspaces(rr, backend=backend)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
