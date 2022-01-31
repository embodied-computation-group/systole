# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bokeh.models import Column

from systole import import_dataset1, import_ppg, import_rr
from systole.detection import ecg_peaks, rr_artefacts
from systole.hrv import frequency_domain, nonlinear_domain, time_domain
from systole.plots import (
    plot_circular,
    plot_ectopic,
    plot_events,
    plot_evoked,
    plot_frequency,
    plot_poincare,
    plot_raw,
    plot_rr,
    plot_rsp,
    plot_shortlong,
    plot_subspaces,
)
from systole.plots.utils import frequency_table, nonlinear_table, time_table
from systole.utils import heart_rate, to_epochs


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

        plt.close("all")

    def test_plot_ectopic(self):
        """Test plot_ectopic function"""
        rr = import_rr().rr
        for backend in ["matplotlib", "bokeh"]:
            plot_ectopic(rr, backend=backend)

        plt.close("all")

    def test_plot_evoked(self):
        """Test plot_evoked function"""
        # Import ECG recording and Stim channel
        ecg_df = import_dataset1(modalities=["ECG", "Stim"])

        # Peak detection in the ECG signal using the Pan-Tompkins method
        _, peaks = ecg_peaks(ecg_df.ecg, method="pan-tompkins", sfreq=1000)

        # Triggers timimng
        triggers_idx = [
            np.where(ecg_df.stim.to_numpy() == 2)[0],
            np.where(ecg_df.stim.to_numpy() == 1)[0],
        ]

        # Epochs array
        rr, _ = heart_rate(peaks, kind="cubic", unit="bpm", input_type="peaks")
        epochs, _ = to_epochs(
            signal=rr,
            triggers_idx=triggers_idx,
            tmin=-1.0,
            tmax=10.0,
            apply_baseline=(-1.0, 0.0),
        )

        for backend in ["matplotlib", "bokeh"]:

            # Using raw ECG signal as input
            plot_evoked(
                signal=ecg_df.ecg.to_numpy(),
                triggers_idx=triggers_idx,
                modality="ecg",
                tmin=-1.0,
                tmax=10.0,
                apply_baseline=(-1.0, 0.0),
                backend=backend,
                palette=[sns.xkcd_rgb["denim blue"], sns.xkcd_rgb["pale red"]],
            )

            # Using instantaneous heart rate as input
            plot_evoked(
                rr=peaks,
                triggers_idx=triggers_idx,
                input_type="peaks",
                tmin=-1.0,
                tmax=10.0,
                apply_baseline=(-1.0, 0.0),
                backend=backend,
                palette=[sns.xkcd_rgb["denim blue"], sns.xkcd_rgb["pale red"]],
            )

            # Using evoked array as input
            plot_evoked(
                epochs=epochs,
                backend=backend,
                palette=[sns.xkcd_rgb["denim blue"], sns.xkcd_rgb["pale red"]],
            )

        plt.close("all")

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

        plt.close("all")

    def test_plot_frequency(self):
        """Test plot_frequency function"""
        rr = import_rr().rr
        for backend in ["matplotlib", "bokeh"]:
            plot_frequency(rr, backend=backend, input_type="rr_ms")

        plt.close("all")

    def test_plot_poincare(self):
        """Test plot_poincare function"""
        rr = import_rr().rr
        for backend in ["matplotlib", "bokeh"]:
            plot_poincare(rr, backend=backend, input_type="rr_ms")

        plt.close("all")

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

        plt.close("all")

    def test_plot_rsp(self):
        """Test plot_rsp function"""
        for backend in ["matplotlib", "bokeh"]:

            rsp = import_dataset1(modalities=["Respiration"])
            plot_rsp(
                rsp.respiration,
                backend=backend,
                sfreq=1000,
            )

        plt.close("all")

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

        plt.close("all")

    def test_plot_shortlong(self):
        """Test plot_shortlong function"""
        rr = import_rr().rr
        for backend in ["matplotlib", "bokeh"]:
            plot_shortlong(rr, backend=backend, input_type="rr_ms")

        plt.close("all")

    def test_plot_subspaces(self):
        """Test plot_subspaces function"""
        rr = import_rr().rr
        artefacts = rr_artefacts(rr)
        for backend in ["matplotlib", "bokeh"]:
            plot_subspaces(rr=rr, backend=backend)
            plot_subspaces(artefacts=artefacts)

        with self.assertRaises(ValueError):
            plot_subspaces(rr=rr, artefacts=artefacts)

        with self.assertRaises(ValueError):
            plot_subspaces(rr=None, artefacts=None)

        plt.close("all")

    def test_time_table(self):
        """Test the time_table function"""
        rr = import_rr().rr
        time_df = time_domain(rr, input_type="rr_ms")

        # With a df as input
        table_df = time_table(time_df=time_df, backend="tabulate")
        assert isinstance(table_df, str)

        table = time_table(time_df=time_df, backend="bokeh")
        assert isinstance(table, Column)

        # With RR intervals as inputs
        table_rr = time_table(rr=rr, backend="tabulate")
        assert isinstance(table_rr, str)

        table = time_table(rr=rr, backend="bokeh")
        assert isinstance(table, Column)

        # Check for consistency between methods
        assert table_rr == table_df

        plt.close("all")

    def test_frequency_table(self):
        """Test plot_subspaces function"""
        rr = import_rr().rr
        frequency_df = frequency_domain(rr, input_type="rr_ms")

        # With a df as input
        table_df = frequency_table(frequency_df=frequency_df, backend="tabulate")
        assert isinstance(table_df, str)

        table = frequency_table(frequency_df=frequency_df, backend="bokeh")
        assert isinstance(table, Column)

        # With RR intervals as inputs
        table_rr = frequency_table(rr=rr, backend="tabulate")
        assert isinstance(table_rr, str)

        table = frequency_table(rr=rr, backend="bokeh")
        assert isinstance(table, Column)

        # Check for consistency between methods
        assert table_rr == table_df

        plt.close("all")

    def test_nonlinear_table(self):
        """Test plot_subspaces function"""
        rr = import_rr().rr
        nonlinear_df = nonlinear_domain(rr, input_type="rr_ms")

        # With a df as input
        table_df = nonlinear_table(nonlinear_df=nonlinear_df, backend="tabulate")
        assert isinstance(table_df, str)

        table = nonlinear_table(nonlinear_df=nonlinear_df, backend="bokeh")
        assert isinstance(table, Column)

        # With RR intervals as inputs
        table_rr = nonlinear_table(rr=rr, backend="tabulate")
        assert isinstance(table_rr, str)

        table = nonlinear_table(rr=rr, backend="bokeh")
        assert isinstance(table, Column)

        # Check for consistency between methods
        assert table_rr == table_df

        plt.close("all")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
