# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import os
import shutil
import unittest
from unittest import TestCase

import matplotlib.pyplot as plt
import pandas as pd
from bokeh.models import Column

from systole import import_dataset1, import_rr
from systole.hrv import frequency_domain, nonlinear_domain, time_domain
from systole.reports import frequency_table, nonlinear_table, time_table
from systole.reports.group_level import (
    artefacts_group_level,
    frequency_domain_group_level,
    nonlinear_domain_group_level,
    time_domain_group_level,
)
from systole.reports.subject_level import subject_level_report


class TestReports(TestCase):
    def test_subject_level(self):
        """Test the subject-level reports"""

        #######
        # ECG #
        #######
        ecg = import_dataset1(modalities=["ECG"]).ecg.to_numpy()

        subject_level_report(
            participant_id="participant_test",
            pattern="task_test",
            modality="beh",
            result_folder="./",
            session="session_test",
            ecg=ecg,
            ecg_sfreq=1000,
        )

        shutil.rmtree("./participant_test")

    def test_group_level(self):
        """Test the group-level reports"""

        summary_df = pd.read_csv(
            os.path.dirname(__file__) + "/group_level_ses-session1_task-hrd.tsv",
            sep="\t",
        )

        time_domain_group_level(summary_df)
        frequency_domain_group_level(summary_df)
        nonlinear_domain_group_level(summary_df)
        artefacts_group_level(summary_df)

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
        """Test frequency_table function"""
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
        """Test nonlinear_table function"""
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
