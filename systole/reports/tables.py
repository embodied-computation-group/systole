# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from bokeh.models import Column, ColumnDataSource, DataTable, TableColumn
from tabulate import tabulate  # type: ignore

from systole.hrv import frequency_domain, nonlinear_domain, time_domain


def time_table(
    rr: Optional[Union[List, np.ndarray]] = None,
    time_df: Optional[pd.DataFrame] = None,
    input_type: str = "rr_ms",
    backend: str = "tabulate",
    width: Optional[int] = 600,
    height: Optional[int] = 300,
) -> Union[Column, str]:
    """Format time domain results for nice rendering.

    Parameters
    ----------
    rr :
        R-R interval time-series, peaks or peaks index vectors. The default expected
        vector is R-R intervals in milliseconds. Other data format can be provided by
        specifying the `"input_type"` (can be `"rr_s"`, `"peaks"` or `"peaks_idx"`).
    time_df :
        The time domain results obtained from:py:func:`systole.hrv.time_domain`.
    input_type :
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.
    bakend :
        Which backend to use. Can be `"bokeh"` or `"tabulate"`. Defaults to `"tabulate"`.
    width, height :
        The table width and height (only for `"bokeh"` backend).

    Returns
    -------
    table :
        The formatted time domain table, either as a string or as a Bokeh Columns.

    """

    if time_df is None:
        if (rr is None) | (input_type is None):
            raise ValueError(
                "If no summary dataframe is provided, you must provide an RR time series."
            )
        time_df = time_domain(rr=rr, input_type=input_type)  # type: ignore

    data = {
        "variable": [
            "Mean RR",
            "Median RR",
            "Minimun RR",
            "Maximum RR",
            None,
            "Mean BPM",
            "Median BPM",
            "Minimun BPM",
            "Maximum BPM",
            None,
            "SDNN",
            "SDSD",
            "RMSSD",
            "nn50",
            "pnn50",
        ],
        "units": [
            "milliseconds (ms)",
            "milliseconds (ms)",
            "milliseconds (ms)",
            "milliseconds (ms)",
            None,
            "beats per minute (bpm)",
            "beats per minute (bpm)",
            "beats per minute (bpm)",
            "beats per minute (bpm)",
            None,
            "milliseconds (ms)",
            "milliseconds (ms)",
            "milliseconds (ms)",
            "counts",
            "%",
        ],
        "value": [
            time_df[time_df.Metric == "MeanRR"]["Values"].iloc[0],
            time_df[time_df.Metric == "MedianRR"]["Values"].iloc[0],
            time_df[time_df.Metric == "MinRR"]["Values"].iloc[0],
            time_df[time_df.Metric == "MaxRR"]["Values"].iloc[0],
            None,
            time_df[time_df.Metric == "MeanBPM"]["Values"].iloc[0],
            time_df[time_df.Metric == "MedianBPM"]["Values"].iloc[0],
            time_df[time_df.Metric == "MinBPM"]["Values"].iloc[0],
            time_df[time_df.Metric == "MaxBPM"]["Values"].iloc[0],
            None,
            time_df[time_df.Metric == "SDNN"]["Values"].iloc[0],
            time_df[time_df.Metric == "SDSD"]["Values"].iloc[0],
            time_df[time_df.Metric == "RMSSD"]["Values"].iloc[0],
            time_df[time_df.Metric == "nn50"]["Values"].iloc[0],
            time_df[time_df.Metric == "pnn50"]["Values"].iloc[0],
        ],
    }

    if backend == "bokeh":
        source = ColumnDataSource(data)

        columns = [
            TableColumn(field="variable", title="Variable"),
            TableColumn(field="units", title="Units"),
            TableColumn(field="value", title="Value"),
        ]
        time_table = DataTable(
            source=source,
            columns=columns,
            width=width,
            height=height,
            index_position=None,
        )

        table = Column(time_table)

    elif backend == "tabulate":
        table = tabulate(
            data, headers=["Variable", "Units", "Value"], tablefmt="rst", floatfmt=".4f"
        )

    return table


def frequency_table(
    rr: Optional[Union[List, np.ndarray]] = None,
    frequency_df: Optional[pd.DataFrame] = None,
    input_type: str = "rr_ms",
    backend: str = "tabulate",
    width: Optional[int] = 600,
    height: Optional[int] = 300,
) -> Union[Column, str]:
    """Format frequency domain results for nice rendering.

    Parameters
    ----------
    rr :
        R-R interval time-series, peaks or peaks index vectors. The default expected
        vector is R-R intervals in milliseconds. Other data format can be provided by
        specifying the `"input_type"` (can be `"rr_s"`, `"peaks"` or `"peaks_idx"`).
    frequency_df :
        The frequency domain results obtained from:py:func:`systole.hrv.frequency_domain`.
    input_type :
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.
    bakend :
        Which backend to use. Can be `"bokeh"` or `"tabulate"`. Defaults to `"tabulate"`.
    width, height :
        The table width and height (only for `"bokeh"` backend).

    Returns
    -------
    table :
        The formatted frequency domain table, either as a string or as a Bokeh Columns.

    """

    if frequency_df is None:
        if (rr is None) | (input_type is None):
            raise ValueError(
                "If no summary dataframe is provided, you must provide an RR time series."
            )
        frequency_df = frequency_domain(rr=rr, input_type=input_type)  # type: ignore

    data = {
        "frequencies": [
            "VLF (0-0.04 Hz)",
            "LF (0.04-0.15 Hz)",
            "HF (0.15-0.4 Hz)",
            "Total",
            "LF/HF",
        ],
        "peaks": [
            frequency_df[frequency_df.Metric == "vlf_peak"]["Values"].iloc[0],
            frequency_df[frequency_df.Metric == "lf_peak"]["Values"].iloc[0],
            frequency_df[frequency_df.Metric == "hf_peak"]["Values"].iloc[0],
            None,
            None,
        ],
        "power": [
            frequency_df[frequency_df.Metric == "vlf_power"]["Values"].iloc[0],
            frequency_df[frequency_df.Metric == "lf_power"]["Values"].iloc[0],
            frequency_df[frequency_df.Metric == "hf_power"]["Values"].iloc[0],
            frequency_df[frequency_df.Metric == "total_power"]["Values"].iloc[0],
            frequency_df[frequency_df.Metric == "lf_hf_ratio"]["Values"].iloc[0],
        ],
        "power_per": [
            frequency_df[frequency_df.Metric == "vlf_power_per"]["Values"].iloc[0],
            frequency_df[frequency_df.Metric == "lf_power_per"]["Values"].iloc[0],
            frequency_df[frequency_df.Metric == "hf_power_per"]["Values"].iloc[0],
            None,
            None,
        ],
        "power_nu": [
            None,
            frequency_df[frequency_df.Metric == "lf_power_nu"]["Values"].iloc[0],
            frequency_df[frequency_df.Metric == "hf_power_nu"]["Values"].iloc[0],
            None,
            None,
        ],
    }

    if backend == "bokeh":
        source = ColumnDataSource(data)

        columns = [
            TableColumn(field="frequencies", title="Frequency Band"),
            TableColumn(field="peaks", title="Peaks (Hz)"),
            TableColumn(field="power", title="Power (ms²)"),
            TableColumn(field="power_per", title="Power (%)"),
            TableColumn(field="power_nu", title="Power (n.u.)"),
        ]
        frequency_table = DataTable(
            source=source,
            columns=columns,
            width=width,
            height=height,
            index_position=None,
        )

        table = Column(frequency_table)

    elif backend == "tabulate":
        table = tabulate(
            data,
            headers=[
                "Frequency Band",
                "Peaks (Hz)",
                "Power (ms²)",
                "Power (%)",
                "Power (n.u.)",
            ],
            tablefmt="rst",
            floatfmt=".4f",
        )

    return table


def nonlinear_table(
    rr: Optional[Union[List, np.ndarray]] = None,
    nonlinear_df: Optional[pd.DataFrame] = None,
    input_type="rr_ms",
    backend: str = "tabulate",
    width: Optional[int] = 600,
    height: Optional[int] = 300,
) -> Union[Column, str]:
    """Format nonlinear domain results for nice rendering.

    Parameters
    ----------
    rr :
        R-R interval time-series, peaks or peaks index vectors. The default expected
        vector is R-R intervals in milliseconds. Other data format can be provided by
        specifying the `"input_type"` (can be `"rr_s"`, `"peaks"` or `"peaks_idx"`).
    nonlinear_df :
        The time domain results obtained from:py:func:`systole.hrv.nonlinear_domain`.
    input_type :
        The type of input provided. Can be `"peaks"`, `"peaks_idx"`, `"rr_ms"` or
        `"rr_s"`. Defaults to `"rr_ms"`.
    bakend :
        Which backend to use. Can be `"bokeh"` or `"tabulate"`. Defaults to `"tabulate"`.
    width, height :
        The table width and height (only for `"bokeh"` backend).

    Returns
    -------
    table :
        The formatted nonlinear domain table, either as a string or as a Bokeh Columns.

    """

    if nonlinear_df is None:
        if (rr is None) | (input_type is None):
            raise ValueError(
                "If no summary dataframe is provided, you must provide an RR time series."
            )
        nonlinear_df = nonlinear_domain(rr=rr, input_type=input_type)  # type: ignore

    data = {
        "variable": [
            "-- Poincaré plot --",
            "SD1",
            "SD2",
            None,
            "-- Recurrence plot --",
            "Mean line length (Lmean)",
            "Max line length (Lmax)",
            "Recurrence rate (REC)",
            "Determinism (DET)",
            "Shannon Entropy (ShanEn)",
        ],
        "units": [
            None,
            "milliseconds (ms)",
            "milliseconds (ms)",
            None,
            None,
            "beats",
            "beats",
            "%",
            "%",
            None,
        ],
        "value": [
            None,
            nonlinear_df[nonlinear_df.Metric == "SD1"]["Values"].iloc[0],
            nonlinear_df[nonlinear_df.Metric == "SD2"]["Values"].iloc[0],
            None,
            None,
            nonlinear_df[nonlinear_df.Metric == "l_mean"]["Values"].iloc[0],
            nonlinear_df[nonlinear_df.Metric == "l_max"]["Values"].iloc[0],
            nonlinear_df[nonlinear_df.Metric == "recurrence_rate"]["Values"].iloc[0],
            nonlinear_df[nonlinear_df.Metric == "determinism_rate"]["Values"].iloc[0],
            nonlinear_df[nonlinear_df.Metric == "shannon_entropy"]["Values"].iloc[0],
        ],
    }

    if backend == "bokeh":
        source = ColumnDataSource(data)

        columns = [
            TableColumn(field="variable", title="Variable"),
            TableColumn(field="units", title="Units"),
            TableColumn(field="value", title="Value"),
        ]
        nonlinear_table = DataTable(
            source=source,
            columns=columns,
            width=width,
            height=height,
            index_position=None,
        )

        table = Column(nonlinear_table)

    elif backend == "tabulate":
        table = tabulate(
            data, headers=["Variable", "Units", "Value"], tablefmt="rst", floatfmt=".4f"
        )

    return table
