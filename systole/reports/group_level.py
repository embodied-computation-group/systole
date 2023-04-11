# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import BoxAnnotation, Column, ColumnDataSource, DataTable, TableColumn
from bokeh.plotting import figure
from bokeh.transform import jitter


def time_domain_group_level(summary_df: pd.DataFrame):
    """Plot group-level HRV metric in the time domain.

    Parameters
    ----------
    summary_df :
        Group-level summary of HRV metrics.

    Returns
    -------
    row :

    """

    bpm_metrics = ["MinBPM", "MeanBPM", "MedianBPM", "MaxBPM"]

    source = ColumnDataSource(summary_df[summary_df.Metric.isin(bpm_metrics)])
    TOOLTIPS = [
        ("Participant", "@participant_id"),
        ("Modality", "@modality"),
        ("Task", "@task"),
    ]

    # BPM
    bpm_figure = figure(
        height=300,
        x_range=bpm_metrics,
        tooltips=TOOLTIPS,
        y_axis_label="BPM",
        sizing_mode="stretch_width",
        title="Beats per minute (bpm)",
    )
    bpm_figure.circle(
        x=jitter("Metric", width=0.4, range=bpm_figure.x_range),
        y="Values",
        source=source,
    )

    high, low = 300, 20
    upper_bound = BoxAnnotation(bottom=high, fill_alpha=0.1, fill_color="red")
    upper_bound.level = "underlay"
    bpm_figure.add_layout(upper_bound)
    lower_bound = BoxAnnotation(top=low, fill_alpha=0.1, fill_color="red")
    lower_bound.level = "underlay"
    bpm_figure.add_layout(lower_bound)

    # RR
    RR_metrics = ["MinRR", "MeanRR", "MedianRR", "MaxRR"]

    source = ColumnDataSource(summary_df[summary_df.Metric.isin(RR_metrics)])

    RR_figure = figure(
        height=300,
        x_range=RR_metrics,
        tooltips=TOOLTIPS,
        y_axis_label="milliseconds (ms)",
        sizing_mode="stretch_width",
        title="RR intervals (ms)",
    )
    RR_figure.circle(
        x=jitter("Metric", width=0.4, range=RR_figure.x_range),
        y="Values",
        source=source,
    )

    high, low = 3000, 200
    upper_bound = BoxAnnotation(bottom=high, fill_alpha=0.1, fill_color="red")
    upper_bound.level = "underlay"
    RR_figure.add_layout(upper_bound)
    lower_bound = BoxAnnotation(top=low, fill_alpha=0.1, fill_color="red")
    lower_bound.level = "underlay"
    RR_figure.add_layout(lower_bound)

    ###############
    # Time domain #
    ###############
    td_metrics = ["SDNN", "SDSD", "RMSSD", "nn50", "pnn50"]
    td = ()

    for metric in td_metrics:
        source = ColumnDataSource(summary_df[summary_df.Metric == metric])

        p = figure(
            height=300,
            width=100,
            x_range=[metric],
            tooltips=TOOLTIPS,
            sizing_mode="fixed",
            title=metric,
        )
        p.toolbar.logo = None
        p.toolbar_location = None
        p.circle(
            x=jitter("Metric", width=0.4, range=p.x_range),
            y="Values",
            source=source,
            alpha=0.8,
        )

        td += (p,)  # type: ignore

    return row(bpm_figure, RR_figure, *td)


def frequency_domain_group_level(summary_df: pd.DataFrame):
    """Plot group-level HRV metric in the frequency domain.

    Parameters
    ----------
    summary_df :
        Group-level summary of HRV metrics.

    Returns
    -------
    row :

    """

    TOOLTIPS = [
        ("Participant", "@participant_id"),
        ("Modality", "@modality"),
        ("Task", "@task"),
    ]

    # Peak
    peak_metrics = ["vlf_peak", "lf_peak", "hf_peak"]
    source = ColumnDataSource(summary_df[summary_df.Metric.isin(peak_metrics)])
    peak_figure = figure(
        height=300,
        x_range=peak_metrics,
        tooltips=TOOLTIPS,
        y_axis_label="Peaks (Hz)",
        sizing_mode="stretch_width",
        title="Frequency peaks",
    )
    peak_figure.circle(
        x=jitter("Metric", width=0.4, range=peak_figure.x_range),
        y="Values",
        source=source,
    )

    # Power (ms²)
    power_metrics = ["vlf_power", "lf_power", "hf_power"]
    source = ColumnDataSource(summary_df[summary_df.Metric.isin(power_metrics)])

    power_figure = figure(
        height=300,
        x_range=power_metrics,
        tooltips=TOOLTIPS,
        y_axis_label="Power (ms²)",
        sizing_mode="stretch_width",
        title="Frequency power",
    )
    power_figure.circle(
        x=jitter("Metric", width=0.4, range=power_figure.x_range),
        y="Values",
        source=source,
    )

    # Power (per)
    per_metrics = ["vlf_power_per", "lf_power_per", "hf_power_per"]
    source = ColumnDataSource(summary_df[summary_df.Metric.isin(per_metrics)])
    per_figure = figure(
        height=300,
        x_range=per_metrics,
        tooltips=TOOLTIPS,
        y_axis_label="Peaks (Hz)",
        sizing_mode="stretch_width",
        title="Frequency power (%)",
    )
    per_figure.circle(
        x=jitter("Metric", width=0.4, range=per_figure.x_range),
        y="Values",
        source=source,
    )

    # Power (nu)
    nu_metrics = ["vlf_power_nu", "lf_power_nu", "hf_power_nu"]
    source = ColumnDataSource(summary_df[summary_df.Metric.isin(nu_metrics)])
    nu_figure = figure(
        height=300,
        x_range=nu_metrics,
        tooltips=TOOLTIPS,
        y_axis_label="Peaks (Hz)",
        sizing_mode="stretch_width",
        title="Frequency power (n.u.)",
    )
    nu_figure.circle(
        x=jitter("Metric", width=0.4, range=nu_figure.x_range),
        y="Values",
        source=source,
    )

    td = ()

    for metric in ["total_power", "lf_hf_ratio"]:
        source = ColumnDataSource(summary_df[summary_df.Metric == metric])

        p = figure(
            height=300,
            width=100,
            x_range=[metric],
            tooltips=TOOLTIPS,
            sizing_mode="fixed",
            title=metric,
        )
        p.toolbar.logo = None
        p.toolbar_location = None
        p.circle(
            x=jitter("Metric", width=0.4, range=p.x_range),
            y="Values",
            source=source,
            alpha=0.8,
        )

        td += (p,)  # type: ignore

    return row(power_figure, peak_figure, per_figure, *td)


def nonlinear_domain_group_level(summary_df: pd.DataFrame):
    """Plot group-level HRV metric in the nonlinear domain.

    Parameters
    ----------
    summary_df :
        Group-level summary of HRV metrics.

    Returns
    -------
    row :
        Sub-plot showing the group-level nonlinear metrics.

    """

    TOOLTIPS = [
        ("Participant", "@participant_id"),
        ("Modality", "@modality"),
        ("Task", "@task"),
    ]
    nonlinear_metrics = [
        "SD1",
        "SD2",
        "recurrence_rate",
        "l_max",
        "l_mean",
        "determinism_rate",
        "shannon_entropy",
    ]
    td = ()

    for metric in nonlinear_metrics:
        source = ColumnDataSource(summary_df[summary_df.Metric == metric])

        p = figure(
            height=300,
            x_range=[metric],
            tooltips=TOOLTIPS,
            sizing_mode="stretch_width",
            title=metric,
        )
        p.toolbar.logo = None
        p.toolbar_location = None
        p.circle(
            x=jitter("Metric", width=0.4, range=p.x_range),
            y="Values",
            source=source,
            alpha=0.8,
        )

        td += (p,)  # type: ignore

    return row(*td)


def artefacts_group_level(summary_df: pd.DataFrame):
    """Create striplot and table visualization for artefacts detected in the RR time
    series.

    Parameters
    ----------
    summary_df :
        Group-level summary of HRV metrics.

    Returns
    -------
    row :
        Sub-plot showing the group-level artefacts metrics.

    """

    # Create a dataframe summarizing the artefacts metrics
    summary_df = summary_df[summary_df.hrv_domain == "artefacts"]
    summary_df = summary_df[["Values", "Metric", "participant_id"]]

    ################################################
    # Creat stripplot for the percent of artefacts #
    ################################################
    n_metrics = [
        "per_artefacts",
        "per_ectopics",
        "per_extra",
        "per_long",
        "per_missed",
        "per_short",
    ]
    source = ColumnDataSource(summary_df[summary_df.Metric.isin(n_metrics)])

    TOOLTIPS = [
        ("Participant", "@participant_id"),
    ]

    per_figure = figure(
        height=300,
        x_range=n_metrics,
        tooltips=TOOLTIPS,
        y_axis_label="% artefacts",
        sizing_mode="stretch_width",
        title="Percentage of artefacts",
    )
    per_figure.circle(
        x=jitter("Metric", width=0.4, range=per_figure.x_range),
        y="Values",
        source=source,
    )

    ####################################
    # Creat table for artefacts report #
    ####################################
    data = summary_df.pivot(index="participant_id", columns="Metric", values="Values")
    source = ColumnDataSource(data)

    columns = [
        TableColumn(field="participant_id", title="participant_id"),
        TableColumn(field="n_artefacts", title="Number of artefacts"),
        TableColumn(field="n_beats", title="Number of heart beats"),
        TableColumn(field="n_ectopics", title="Number of ectopic beats"),
        TableColumn(field="n_extra", title="Number of extra beats"),
        TableColumn(field="n_long", title="Number of long beats"),
        TableColumn(field="n_missed", title="Number of missed beats"),
        TableColumn(field="n_short", title="Number of shorts beats"),
        TableColumn(field="per_artefacts", title="Percent of artefacts"),
        TableColumn(field="per_ectopics", title="Percent of ectopic beats"),
        TableColumn(field="per_extra", title="Percent of extra beats"),
        TableColumn(field="per_long", title="Percent of long beats"),
        TableColumn(field="per_miseed", title="Percent of missed beats"),
        TableColumn(field="per_short", title="Percent of short beats"),
    ]

    artefacts_table = DataTable(
        source=source,
        columns=columns,
        index_position=None,
    )

    table = Column(artefacts_table)

    return column(per_figure, table)
