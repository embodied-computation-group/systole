# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pkg_resources
from bokeh.embed import components
from bokeh.resources import INLINE
from jinja2 import Template

from systole import __version__ as version
from systole.detection import ecg_peaks, ppg_peaks
from systole.hrv import frequency_domain, nonlinear_domain, time_domain
from systole.plots import plot_frequency, plot_poincare, plot_raw, plot_subspaces
from systole.plots.utils import frequency_table, nonlinear_table, time_table


def subject_level_report(
    participant_id: str,
    task: str,
    result_folder: str,
    session: str,
    ecg: Optional[np.ndarray] = None,
    ppg: Optional[np.ndarray] = None,
    resp: Optional[np.ndarray] = None,
    ecg_sfreq: Optional[int] = None,
    ppg_sfreq: Optional[int] = None,
    resp_sfreq: Optional[int] = None,
    ecg_events_idx: Optional[Union[List, np.ndarray]] = None,
    ppg_events_idx: Optional[Union[List, np.ndarray]] = None,
    resp_events_idx: Optional[Union[List, np.ndarray]] = None,
    ecg_method: str = "pan-tompkins",
    template_file=pkg_resources.resource_filename(__name__, "subject_level.html"),
):
    """Analyse physiological signals for one participant / task, create HTML report
    and return summary dataframe.

    Parameters
    ----------
    participant_id : str
        The participant ID. The string should match with one participant in the BIDS
        folder.
    task : str
        The task name. The string should match with a task in the BIDS folder provided
        as `bids_folder`.
    result_folder : str
        The result folder where the individual HTML reports, the group level reports
        the summary dataframes will be stored.
    session : str | list
        The session reference that should be analyzed. Should match a session number in
        the BIDS folder. Defaults to `"session1"`.
    ecg, ppg, resp :

    Returns
    -------

    This function will save the following files in the report folder.

    summary_df : pd.DataFrame
        Summary HRV statistics (time, frequency and nonlinear domain). Save the
        dataframe as a `.tsv` file in the `result_folder`.
    report_html : html file
        Interactive report of the processing pipeline. Save the HTML file in the
        `result_folder`.

    """

    print(
        f"Creating report for participant {participant_id} - session: {session} - task : {task}. Systole v{version}"
    )

    #######################
    # Paths and filenames #
    #######################

    # Create participant folder if does not exit
    participant_path = os.path.join(result_folder, participant_id, session)
    if not os.path.exists(participant_path):
        os.makedirs(participant_path)

    # The participant's summary dataframe
    tsv_physio_filename = (
        f"{participant_path}/{participant_id}_ses-{session}_task-{task}_physio.tsv"
    )

    # The participant's summary dataframe
    tsv_features_filename = (
        f"{participant_path}/{participant_id}_ses-{session}_task-{task}_features.tsv"
    )

    # The participant's HTML report
    html_filename = f"{result_folder}/{participant_id}_ses-{session}_task-{task}.html"

    # Load HTML template
    with open(template_file, "r", encoding="utf-8") as f:
        html_template = f.read()

    # Create empty summary dataframe
    summary_df, physio_df = pd.DataFrame([]), pd.DataFrame([])

    # Embed plots in a dictionary
    plots: Dict[str, Any] = {}

    #######
    # ECG #
    #######
    if ecg is not None:

        if ecg_sfreq is None:
            raise ValueError("The ECG sampling frequency should be provided")

        print("... Processing ECG recording.")

        new_signal, peaks = ecg_peaks(ecg, sfreq=ecg_sfreq, method=ecg_method)

        physio_df = physio_df.append(
            {"ecg_raw": ecg, "ecg_processed": new_signal, "ecg_peaks": peaks},
            ignore_index=True,
        )

        ecg_raw = plot_raw(
            signal=ecg,
            sfreq=ecg_sfreq,
            modality="ecg",
            ecg_method=ecg_method,
            show_heart_rate=True,
            backend="bokeh",
        )

        ecg_artefacts = plot_subspaces(
            rr=peaks, input_type="peaks", backend="bokeh", figsize=500
        )

        ecg_time_table = time_table(rr=peaks, input_type="peaks", backend="bokeh")

        ecg_plot_frequency = plot_frequency(
            rr=peaks, input_type="peaks", backend="bokeh", figsize=400
        )

        ecg_frequency_table = frequency_table(
            peaks, input_type="peaks", backend="bokeh"
        )

        ecg_plot_poincare = plot_poincare(
            rr=peaks, input_type="peaks", backend="bokeh", figsize=400
        )
        ecg_nonlinear_table = nonlinear_table(
            peaks, input_type="peaks", backend="bokeh"
        )

        # HRV - Time domain
        ecg_time_domain = time_domain(peaks, input_type="peaks")
        ecg_time_domain["hrv_domain"] = "time_domain"
        ecg_time_domain["participant_id"] = participant_id
        ecg_time_domain["task"] = task
        ecg_time_domain["modality"] = "ecg"
        summary_df = summary_df.append(ecg_time_domain, ignore_index=True)

        # HRV - Frequency domain
        ecg_frequency_domain = frequency_domain(peaks, input_type="peaks")
        ecg_frequency_domain["hrv_domain"] = "frequency_domain"
        ecg_frequency_domain["participant_id"] = participant_id
        ecg_frequency_domain["task"] = task
        ecg_frequency_domain["modality"] = "ecg"
        summary_df = summary_df.append(ecg_frequency_domain, ignore_index=True)

        # HRV - Nonlinear domain
        ecg_nonlinear_domain = nonlinear_domain(peaks, input_type="peaks")
        ecg_nonlinear_domain["hrv_domain"] = "nonlinear_domain"
        ecg_nonlinear_domain["participant_id"] = participant_id
        ecg_nonlinear_domain["task"] = task
        ecg_nonlinear_domain["modality"] = "ecg"
        summary_df = summary_df.append(ecg_nonlinear_domain, ignore_index=True)

        plots = {
            **plots,
            **dict(
                ecg_raw=ecg_raw,
                ecg_artefacts=ecg_artefacts,
                ecg_time_table=ecg_time_table,
                ecg_frequency_table=ecg_frequency_table,
                ecg_nonlinear_table=ecg_nonlinear_table,
                ecg_plot_frequency=ecg_plot_frequency,
                ecg_plot_poincare=ecg_plot_poincare,
            ),
        }

    #######
    # PPG #
    #######
    if ppg is not None:

        if ppg_sfreq is None:
            raise ValueError("The PPG sampling frequency should be provided")

        print("... Processing PPG recording")

        new_signal, peaks = ppg_peaks(ppg, sfreq=ppg_sfreq)

        physio_df = physio_df.append(
            {"ppg_raw": ppg, "ppg_processed": new_signal, "ppg_peaks": peaks},
            ignore_index=True,
        )

        ppg_raw = plot_raw(
            signal=ppg,
            sfreq=ppg_sfreq,
            modality="ppg",
            show_heart_rate=True,
            backend="bokeh",
        )

        ppg_artefacts = plot_subspaces(
            rr=peaks, input_type="peaks", backend="bokeh", figsize=500
        )

        ppg_time_table = time_table(peaks, input_type="peaks", backend="bokeh")

        ppg_plot_frequency = plot_frequency(
            rr=peaks, input_type="peaks", backend="bokeh", figsize=400
        )

        ppg_frequency_table = frequency_table(
            peaks, input_type="peaks", backend="bokeh"
        )

        ppg_plot_poincare = plot_poincare(
            rr=peaks, input_type="peaks", backend="bokeh", figsize=400
        )
        ppg_nonlinear_table = nonlinear_table(
            peaks, input_type="peaks", backend="bokeh"
        )

        # HRV - Time domain
        ppg_time_domain = time_domain(peaks, input_type="peaks")
        ppg_time_domain["hrv_domain"] = "time_domain"
        ppg_time_domain["participant_id"] = participant_id
        ppg_time_domain["task"] = task
        ppg_time_domain["modality"] = "ppg"
        summary_df = summary_df.append(ppg_time_domain, ignore_index=True)

        # HRV - Frequency domain
        ppg_frequency_domain = frequency_domain(peaks, input_type="peaks")
        ppg_frequency_domain["hrv_domain"] = "frequency_domain"
        ppg_frequency_domain["participant_id"] = participant_id
        ppg_frequency_domain["task"] = task
        ppg_frequency_domain["modality"] = "ppg"
        summary_df = summary_df.append(ppg_frequency_domain, ignore_index=True)

        # HRV - Nonlinear domain
        ppg_nonlinear_domain = nonlinear_domain(peaks, input_type="peaks")
        ppg_nonlinear_domain["hrv_domain"] = "nonlinear_domain"
        ppg_nonlinear_domain["participant_id"] = participant_id
        ppg_nonlinear_domain["task"] = task
        ppg_nonlinear_domain["modality"] = "ppg"
        summary_df = summary_df.append(ppg_nonlinear_domain, ignore_index=True)

        plots = {
            **plots,
            **dict(
                ppg_raw=ppg_raw,
                ppg_artefacts=ppg_artefacts,
                ppg_time_table=ppg_time_table,
                ppg_frequency_table=ppg_frequency_table,
                ppg_nonlinear_table=ppg_nonlinear_table,
                ppg_plot_frequency=ppg_plot_frequency,
                ppg_plot_poincare=ppg_plot_poincare,
            ),
        }

    ###############
    # RESPIRATION #
    ###############
    if resp is not None:

        if resp_sfreq is None:
            raise ValueError("The Respiration sampling frequency should be provided")

        print("... Processing Respiration recording")

    ##################
    # Saving as HTML #
    ##################

    print(f"... Saving the report as HTML - filename: {html_filename}")

    # Create script and div variables that will be passed to the template
    script, div = components(plots)

    # Load HTML template in memory
    template = Template(html_template)
    resources = INLINE.render()

    # Generate HTML txt variables
    show_ecg, show_ppg, show_respiration = (
        (ecg is not None),
        (ppg is not None),
        (resp is not None),
    )
    html = template.render(
        resources=resources,
        script=script,
        div=div,
        systole_version=version,
        show_ecg=show_ecg,
        show_ppg=show_ppg,
        show_respiration=show_respiration,
    )

    # Save the HTML file locally
    with open(html_filename, mode="w", encoding="utf-8") as f:
        f.write(html)

    #############################
    # Save the physio dataframe #
    #############################
    print(
        f"... Saving the summary result as .tsv file - filename: {tsv_physio_filename}."
    )
    physio_df.to_csv(tsv_physio_filename, sep="\t", index=False)

    ##############################
    # Save the summary dataframe #
    ##############################
    print(
        f"... Saving the summary result as .tsv file - filename: {tsv_features_filename}."
    )
    summary_df.to_csv(tsv_features_filename, sep="\t", index=False)
