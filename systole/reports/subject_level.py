# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pkg_resources  # type: ignore
from bokeh.embed import components
from bokeh.resources import INLINE
from jinja2 import Template

from systole import __version__ as version
from systole.detection import ecg_peaks, ppg_peaks, rr_artefacts, rsp_peaks
from systole.hrv import frequency_domain, nonlinear_domain, time_domain
from systole.plots import (
    plot_frequency,
    plot_poincare,
    plot_rr,
    plot_rsp,
    plot_subspaces,
)
from systole.reports.tables import frequency_table, nonlinear_table, time_table


def subject_level_report(
    participant_id: str,
    task: str,
    result_folder: str,
    session: str,
    ecg: Optional[np.ndarray] = None,
    ppg: Optional[np.ndarray] = None,
    rsp: Optional[np.ndarray] = None,
    ecg_sfreq: Optional[int] = None,
    ppg_sfreq: Optional[int] = None,
    rsp_sfreq: Optional[int] = None,
    ecg_events_idx: Optional[Union[List, np.ndarray]] = None,
    ppg_events_idx: Optional[Union[List, np.ndarray]] = None,
    rsp_events_idx: Optional[Union[List, np.ndarray]] = None,
    ecg_method: str = "sleepecg",
    show_raw: bool = False,
    template_file=pkg_resources.resource_filename(__name__, "subject_level.html"),
):
    """Analyse physiological signals for one participant / task, create HTML report
    and save a summary dataframe.

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
    ecg, ppg, rsp : np.ndarray | None
        The physiological signal that will be analyzed. If `None`, no analyse are
        performed.
    ecg_sfreq, ppg_sfreq, rsp_sfreq : int | None
        The sampling frequencies of the signal of interest.
    ecg_events_idx, ppg_events_idx, rsp_events_idx : np.ndarray | list | None
        The sample indexes of events of interest associated with the recordings.
    ecg_method : str
        The peak detection algorithm used for the ECG signal. Defaults to `"sleepecg"`.
    show_raw : bool
        If `False` (default), the individual report shows the instantaneous heart rate
        without the raw signal to save memory, otherwise will show both.
    template_file : str
        Path to the HTML template to use for individual reports.

    Returns
    -------

    This function will save the following files in the report folder.

    summary_df : pd.DataFrame
        Summary HRV statistics (time, frequency and nonlinear domain). Save the
        dataframe as a `.tsv` file in the `result_folder`.
    report_html : html file
        Interactive report of the processing pipeline. Save the HTML file in the
        `result_folder`.

    Raises
    ------
    ValueError
        If ecg, ppg or rsp are provided but ecg_sfreq, ppg_sfreq or rsp_sfreq are set to
        `None` (respectively).

    """

    print(
        f"Creating report for participant {participant_id} - session: {session} - task : {task}. Systole v{version}"
    )

    #######################
    # Paths and filenames #
    #######################

    # Create participant folder if does not exit
    participant_path = os.path.join(result_folder, participant_id, f"ses-{session}")
    if not os.path.exists(participant_path):
        os.makedirs(participant_path)

    # The participant's summary dataframe
    tsv_physio_filename = (
        f"{participant_path}/{participant_id}_ses-{session}_task-{task}_physio.tsv.gz"
    )

    # The participant's summary dataframe
    tsv_features_filename = (
        f"{participant_path}/{participant_id}_ses-{session}_task-{task}_features.tsv"
    )

    # The participant's HTML report
    html_filename = (
        f"{participant_path}/{participant_id}_ses-{session}_task-{task}.html"
    )

    # Load HTML template
    with open(template_file, "r", encoding="utf-8") as f:
        html_template = f.read()

    # Create empty summary dataframe
    summary_df, physio_df = pd.DataFrame([]), pd.DataFrame([])

    # Embed plots in a dictionary
    plots: Dict[str, Any] = {}

    #####################
    # Drop bad channels #
    #####################

    # Flat signal
    if ecg is not None:
        if np.all(ecg == ecg[0]):
            ecg, ecg_events_idx, ecg_sfreq = None, None, None
    if rsp is not None:
        if np.all(rsp == rsp[0]):
            rsp, rsp_events_idx, rsp_sfreq = None, None, None
    if ppg is not None:
        if np.all(ppg == ppg[0]):
            ppg, ppg_events_idx, ppg_sfreq = None, None, None

    #######
    # ECG #
    #######
    if ecg is not None:

        if ecg_sfreq is None:
            raise ValueError("The ECG sampling frequency should be provided")

        print("... Processing ECG recording.")

        # R wave detection
        _, peaks = ecg_peaks(ecg, sfreq=ecg_sfreq, method=ecg_method, clean_nan=True)
        physio_df["ecg_peaks"] = peaks

        # Artefacts detection
        artefacts = rr_artefacts(peaks, input_type="peaks")

        # Extract some metrics of interest from the artefacts detection and create a
        # data frame that will be added to the summary
        n_beats = sum(peaks)
        values = [
            n_beats,
            sum(artefacts["missed"]),
            (sum(artefacts["missed"]) / n_beats) * 100,
            sum(artefacts["long"]),
            (sum(artefacts["long"]) / n_beats) * 100,
            sum(artefacts["extra"]),
            (sum(artefacts["extra"]) / n_beats) * 100,
            sum(artefacts["short"]),
            (sum(artefacts["short"]) / n_beats) * 100,
            sum(artefacts["ectopic"]),
            (sum(artefacts["ectopic"]) / n_beats) * 100,
            sum(artefacts["ectopic"])
            + sum(artefacts["missed"])
            + sum(artefacts["long"])
            + sum(artefacts["extra"])
            + sum(artefacts["short"]),
            (
                (
                    sum(artefacts["ectopic"])
                    + sum(artefacts["missed"])
                    + sum(artefacts["long"])
                    + sum(artefacts["extra"])
                    + sum(artefacts["short"])
                )
                / n_beats
            )
            * 100,
        ]
        metrics = [
            "n_beats",
            "n_missed",
            "per_missed",
            "n_long",
            "per_long",
            "n_extra",
            "per_extra",
            "n_short",
            "per_short",
            "n_ectopics",
            "per_ectopics",
            "n_artefacts",
            "per_artefacts",
        ]
        ecg_artefacts_df = pd.DataFrame({"Values": values, "Metric": metrics})
        ecg_artefacts_df["participant_id"] = participant_id
        ecg_artefacts_df["task"] = task
        ecg_artefacts_df["modality"] = "ecg"
        ecg_artefacts_df["hrv_domain"] = "artefacts"
        summary_df = pd.concat([summary_df, ecg_artefacts_df], ignore_index=True)

        ecg_rr = plot_rr(
            rr=peaks,
            input_type="peaks",
            show_artefacts=True,
            backend="bokeh",
        )

        ecg_artefacts = plot_subspaces(
            artefacts=artefacts, backend="bokeh", figsize=500
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
        summary_df = pd.concat([summary_df, ecg_time_domain], ignore_index=True)

        # HRV - Frequency domain
        ecg_frequency_domain = frequency_domain(peaks, input_type="peaks")
        ecg_frequency_domain["hrv_domain"] = "frequency_domain"
        ecg_frequency_domain["participant_id"] = participant_id
        ecg_frequency_domain["task"] = task
        ecg_frequency_domain["modality"] = "ecg"
        summary_df = pd.concat([summary_df, ecg_frequency_domain], ignore_index=True)

        # HRV - Nonlinear domain
        ecg_nonlinear_domain = nonlinear_domain(peaks, input_type="peaks")
        ecg_nonlinear_domain["hrv_domain"] = "nonlinear_domain"
        ecg_nonlinear_domain["participant_id"] = participant_id
        ecg_nonlinear_domain["task"] = task
        ecg_nonlinear_domain["modality"] = "ecg"
        summary_df = pd.concat([summary_df, ecg_nonlinear_domain], ignore_index=True)

        ############################
        # Instantaneous heart rate #
        ############################
        if ecg_events_idx is not None:
            pass

        plots = {
            **plots,
            **dict(
                ecg_rr=ecg_rr,
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

        _, peaks = ppg_peaks(ppg, sfreq=ppg_sfreq, clean_nan=True)

        physio_df["ppg_peaks"] = peaks

        ppg_rr = plot_rr(
            rr=peaks,
            input_type="peaks",
            show_artefacts=True,
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
        summary_df = pd.concat([summary_df, ppg_time_domain], ignore_index=True)

        # HRV - Frequency domain
        ppg_frequency_domain = frequency_domain(peaks, input_type="peaks")
        ppg_frequency_domain["hrv_domain"] = "frequency_domain"
        ppg_frequency_domain["participant_id"] = participant_id
        ppg_frequency_domain["task"] = task
        ppg_frequency_domain["modality"] = "ppg"
        summary_df = pd.concat([summary_df, ppg_frequency_domain], ignore_index=True)

        # HRV - Nonlinear domain
        ppg_nonlinear_domain = nonlinear_domain(peaks, input_type="peaks")
        ppg_nonlinear_domain["hrv_domain"] = "nonlinear_domain"
        ppg_nonlinear_domain["participant_id"] = participant_id
        ppg_nonlinear_domain["task"] = task
        ppg_nonlinear_domain["modality"] = "ppg"
        summary_df = pd.concat([summary_df, ppg_nonlinear_domain], ignore_index=True)

        ############################
        # Instantaneous heart rate #
        ############################
        if ppg_events_idx is not None:
            pass

        plots = {
            **plots,
            **dict(
                ppg_rr=ppg_rr,
                ppg_artefacts=ppg_artefacts,
                ppg_time_table=ppg_time_table,
                ppg_frequency_table=ppg_frequency_table,
                ppg_nonlinear_table=ppg_nonlinear_table,
                ppg_plot_frequency=ppg_plot_frequency,
                ppg_plot_poincare=ppg_plot_poincare,
            ),
        }

    ###############
    # Respiration #
    ###############
    if rsp is not None:

        if rsp_sfreq is None:
            raise ValueError("The respiration sampling frequency should be provided")

        print("... Processing respiration recording")

        _, out = rsp_peaks(rsp, sfreq=rsp_sfreq, clean_nan=True)
        peaks, troughs = out

        physio_df["rsp_peaks"] = peaks
        physio_df["rsp_troughs"] = troughs

        rsp_raw = plot_rsp(
            signal=rsp,
            sfreq=rsp_sfreq,
            backend="bokeh",
        )

        ############################
        # Instantaneous heart rate #
        ############################
        if rsp_events_idx is not None:
            pass

        plots = {
            **plots,
            **dict(
                rsp_raw=rsp_raw,
            ),
        }

    ##################
    # Saving as HTML #
    ##################
    if (ppg is None) & (ecg is None) & (rsp is None):
        return
    else:
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
            (rsp is not None),
        )
        html = template.render(
            resources=resources,
            script=script,
            div=div,
            systole_version=version,
            show_ecg=show_ecg,
            show_ppg=show_ppg,
            show_respiration=show_respiration,
            show_raw=show_raw,
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
        if len(physio_df) > 0:
            physio_df.to_csv(
                tsv_physio_filename, sep="\t", index=False, compression="gzip"
            )

        ##############################
        # Save the summary dataframe #
        ##############################
        print(
            f"... Saving the summary result as .tsv file - filename: {tsv_features_filename}."
        )
        if len(summary_df):
            summary_df.to_csv(tsv_features_filename, sep="\t", index=False)
