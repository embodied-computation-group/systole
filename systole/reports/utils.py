# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import json
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pkg_resources
from bokeh.embed import components
from bokeh.resources import INLINE
from jinja2 import Template

from systole import __version__ as version
from systole.reports.group_level import (
    frequency_domain_group_level,
    nonlinear_domain_group_level,
    time_domain_group_level,
)
from systole.reports.subject_level import subject_level_report


def import_data(
    participant_id: str, bids_folder: str, task: str
) -> Union[
    Tuple[
        Tuple[np.ndarray, int, Optional[Union[np.ndarray, List[int]]]],
        Tuple[np.ndarray, int, Optional[Union[np.ndarray, List[int]]]],
        Tuple[np.ndarray, int, Optional[Union[np.ndarray, List[int]]]],
    ],
    Tuple[Tuple[None, None, None], Tuple[None, None, None], Tuple[None, None, None]],
]:
    """Load ECG, PPG and RESPIRataion dataframes from BIDS folder given participant_id
    and task names.

    Parameters
    ----------
    participant_id : str
        The participant ID. The string should match with one participant in the BIDS
        folder.
    bids_folder : str
        The path to the BIDS folder. This folder should containt the participant
        `participant_id` and have a task `task` with at least one of the possible
        physiological recordings (ECG, PPG, RESPIRATION).
    task : str
        The task name. The string should match with a task in the BIDS folder provided
        as `bids_folder`.

    Returns
    -------
    (
        (ecg, ecg_sfreq, ecg_events_idx),
        (ppg, ppg_sfreq, ppg_events_idx),
        (resp, resp_sfreq, resp_events_idx)
        ) : tuples
        Tuples of signal, sampling frequency and events indexs for ECG, PPG and
        RESPIRATION.

    ecg, ppg, resp : np.ndarray | None
        The ECG, PPG and RESPIRATION signals as Numpy arrays (when available). Otherwise
        returns `None`.
    ecg_sfreq, ppg_sfreq, resp_sfreq: int | None
        The ECG, PPG and RESPIRATION sampling frequency (when the signal is available).
        Otherwise returns `None`.
    ecg_events_idx, ppg_events_idx, resp_events_idx : list | np.ndarray | None
        The ECG, PPG and RESPIRATION events associated with the signals (when available).
        Otherwise returns `None`.

    """
    # Initialize default results
    (
        (ecg, ecg_sfreq, ecg_events_idx),
        (ppg, ppg_sfreq, ppg_events_idx),
        (resp, resp_sfreq, resp_events_idx),
    ) = ((None, None, None), (None, None, None), (None, None, None))

    physio_file = f"{bids_folder}{participant_id}/ses-session1/beh/{participant_id}_ses-session1_task-{task}_physio.tsv.gz"
    json_file = f"{bids_folder}{participant_id}/ses-session1/beh/{participant_id}_ses-session1_task-{task}_physio.json"

    # Opening JSON file
    f = open(json_file)
    sfreq = json.load(f)["SamplingFrequency"]

    # Verify that the file exists, otherwise, return None
    if not os.path.exists(physio_file):
        print(
            f"No physiological recording was found for participant {participant_id} - task: {task}"
        )
        return (
            (ecg, ecg_sfreq, ecg_events_idx),
            (ppg, ppg_sfreq, ppg_events_idx),
            (resp, resp_sfreq, resp_events_idx),
        )

    # Gather physiological signal in the BIDS folder for this participant_id / task
    physio_df = pd.read_csv(physio_file, sep="\t", compression="gzip")
    physio_df.columns = physio_df.columns.str.lower()

    # Find ECG recording if any
    ecg_names = ["ecg", "ekg", "cardiac"]
    ecg_col = [col for col in physio_df.columns if col in ecg_names]
    ecg_col = ecg_col[0] if len(ecg_col) > 0 else None

    if ecg_col:
        ecg = physio_df[ecg_col].to_numpy()
        ecg_sfreq = sfreq

    # Find PPG recording if any
    ppgg_names = ["ppg", "photoplethysmography", "pulse"]
    ppg_col = [col for col in physio_df.columns if col in ppgg_names]
    ppg_col = ppg_col[0] if len(ppg_col) > 0 else None

    if ppg_col:
        ppg = physio_df[ppg_col].to_numpy()
        ppg_sfreq = sfreq

    # Find RESPIRATION recording if any
    resp_names = ["res", "resp", "respiration"]
    resp_col = [col for col in physio_df.columns if col in resp_names]
    resp_col = resp_col[0] if len(resp_col) > 0 else None

    if resp_col:
        resp = physio_df[resp_col].to_numpy()
        resp_sfreq = sfreq

    return (
        (ecg, ecg_sfreq, ecg_events_idx),
        (ppg, ppg_sfreq, ppg_events_idx),
        (resp, resp_sfreq, resp_events_idx),
    )


def create_reports(
    bids_folder: str,
    result_folder: str,
    sessions: Union[str, List[str]],
    tasks: Union[str, List[str]] = None,
    participants_id: Union[str, List[str]] = None,
):
    """Create individual HTML and summary results from BIDS folder and generate a group
    level overview of the results.

    Parameters
    ----------
    bids_folder : str
        Path to the main folder organized according to BIDS standards. The folder must
        contain a task matching with the `task` parameter (if provided) and the
        participants listed in `participants_id` (if provided).
    result_folder : str
        Path to the main output folder. A report folder will be created for each
        participant, containing the summary statistics and HTML reports for each task
        provided in the `task` parameter.
    sessions : str | list
        The session reference that should be analyzed. Should match a session number in
        the BIDS folder. Defaults to `"session1"`.
    tasks : str | list
        The task(s) that should be analyzed.
    participants_id : list | None
        List of participants ID that will be processed. If `None`, all the participants
        listed in the folder will be processed.

    """

    if isinstance(participants_id, str):
        participants_id = [participants_id]
    else:
        raise ValueError("Invalid participants_id parameter.")

    if isinstance(tasks, str):
        tasks = [tasks]
    else:
        raise ValueError("Invalid tasks parameter.")

    for session in sessions:

        for task in tasks:

            for participant_id in participants_id:

                # Import data
                (
                    (ecg, ecg_sfreq, ecg_events_idx),
                    (ppg, ppg_sfreq, ppg_events_idx),
                    (resp, resp_sfreq, resp_events_idx),
                ) = import_data(
                    participant_id=participant_id, bids_folder=bids_folder, task=task
                )

                # End here if no signal was found
                if np.all(
                    [
                        i is None
                        for i in [
                            ecg,
                            ecg_sfreq,
                            ecg_events_idx,
                            ppg,
                            ppg_sfreq,
                            ppg_events_idx,
                            resp,
                            resp_sfreq,
                            resp_events_idx,
                        ]
                    ]
                ):

                    continue

                #########################################
                # Create reports and summary dataframes #
                #########################################
                subject_level_report(
                    participant_id=participant_id,
                    task=task,
                    session=session,
                    result_folder=result_folder,
                    ecg=ecg,
                    ecg_sfreq=ecg_sfreq,
                    ecg_events_idx=ecg_events_idx,
                    ppg=ppg,
                    ppg_sfreq=ppg_sfreq,
                    ppg_events_idx=ppg_events_idx,
                    resp=resp,
                    resp_sfreq=resp_sfreq,
                    resp_events_idx=resp_events_idx,
                )


def wrapper(
    bids_folder: str,
    participants_id: Union[str, List],
    tasks: Union[str, List],
    result_folder: str,
    sessions: Union[str, List[str]] = "session1",
    template_file=pkg_resources.resource_filename(__name__, "./group_level.html"),
    create_individual_reports=True,
):
    """Create group-level interactive report. Will preprocesses subject level data and
    create reports if requested.

    Parameters
    ----------
    bids_folder : str
        The path to the BIDS structured folder containing the raw data.
    participants_id : str | list
        String or list of strings defining the participant(s) ID that should be
        processed.
    tasks : str | list
        The task(s) that will be analyzed. Should match with a task in the BIDS folder.
    result_folder : str
        The result folder.
    sessions : str | list
        The session reference that should be analyzed. Should match a session number in
        the BIDS folder. Defaults to `"session1"`.
    template_file : str
        The template file for group-level report.

    """

    if isinstance(sessions, str):
        sessions = [sessions]
    else:
        raise ValueError("Invalid sessions parameter.")

    if isinstance(tasks, str):
        tasks = [tasks]
    else:
        raise ValueError("Invalid tasks parameter.")

    # Load HTML template
    with open(template_file, "r", encoding="utf-8") as f:
        html_template = f.read()

    for session in sessions:

        for task in tasks:

            # Output file name
            html_filename = (
                f"{result_folder}/group_level_ses-{session}_task-{task}.html"
            )

            # Create individual reports
            if create_individual_reports:
                create_reports(
                    tasks=tasks,
                    sessions=session,
                    result_folder=result_folder,
                    participants_id=participants_id,
                    bids_folder=bids_folder,
                )

            # Gather individual metrics
            summary_df = pd.DataFrame([])
            for sub in participants_id:
                summary_file = f"{result_folder}/{sub}/{session}/{sub}_ses-{session}_task-{task}.tsv"
                if os.path.isfile(summary_file):
                    summary_df = summary_df.append(
                        pd.read_csv(summary_file, sep="\t"),
                        ignore_index=True,
                    )

            time_domain = time_domain_group_level(summary_df)
            frequency_domain = frequency_domain_group_level(summary_df)
            nonlinear_domain = nonlinear_domain_group_level(summary_df)

            # Embed plots in a dictionary
            plots = dict(
                time_domain=time_domain,
                frequency_domain=frequency_domain,
                nonlinear_domain=nonlinear_domain,
            )

            # Create script and div variables that will be passed to the template
            script, div = components(plots)

            # Load HTML template in memory
            template = Template(html_template)
            resources = INLINE.render()

            # Generate HTML txt variables
            show_ecg, show_ppg, show_respiration = True, True, True
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
            print(f"Group-level report saved as {html_filename}")
