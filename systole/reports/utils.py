# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import json
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from systole.reports.subject_level import subject_level_report


def import_data(
    participant_id: str,
    task: str,
    bids_folder: str,
    session: str = "session1",
) -> Union[
    Tuple[
        Tuple[np.ndarray, int, Optional[Union[np.ndarray, List[int]]]],
        Tuple[np.ndarray, int, Optional[Union[np.ndarray, List[int]]]],
        Tuple[np.ndarray, int, Optional[Union[np.ndarray, List[int]]]],
    ],
    Tuple[Tuple[None, None, None], Tuple[None, None, None], Tuple[None, None, None]],
]:
    """Load ECG, PPG and respiration dataframes from BIDS folder given participant_id,
    session and task names.

    Parameters
    ----------
    participant_id : str
        The participant ID. The string should match with one participant in the BIDS
        folder provided as `"bids_folder"`.
    task : str
        The task name. The string should match with a task in the BIDS folder provided
        as `"bids_folder"`.
    session : str
        The session name. The string should match with a session in the BIDS folder
        provided as `"bids_folder"`. Defaults to `"session1"`.
    bids_folder : str
        The path to the BIDS folder. This folder should containt the participant
        `participant_id` and have a task `task` with at least one of the possible
        physiological recordings (ECG, PPG, RESPIRATION).

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

    physio_file = f"{bids_folder}{participant_id}/ses-session1/beh/{participant_id}_ses-{session}_task-{task}_physio.tsv.gz"
    json_file = f"{bids_folder}{participant_id}/ses-session1/beh/{participant_id}_ses-{session}_task-{task}_physio.json"

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
    if not isinstance(participants_id, list):
        raise ValueError("Invalid participants_id parameter.")

    if isinstance(tasks, str):
        tasks = [tasks]
    if not isinstance(tasks, list):
        raise ValueError("Invalid tasks parameter.")

    if isinstance(sessions, str):
        sessions = [sessions]
    if not isinstance(sessions, list):
        raise ValueError("Invalid sessions parameter.")

    for session in sessions:

        for task in tasks:

            for participant_id in participants_id:

                # Import data
                (
                    (ecg, ecg_sfreq, ecg_events_idx),
                    (ppg, ppg_sfreq, ppg_events_idx),
                    (resp, resp_sfreq, resp_events_idx),
                ) = import_data(
                    participant_id=participant_id,
                    bids_folder=bids_folder,
                    task=task,
                    session=session,
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
