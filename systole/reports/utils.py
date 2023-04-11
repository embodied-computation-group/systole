# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import json
from os import PathLike
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from systole.reports.subject_level import subject_level_report
from systole.utils import ecg_strings, ppg_strings, resp_strings


def import_data(
    bids_folder: Union[str, PathLike],
    participant_id: Union[str, PathLike],
    pattern: str = "",
    modality: Union[str, PathLike] = "beh",
    session: Union[str, PathLike] = "ses-session1",
) -> Tuple[
    Tuple[Optional[np.ndarray], Optional[int], Optional[np.ndarray]],
    Tuple[Optional[np.ndarray], Optional[int], Optional[np.ndarray]],
    Tuple[Optional[np.ndarray], Optional[int], Optional[np.ndarray]],
    Optional[str],
]:
    """Load ECG, PPG and respiration dataframes from BIDS folder given participant_id,
    session and pattern names.

    Parameters
    ----------
    bids_folder :
        The path to the BIDS folder. This folder should containt the participant
        `participant_id` and have a pattern `pattern` with at least one of the possible
        physiological recordings (ECG, PPG, respiration).
    participant_id :
        The participant ID. The string should match with one participant in the BIDS
        folder provided as `"bids_folder"`.
    pattern :
        The pattern name. The string should match with a pattern in the BIDS folder provided
        as `"bids_folder"`.
    modality :
        The type of data (e.g. `"beh"`, `"func"`...) where the physiological recording
        is stored. Defaults to `"beh"`.
    session :
        The session name. The string should match with a session in the BIDS folder
        provided as `"bids_folder"`. Defaults to `"session1"`.

    Returns
    -------
    (
        (ecg, ecg_sfreq, ecg_events_idx),
        (ppg, ppg_sfreq, ppg_events_idx),
        (rsp, rsp_sfreq, rsp_events_idx),
        file_name
        ) :
        Tuples of signal, sampling frequency and events indexs for ECG, PPG and
        respiration with the file name from the BIDS folder.

    ecg, ppg, rsp :
        The ECG, PPG and respiration signals as Numpy arrays (when available). Otherwise
        returns `None`.
    ecg_sfreq, ppg_sfreq, rsp_sfreq :
        The ECG, PPG and respiration sampling frequency (when the signal is available).
        Otherwise returns `None`.
    ecg_events_idx, ppg_events_idx, rsp_events_idx :
        The ECG, PPG and respiration events associated with the signals (when available).
        Otherwise returns `None`.
    file_name :
        File name that will be used to save the results in the corresponding derivative
        folder.

    """

    # Initialize default results
    (
        (ecg, ecg_sfreq, ecg_events_idx),
        (ppg, ppg_sfreq, ppg_events_idx),
        (rsp, rsp_sfreq, rsp_events_idx),
    ) = ((None, None, None), (None, None, None), (None, None, None))

    # Try to find a uniqe file corresponding to the info provided, otherwise print an
    # error message and return None
    physio_files = list(
        Path(bids_folder, participant_id, session, modality).glob(
            f"*{pattern}*_physio.tsv.gz"
        )
    )

    if len(physio_files) == 0:
        print(
            f"... No physiological recording was found for participant: {participant_id}"
        )
        return (
            (ecg, ecg_sfreq, ecg_events_idx),
            (ppg, ppg_sfreq, ppg_events_idx),
            (rsp, rsp_sfreq, rsp_events_idx),
            None,
        )
    elif len(physio_files) > 1:
        print(
            f"... More than one physiological recording was found for participant: {participant_id}"
        )
        return (
            (ecg, ecg_sfreq, ecg_events_idx),
            (ppg, ppg_sfreq, ppg_events_idx),
            (rsp, rsp_sfreq, rsp_events_idx),
            None,
        )

    json_files = list(
        Path(bids_folder, participant_id, session, modality).glob(
            f"*{pattern}*_physio.json"
        )
    )

    # Get the unique path for this recording
    json_file = json_files[0]
    physio_file = physio_files[0]

    # Extract the file name (remove the "physio.tsv.gz" suffix)
    file_name = physio_file.name[:-14]

    # Opening JSON file to find the sampling frequency
    f = open(json_file)
    json_data = json.load(f)

    sfreq = json_data["SamplingFrequency"]
    try:
        start_time = json_data["StartTime"]
        end_time = json_data["EndTime"]
    except KeyError:
        start_time, end_time = None, None

    f.close()

    # Gather physiological signal in the BIDS folder for this participant_id / pattern
    physio_df = pd.read_csv(
        physio_file, sep="\t", compression="gzip", names=json_data["Columns"]
    )
    physio_df.columns = physio_df.columns.str.lower()

    # Find ECG recording if any
    ecg_col = [col for col in physio_df.columns if col in ecg_strings]
    ecg_col = ecg_col[0] if len(ecg_col) > 0 else None

    if ecg_col:
        ecg = physio_df[ecg_col].to_numpy()
        ecg_sfreq = sfreq

    # Find PPG recording if any
    ppg_col = [col for col in physio_df.columns if col in ppg_strings]
    ppg_col = ppg_col[0] if len(ppg_col) > 0 else None

    if ppg_col:
        ppg = physio_df[ppg_col].to_numpy()
        ppg_sfreq = sfreq

    # Find respiration recording if any
    rsp_col = [col for col in physio_df.columns if col in resp_strings]
    rsp_col = rsp_col[0] if len(rsp_col) > 0 else None

    if rsp_col:
        rsp = physio_df[rsp_col].to_numpy()
        rsp_sfreq = sfreq

    # If start_time and end_time are provided, trim the signal accordingly and correct
    # the event indexes.
    if start_time is not None:
        start_time_idx = int(start_time * sfreq)
        end_time_idx = int(end_time * sfreq)

        if ecg is not None:
            ecg = ecg[start_time_idx:end_time_idx]
            if ecg_events_idx is not None:
                ecg_events_idx -= start_time_idx

        if ppg is not None:
            ppg = ppg[start_time_idx:end_time_idx]
            if ppg_events_idx is not None:
                ppg_events_idx -= start_time_idx

        if rsp is not None:
            rsp = rsp[start_time_idx:end_time_idx]
            if rsp_events_idx is not None:
                rsp_events_idx -= start_time_idx

    return (
        (ecg, ecg_sfreq, ecg_events_idx),
        (ppg, ppg_sfreq, ppg_events_idx),
        (rsp, rsp_sfreq, rsp_events_idx),
        file_name,
    )


def create_reports(
    participant_id: str,
    bids_folder: str,
    result_folder: str,
    pattern: str,
    modality: str = "beh",
    session: str = "ses-session1",
    html_report: bool = False,
):
    """Create individual HTML and summary results from one participant in the BIDS
    folder.

    Parameters
    ----------
    participant_id :
        List of participants ID that will be processed. If `None`, all the participants
        listed in the folder will be processed.
    bids_folder :
        Path to the main folder organized according to BIDS standards. The folder must
        contain a pattern matching with the `pattern` parameter (if provided) and the
        participants listed in `participants_id` (if provided).
    result_folder :
        Path to the main output folder. A report folder will be created for each
        participant, containing the summary statistics and HTML reports for each pattern
        provided in the `pattern` parameter.
    pattern :
        The pattern(s) that should be analyzed. Should match a pattern reference in the BIDS
        folder.
    modality :
        The type of data (e.g. `"beh"`, `"func"`...) where the physiological recording
        is stored. Defaults to `"beh"`.
    session :
        The session reference that should be analyzed. Should match a session number in
        the BIDS folder. Defaults to `"session1"`.
    html_report :
        If `True`, save an html report. This file embeds the signal for
        interactive visualization and can therefore be large, it is recommended to
        generate HTML reports for review or problematic recordings only.

    """

    # Import ECG, PPG and RESPIRATION recording from the BIDS folder
    (
        (ecg, ecg_sfreq, ecg_events_idx),
        (ppg, ppg_sfreq, ppg_events_idx),
        (rsp, rsp_sfreq, rsp_events_idx),
        file_name,
    ) = import_data(
        participant_id=participant_id,
        bids_folder=bids_folder,
        modality=modality,
        pattern=pattern,
        session=session,
    )

    #######################
    # Detect bad channels #
    #######################
    if ecg is not None:
        if (ecg == ecg[0]).all():
            ecg, ecg_sfreq, ecg_events_idx = None, None, None
    if ppg is not None:
        if (ppg == ppg[0]).all():
            ppg, ppg_sfreq, ppg_events_idx = None, None, None
    if rsp is not None:
        if (rsp == rsp[0]).all():
            rsp, rsp_sfreq, rsp_events_idx = None, None, None

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
                rsp,
                rsp_sfreq,
                rsp_events_idx,
            ]
        ]
    ):
        return

    #########################################
    # Create reports and summary dataframes #
    #########################################
    subject_level_report(
        participant_id=participant_id,
        pattern=pattern,
        session=session,
        modality=modality,
        result_folder=result_folder,
        ecg=ecg,
        ecg_sfreq=ecg_sfreq,
        ecg_events_idx=ecg_events_idx,
        ppg=ppg,
        ppg_sfreq=ppg_sfreq,
        ppg_events_idx=ppg_events_idx,
        rsp=rsp,
        rsp_sfreq=rsp_sfreq,
        rsp_events_idx=rsp_events_idx,
        file_name=file_name,
        html_report=html_report,
    )
