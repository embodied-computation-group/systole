# Author:  Leah Banellis <leahbanellis@cfin.au.dk>
import json
from pathlib import Path

import pandas as pd


def import_manual_correction(
    bids_path: str, participant_id: str, session: str, modality: str, pattern: str
) -> pd.DataFrame:

    """Correct extra and missed peaks identified via manual correction (i.e., using saved .json file via the systole viewer)

    Parameters
    ----------
    bids_path : str
        path of bids folder (i.e., "mnt/scratch/BIDS")
    participant_id : str
        participant id (i.e., "sub-0001").
    session : str
        data recording session (i.e., "'ses-session1'").
    modality : str
        data recording modality (i.e., "func").
    pattern : str
        data file pattern (i.e., "task-rest_run-001_recording-exg").

    Returns
    -------
    ppg_df : pd.DataFrame
        DataFrame of raw cardiac signal, preprocessed cardiac signal, uncorrected peaks and corrected peaks.
    """

    # load raw physio signal
    raw_file = Path(
        f"{bids_path}/{participant_id}/{session}/{modality}/{participant_id}_{session}_{pattern}_physio.tsv.gz"
    )
    signal_raw = pd.read_csv(
        raw_file,
        compression="gzip",
        sep="\t",
        header=None,
    )

    # load preprocessed signal
    preproc_file = Path(
        f"{bids_path}/derivatives/systole/{participant_id}/{session}/{modality}/{participant_id}_{session}_{pattern}_physio.tsv.gz"
    )
    signal_preproc = pd.read_csv(
        preproc_file,
        compression="gzip",
        sep="\t",
        header=None,
    )

    # json with manual corrections
    corrected_json_path = Path(
        f"{bids_path}/derivatives/systole/corrected/{participant_id}/{session}/{modality}/sub-{participant_id}_{session}_{pattern}_corrected.json"
    )

    # if corrected peaks json file exists
    if corrected_json_path.is_file():
        f = open(f"{corrected_json_path}")
        json_dict = json.load(f)

        peaks_uncorrected = json_dict["ppg"]["peaks"]

        peaks_corrected = peaks_uncorrected

        # add manually selected peaks
        peaks_corrected[json_dict["ppg"]["add_idx"]] = True

        # remove manually selected peaks
        peaks_corrected[json_dict["ppg"]["remove_idx"]] = False

        # save all
        ppg_df = pd.DataFrame(
            {
                "raw_signal": signal_raw,
                "preproc_signal": signal_preproc,
                "peaks_uncorrected": peaks_uncorrected,
                "peaks_corrected": peaks_corrected,
            }
        )
        ppg_df.to_csv(
            f"{bids_path}/derivatives/systole/corrected/{participant_id}/{session}/{modality}/sub-{participant_id}_{session}_{pattern}_corrected.tsv",
            sep="\t",
        )

    return
