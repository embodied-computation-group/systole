# Author:  Leah Banellis <leahbanellis@cfin.au.dk>
import json
from pathlib import Path

import pandas as pd


def import_manual_correction(
    bids_path: str,
    participant_id: str,
    session: str,
    modality: str,
    pattern: str,
    cardiac_name: str,
) -> pd.DataFrame:

    """Correct extra and missed peaks identified via manual correction (i.e., using saved .json file via the systole viewer)

    Parameters
    ----------
    bids_path : str
        path of bids folder (i.e., "/mnt/scratch/BIDS")
    participant_id : str
        participant id (i.e., "sub-0001").
    session : str
        data recording session (i.e., "ses-session1").
    modality : str
        data recording modality (i.e., "func").
    pattern : str
        data file pattern (i.e., "task-rest_run-001_recording-exg").
    cardiac_name : str
        name of cardiac column in .json (i.e., "ECG", "PPG", "PLETH" etc)

    Returns
    -------
    ppg_df : pd.DataFrame
        DataFrame of raw cardiac signal, preprocessed cardiac signal, uncorrected peaks and corrected peaks.
    """

    # load preprocessed signal (if exists)
    preproc_file = Path(
        f"{bids_path}/derivatives/systole/{participant_id}/{session}/{modality}/{participant_id}_{session}_{pattern}_physio.tsv.gz"
    )
    if preproc_file.is_file():
        cardiac_df = pd.read_csv(
            preproc_file,
            compression="gzip",
            sep="\t",
            header=None,
        )

        # using json - find cardiac signal
        json_file = Path(
            f"{bids_path}/derivatives/systole/{participant_id}/{session}/{modality}/{participant_id}_{session}_{pattern}_physio.json"
        )
        f = open(f"{json_file}")
        json_dict = json.load(f)
        cardiac_df.columns = json_dict["Columns"]
        # select cardiac data as dataframe
        if cardiac_name not in json_dict["Columns"]:
            print(f"{participant_id} - no cardiac data in file")
            return

        cardiac_idx = json_dict["Columns"].index(cardiac_name)
        signal = cardiac_df.iloc[:, cardiac_idx]
        signal = pd.Series.to_frame(signal).rename(columns={cardiac_name: "cardiac"})

    else:
        # if no preproc file - load raw physio signal
        raw_file = Path(
            f"{bids_path}/{participant_id}/{session}/{modality}/{participant_id}_{session}_{pattern}_physio.tsv.gz"
        )
        if raw_file.is_file():
            cardiac_df = pd.read_csv(
                raw_file,
                compression="gzip",
                sep="\t",
                header=None,
            )

            # using json - find cardiac signal
            json_file = Path(
                f"{bids_path}/{participant_id}/{session}/{modality}/{participant_id}_{session}_{pattern}_physio.json"
            )
            f = open(f"{json_file}")
            json_dict = json.load(f)
            cardiac_df.columns = json_dict["Columns"]
            # select cardiac data as dataframe
            if cardiac_name not in json_dict["Columns"]:
                print(f"{participant_id} - no cardiac data in file")
                return

            cardiac_idx = json_dict["Columns"].index(cardiac_name)
            signal = cardiac_df.iloc[:, cardiac_idx]
            signal = pd.Series.to_frame(signal).rename(
                columns={cardiac_name: "cardiac"}
            )

            # # preprocess (resample at 1000 Hz)
            #
        else:
            print(f"{participant_id} - no raw or preproc cardiac file found")
            return

    # json with manual corrections
    corrected_json_path = Path(
        f"{bids_path}/derivatives/systole/corrected/{participant_id}/{session}/{modality}/{participant_id}_{session}_{pattern}_corrected.json"
    )

    # if corrected peaks json file exists
    if corrected_json_path.is_file():
        f_corrected = open(f"{corrected_json_path}")
        json_dict_corrected = json.load(f_corrected)

        peaks_corrected = json_dict_corrected["ppg"]["corrected_peaks"]

        # add corrected peaks column to preprocessed cardiac signal DataFrame
        signal_peaks = pd.concat(
            [signal, pd.DataFrame(peaks_corrected, columns=["peaks_corrected"])], axis=1
        )

        # save all
        signal_peaks.to_csv(
            f"{bids_path}/derivatives/systole/corrected/{participant_id}/{session}/{modality}/{participant_id}_{session}_{pattern}_peakscorrected.tsv.gz",
            sep="\t",
            index=False,
            header=None,
            compression="gzip",
        )

    return
