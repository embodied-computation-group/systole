# Author:  Leah Banellis <leahbanellis@cfin.au.dk>

def import_manual_correction(peaks_ms:np.ndarray, participant_id:str, session:str, modality:str, pattern:str) -> np.ndarray:

    """ Correct extra and missed peaks identified via manual correction (i.e., using .json file via the systole viewer) 

    Parameters
    ----------
    peaks_ms : np.ndarray
            1d array of RR intervals, or  interbeat intervals (IBI), expressed in milliseconds.
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
    peaks_ms_corrected : np.ndarray
        The corrected RR time series
    """

    from os.path import exists
    import json 

    corrected_path = "/mnt/scratch/BIDS/derivatives/systole/corrected/"
    corrected_json = f"{corrected_path}{participant_id}/{session}/{modality}/sub-{participant_id}_{session}_{pattern}_corrected.json"
    
    # if corrected peaks file exists
    if exists(corrected_json):
        f = open(f'{corrected_json}')
        json_dict = json.load(f)
        
        # add manually selected peaks
        peaks_ms[json_dict['ppg']['add_idx']] = True
        
        # remove manually selected peaks
        peaks_ms[json_dict['ppg']['remove_idx']] = False