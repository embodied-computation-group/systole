# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import io
import os
import os.path as op
import time
from typing import List

import numpy as np
import pandas as pd
import requests  # type: ignore
from tqdm import tqdm

ddir = op.dirname(op.realpath(__file__))

__all__ = ["import_ppg", "import_rr", "serialSim", "import_dataset1"]


def get_data_dir() -> str:
    """Return the directory used to cache the example datasets.

    Defaults to `~/.systole/data`, and can be redirected with the
    `SYSTOLE_DATA_DIR` environment variable, which is useful on CI where the
    directory is often restored from a cache between runs.
    """
    default = op.join(op.expanduser("~"), ".systole", "data")
    data_dir = os.environ.get("SYSTOLE_DATA_DIR", default)
    os.makedirs(data_dir, exist_ok=True)

    return data_dir


def _download(url: str, filename: str) -> bytes:
    """Return the content of `url`, caching it under :py:func:`get_data_dir`.

    The example datasets are several megabytes each and are re-imported many
    times over a documentation build or a test session. Caching them keeps the
    download to once per machine rather than once per call.
    """
    target = op.join(get_data_dir(), filename)
    if op.exists(target):
        with open(target, "rb") as handle:
            return handle.read()

    response = requests.get(url)
    response.raise_for_status()

    # Write to a temporary name first so an interrupted download cannot leave a
    # truncated file behind that later calls would happily read back.
    partial = f"{target}.part"
    with open(partial, "wb") as handle:
        handle.write(response.content)
    os.replace(partial, target)

    return response.content


# Simulate serial inputs from ppg recording
# =========================================
class serialSim:
    """Simulate online data acquisition using pre recorded signal and realistic
    sampling rate (75 Hz).
    """

    def __init__(self):
        self.sfreq = 75
        self.ppg = import_ppg().ppg.to_numpy()
        self.start = time.time()

    def inWaiting(self):
        if time.time() - self.start > 1 / self.sfreq:
            self.start = time.time()
            lenInWating = 5
        else:
            lenInWating = 0

        return lenInWating

    def read(self, lenght):
        if len(self.ppg) == 0:
            self.ppg = import_ppg().ppg.to_numpy()

        # Read 1rst item of ppg signal
        rec = self.ppg[:1]
        self.ppg = self.ppg[1:]

        # Build valid paquet
        paquet = [1, 255, rec[0], 127]
        paquet.append(sum(paquet) % 256)

        return paquet[0], paquet[1], paquet[2], paquet[3], paquet[4]

    def reset_input_buffer(self):
        print("Reset input buffer")


def import_ppg() -> pd.DataFrame:
    """Import a 5 minutes long PPG recording.

    Returns
    -------
    df : :py:class:`pandas.DataFrame`
        Dataframe containing the PPG signale.
    """
    path = (
        "https://github.com/embodied-computation-group/systole/raw/"
        "master/systole/datasets/"
    )
    content = _download(f"{path}ppg.npy", "ppg.npy")
    ppg = np.load(io.BytesIO(content), allow_pickle=True)
    df = pd.DataFrame({"ppg": ppg})
    df["time"] = np.arange(0, len(df)) / 75

    return df


def import_rr() -> pd.DataFrame:
    """Import PPG recording.

    Returns
    -------
    rr : :py:class:`pandas.DataFrame`
        Dataframe containing the RR time-serie.
    """
    path = (
        "https://github.com/embodied-computation-group/systole/raw/"
        "master/systole/datasets/"
    )
    rr = pd.read_csv(io.BytesIO(_download(f"{path}rr.txt", "rr.txt")))

    return rr


def import_dataset1(
    modalities: List[str] = ["ECG", "EDA", "Respiration", "Stim"], disable: bool = False
) -> pd.DataFrame:
    """Import ECG, EDA and respiration recording.

    Parameters
    ----------
    modalities : list
        The list of modalities that should be downloaded. Can contain `"ECG"`, `"EDA"`,
        `"Respiration"` or `"Stim"`.
    disable : bool
        Whether to disable the progress bar or not. Default is `False` (show progress
        bar).

    Returns
    -------
    df : :py:class:`pandas.DataFrame`
        Dataframe containing the signal.

    Notes
    -----
    Load a 20 minutes recording of ECG, EDA and respiration of a young healthy
    participant undergoing the emotional task (valence rating of neutral and
    disgusting images) described in _[1]. The sampling frequency is 1000 Hz.

    References
    ----------
    [1] : Legrand, N., Etard, O., Vandevelde, A., Pierre, M., Viader, F., Clochon, P.,
        Doidy, F., Peschanski, D., Eustache, F., & Gagnepain, P. (2020). Long-term
        modulation of cardiac activity induced by inhibitory control over emotional
        memories. Scientific Reports, 10(1). https://doi.org/10.1038/s41598-020-71858-2

    """
    path = "https://github.com/embodied-computation-group/systole/raw/dev/systole/datasets/Task1_"
    pbar = tqdm(modalities, position=0, leave=True, disable=disable)
    data = {}
    for item in pbar:
        pbar.set_description(f"Downloading {item} channel")
        content = _download(f"{path}{item}.npy", f"Task1_{item}.npy")
        data[item.lower()] = np.load(io.BytesIO(content), allow_pickle=True)

    df = pd.DataFrame(data)
    df["time"] = np.arange(0, len(df)) / 1000

    return df
