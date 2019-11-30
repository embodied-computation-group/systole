import numpy as np
import pandas as pd
import os.path as op

ddir = op.dirname(op.realpath(__file__))

__all__ = ["import_ppg", "import_rr"]


def import_ppg(id='1'):
    """Import PPG recording.

    Parameters
    ----------
    id : int
        Signal number (1 or 2).

    Returns
    -------
    signal : array
        1d array containing the PPG signal.
    """
    signal = np.load(op.join(ddir, 'ppg' + id + '.npy'))

    return signal


def import_rr():
    """Import PPG recording.

    Parameters
    ----------
    id : int
        Signal number (1 or 2).

    Returns
    -------
    rr : pandas DataFrame
        Dataframe containing the RR time-serie.
    """
    rr = pd.read_csv(op.join(ddir, 'rr.txt'))

    return rr
