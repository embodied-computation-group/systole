import numpy as np
import pandas as pd


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
    signal = np.load('datasets/ppgSignal' + id + '.npy')

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
    rr = pd.read_csv('datasets/rr.txt')

    return rr
