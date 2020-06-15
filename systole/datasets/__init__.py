import time
import numpy as np
import pandas as pd
import os.path as op

ddir = op.dirname(op.realpath(__file__))

__all__ = ["import_ppg", "import_rr", "serialSim"]


# Simulate serial inputs from ppg recording
# =========================================
class serialSim():
    """Simulate online data cquisition using pre recorded signal and realistic
    recording duration.
    """

    def __init__(self, id='1'):
        self.sfreq = 75
        self.ppg = import_ppg()[0]
        self.start = time.time()

    def inWaiting(self):
        if time.time() - self.start > 1 / self.sfreq:
            self.start = time.time()
            lenInWating = 5
        else:
            lenInWating = 0

        return lenInWating

    def read(self, length):

        if len(self.ppg) == 0:
            self.ppg = import_ppg()[0]

        # Read 1rst item of ppg signal
        rec = self.ppg[:1]
        self.ppg = self.ppg[1:]

        # Build valid paquet
        paquet = [1, 255, rec[0], 127]
        paquet.append(sum(paquet) % 256)

        return paquet[0], paquet[1], paquet[2], paquet[3], paquet[4]

    def reset_input_buffer(self):
        print('Reset input buffer')


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

    Returns
    -------
    rr : pandas DataFrame
        Dataframe containing the RR time-serie.
    """
    rr = pd.read_csv(op.join(ddir, 'rr.txt'))

    return rr
