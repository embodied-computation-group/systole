import time
import numpy as np
import pandas as pd
import os.path as op

ddir = op.dirname(op.realpath(__file__))

__all__ = ["import_ppg", "import_rr", "serialSim", "import_dataset"]


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


def import_dataset():
    """Import PPG recording.

    Returns
    -------
    df : pandas DataFrame
        Dataframe containing the RR time-serie.

    Notes
    -----
    Load a 20 minutes recording of ECG, EDA and respiration of a young healthy
    participant undergoing the emotional task (valence rating of neutral and
    disgusting images) described in _[1].

    References
    ----------
    [1] : Legrand, N., Etard, O., Vandevelde, A., Pierre, M., Viader, F.,
        Clochon, P., Doidy, F., Peschanski, D., Eustache, F. & Gagnepain, P.
        (2018). Preprint version 3.0.
        doi: https://www.biorxiv.org/content/10.1101/376954v3
    """
    df = pd.read_csv(op.join(ddir, 'task1.txt'))

    return df


# import numpy as np
# import pandas as pd
# from scipy.io import loadmat
# import matplotlib.pyplot as plt
#
# data = loadmat('67LCL_EMO1.mat')
# df = pd.read_table('Eval_Emo_1.txt')
# df.rename(columns=lambda x: x.strip(), inplace=True)
#
# stim = data['data'][:, 4]
# ecg = data['data'][:, 0]
# eda = data['data'][:, 1]
# resp = data['data'][:, 3]
#
# np.where(stim)[0][0]
#
# s = [int(t*2000) + np.where(stim)[0][0] for t in df['Start Time']]
# len(s)
#
#
# for i in range(len(stim)):
#     if stim[i] != 0:
#         stim[i] = 1
#         stim[i+1:i+400] = 0
#
#
# clean_stim = np.zeros(len(stim), dtype=int)
# clean_stim[s[:36]] = 2
# clean_stim[s[36:]] = 3
#
# data = pd.DataFrame({'ecg': np.float32(ecg),
#                      'respiration': np.float32(resp),
#                      'eda': np.float32(eda),
#                      'stim': clean_stim})
# data.to_csv('task1.txt')
