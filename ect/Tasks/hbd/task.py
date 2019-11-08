# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from psychopy import visual, gui, event
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ect.recording import Oximeter


"""Heart beat counting task.

References
----------
[1] Whitehead, W., Drescher, V. M., Heiman, P., & Blackwell, B. (1977).
    Relation of heart rate control to heartbeat perception. Biofeedback and
    Self Regulation, 2, 371–392.

[2] Wiens, S., & Palmer, S. N. (2001). Quadratic trend analysis and
    heartbeat detection. Biological Psychology, 58, 159–175.

[3] Brener, J., & Kluvitse, C. (1988). Heartbeat detection: Judgments of
    the simultaneity of external stimuli and heartbeats. Psychophysiology,
    25(5), 554–561.
"""


def run(win, oxi):
    """Run the entire task.
    """


def trial(parameters, duration, win=None, oxi=None):
    """Run one trial.

    Parameters
    ----------
    duration : int
        The duration of the recording (in seconds).
    """
    if win is None:
        win = parameters['win']
    if oxi is None:
        oxi = parameters['oxi']
