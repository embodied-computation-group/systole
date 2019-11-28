# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import pandas as pd


def nnX(x, t=50):
    """Number of difference in successive R-R interval > t ms

    Parameters
    ----------
    x : array like
        Length of R-R intervals (in miliseconds).
    t : int
        Threshold value: Defaut is set to 50 ms to calculate the nn50 index.

    Returns
    -------
    nnX : float
        The number of successive differences larger than a value.
    """
    if isinstance(x, list):
        x = np.asarray(x)
    if len(x.shape) > 1:
        raise ValueError('X must be a 1darray')

    # NN50: number of successive differences larger than t ms
    nn = np.sum(np.abs(np.diff(x)) > t)
    return nn


def pnnX(x, t=50):
    """Number of successive differences larger than a value (def = 50ms)

    Parameters
    ----------
    x : array like
        Length of R-R intervals (in miliseconds).
    t : int
        Threshold value: Defaut is set to 50 ms to calculate the nn50 index.

    Returns
    -------
    nn : float
        The proportion of successive differences larger than a value (%).
    """
    if isinstance(x, list):
        x = np.asarray(x)
    if len(x.shape) > 1:
        raise ValueError('X must be a 1darray')

    # nnX: number of successive differences larger than t ms
    nn = nnX(x, t)

    # Proportion of successive differences larger than t ms
    pnnX = 100 * nn / len(np.diff(x))

    return pnnX


def rmssd(x):
    """Root Mean Square of Successive Differences.

    Parameters
    ----------
    x : array like
        Length of R-R intervals (in miliseconds).

    Returns
    -------
    y : float
        The Root Mean Square of Successive Differences (RMSSD).

    Notes
    ------
        The RMSSD is commonly used in the litterature as a good indexe of the
        Autonomic Nervous System’s Parasympathetic activity. The RMSSD iS
        computed using the following formula:

    Examples
    --------
    >>> rr = [800, 850, 810, 720]
    >>> rmssd(rr)
    """
    if isinstance(x, list):
        x = np.asarray(x)
    if len(x.shape) > 1:
        raise ValueError('X must be a 1darray')

    y = np.sqrt(np.mean(np.square(np.diff(x))))

    return y


def time_domain(x):
    """Extract all time domain parameters from R-R intervals.

    Parameters
    ----------
    x : array like
        Length of R-R intervals (in miliseconds).

    Returns
    -------
    stats : Pandas DataFrame
        Time domain summary:

        'Mean R-R' : Mean of R-R intervals
        'Median R-R : Median of R-R intervals'
        'Min' : Minimum R-R intervals
        'Max' : Maximum R-R intervals
        'Std' : Standard deviation
        'RMSSD' : Root Mean Square of Successive Differences
        'NN50' : number of successive differences larger than 50ms
        'pNN50'
    """

    if isinstance(x, list):
        x = np.asarray(x)
    if len(x.shape) > 1:
        raise ValueError('X must be a 1darray')

    # Mean R-R intervals
    mean_rr = round(np.mean(x))

    # Mean BPM
    mean_bpm = round(np.mean(60000/x), 2)

    # Median BPM
    median_rr = round(np.median(x), 2)

    # Median BPM
    median_bpm = round(np.median(60000/x), 2)

    # Minimum RR
    min_rr = round(np.min(x), 2)

    # Minimum BPM
    min_bpm = round(np.min(60000/x), 2)

    # Maximum RR
    max_rr = round(np.max(x), 2)

    # Maximum BPM
    max_bpm = round(np.max(60000/x), 2)

    # Standard deviation of R-R intervals
    sdnn = round(x.std(ddof=1), 2)

    # Root Mean Square of Successive Differences (RMSSD)
    rms = round(rmssd(x), 2)

    # NN50: number of successive differences larger than 50ms
    nn = round(nnX(x, t=50), 2)

    # pNN50: Proportion of successive differences larger than 50ms
    pnn = round(pnnX(x, t=50), 2)

    # Create summary dataframe
    values = [mean_rr, mean_bpm, median_rr, median_bpm, min_rr, min_bpm,
              max_rr, max_bpm, sdnn, rms, nn, pnn]
    metrics = ['MeanRR', 'MeanBPM', 'MedianRR', 'MedianBPM', 'MinRR', 'MinBPM',
               'MaxRR', 'MaxBPM', 'SDNN', 'RMSSD', 'nn50', 'pnn50']

    stats = pd.DataFrame({'Values': values, 'Metric': metrics})

    return stats
