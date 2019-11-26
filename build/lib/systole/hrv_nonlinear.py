# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import pandas as pd


def nonlinear_domain(x):
    """Extract the frequency domain features of heart rate variability.

    Parameters
    ----------
    x : list or numpy array
        Length of R-R intervals (in miliseconds).

    Returns
    -------
    stats : pandas DataFrame
        DataFrame of HRV parameters (frequency domain)
    """

    diff_rr = np.diff(x)
    sd1 = np.sqrt(np.std(diff_rr, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(x, ddof=1) ** 2 - 0.5 * np.std(diff_rr,
                                                            ddof=1) ** 2)
    values = [sd1, sd2]
    metrics = ['SD1', 'SD2']

    stats = pd.DataFrame({'Values': values, 'Metric': metrics})

    return stats
