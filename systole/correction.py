# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
from scipy.interpolate import interp1d
from systole.detection import rr_artefacts


def correct_extra(rr, idx):
    """Correct extra beats by removing the RR interval.

    Parameters
    ----------
    rr : 1d array-like
        RR intervals.
    idx : int
        Index of the extra RR interval.

    Returns
    -------
    clean_rr : 1d array-like
        Corrected RR intervals.
    """
    if isinstance(rr, list):
        rr = np.asrray(rr)

    clean_rr = rr

    if idx == len(clean_rr):
        clean_rr = np.delete(clean_rr, idx)
    else:
        # Add the extra interval to the next one
        clean_rr[idx+1] = clean_rr[idx+1] + clean_rr[idx]
        # Remove current interval
        clean_rr = np.delete(clean_rr, idx)

    return clean_rr


def correct_missed(rr, idx):
    """Correct missed beats by adding a new RR interval.

    Parameters
    ----------
    rr : 1d array-like
        RR intervals.
    idx : int
        Index of the missed RR interval.

    Returns
    -------
    clean_rr : 1d array-like
        Corrected RR intervals.
    """
    if isinstance(rr, list):
        rr = np.asrray(rr)

    clean_rr = rr

    # Divide current interval by 2
    clean_rr[idx] /= 2

    # Add a second interval
    clean_rr = np.insert(clean_rr, idx, clean_rr[idx])

    return clean_rr


def interpolate_bads(rr, idx):
    """Correct long and short beats using interpolation.

    Parameters
    ----------
    rr : 1d array-like
        RR intervals (ms).
    idx : int or 1d array-like
        Index of the RR interval to correct.

    Returns
    -------
    clean_rr : 1d array-like
        Corrected RR intervals.
    """
    if isinstance(rr, list):
        rr = np.asrray(rr)

    x = np.arange(0, len(rr))

    # Correction of artefacts
    f = interp1d(np.delete(x, idx), np.delete(rr, idx))
    clean_rr = f(x)

    return clean_rr


def correct_artefacts(rr):
    """Correct long and short beats using interpolation.

    Parameters
    ----------
    rr : 1d array-like
        RR intervals (ms).

    Returns
    -------
    correction : dictionnary
        The corrected RR time series and the number of artefacts corrected:

        * clean_rr: 1d array-like
            The corrected RR time-serie.
        * ectopic: int
            The number of ectopic beats corrected.
        * short: int
            The number of short beats corrected.
        * long: int
            The number of long beats corrcted.
        * extra: int
            The number of extra beats corrected.
        * missed: int
            The number of missed beats corrected.
    """
    if isinstance(rr, list):
        rr = np.asrray(rr)

    clean_rr = rr
    nEctopic, nShort, nLong, nExtra, nMissed = 0, 0, 0, 0, 0

    artefacts = rr_artefacts(clean_rr)

    # Correct missed beats
    if np.any(artefacts['missed']):
        for this_id in np.where(artefacts['missed'])[0]:
            this_id += nMissed
            clean_rr = correct_missed(clean_rr, this_id)
            nMissed += 1

    artefacts = rr_artefacts(clean_rr)

    # Correct extra beats
    if np.any(artefacts['extra']):
        for this_id in np.where(artefacts['extra'])[0]:
            this_id -= nExtra
            clean_rr = correct_missed(clean_rr, this_id)
            nExtra += 1

    artefacts = rr_artefacts(clean_rr)

    # Correct ectopic beats
    if np.any(artefacts['ectopic']):
        # Also correct the beat before
        for i in np.where(artefacts['ectopic'])[0]:
            if (i > 0) & (i < len(artefacts['ectopic'])):
                artefacts['ectopic'][i-1] = True
        this_id = np.where(artefacts['ectopic'])[0]
        clean_rr = interpolate_bads(clean_rr, [this_id])
        nEctopic = np.sum(artefacts['ectopic'])

    # Correct short beats
    if np.any(artefacts['short']):
        this_id = np.where(artefacts['short'])[0]
        clean_rr = interpolate_bads(clean_rr, this_id)
        nShort = len(this_id)

    # Correct long beats
    if np.any(artefacts['long']):
        this_id = np.where(artefacts['long'])[0]
        clean_rr = interpolate_bads(clean_rr, this_id)
        nLong = len(this_id)

    return {'clean_rr': clean_rr, 'ectopic': nEctopic, 'short': nShort,
            'long': nLong, 'extra': nExtra, 'missed': nMissed}
