# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def rolling_average_ppg(
    signal: np.ndarray,
    sfreq: int = 1000,
    win: float = 0.75,
    moving_average: bool = True,
    moving_average_length: float = 0.05,
    peak_enhancement: bool = True,
    distance: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """A simple systolic peak finder for PPG signals.

    This method uses a rolling average + standard deviation approach to update a
    detection threshold. All the peaks found above this threshold are potential
    systolic peaks.

    Parameters
    ----------
    signal :
        The raw signal recorded from the pulse oximeter time series.
    sfreq :
        The sampling frequency (Hz). Defaults to `1000`.
    win :
        Window size (in seconds) used to compute the threshold (i.e. rolling mean +
        standard deviation).
    moving_average :
        Apply mooving average to remove high frequency noise before peaks detection. The
        length of the time windows can be controlled with `moving_average_length`.
    moving_average_length :
        The length of the window used for moveing average (seconds). Default to `0.05`.
    peak_enhancement :
        If `True` (default), the ppg signal is squared before peaks detection.
    distance :
        The minimum interval between two peaks (seconds).
    verbose :
        Control function verbosity. Defaults to `False` (do not print processing steps).

    Returns
    -------
    peaks_idx :
        Indices of detected systolic peaks.

    Raises
    ------
    ValueError
        If `clipping_thresholds` is not a tuple, a list or `"auto"`.

    Notes
    -----
    This algorithm use a simple rolling average to detect peaks. The signal is
    first resampled and a rolling average is applyed to correct high frequency
    noise and clipping, using method detailled in [1]_. The signal is then
    squared and detection of peaks is performed using threshold corresponding
    to the moving averagte + stadard deviation.

    Examples
    --------
    >>> from systole import import_ppg
    >>> from systole.detection import ppg_peaks
    >>> df = import_ppg()  # Import PPG recording
    >>> signal, peaks = ppg_peaks(signal=df.ppg.to_numpy())
    >>> print(f'{sum(peaks)} peaks detected.')
    378 peaks detected.

    References
    ----------
    .. [1] van Gent, P., Farah, H., van Nes, N. and van Arem, B., 2019.
       Analysing Noisy Driver Physiology Real-Time Using Off-the-Shelf Sensors:
       Heart Rate Analysis Software from the Taking the Fast Lane Project. Journal
       of Open Research Software, 7(1), p.32. DOI: http://doi.org/10.5334/jors.241

    """
    if moving_average is True:
        # Moving average (high frequency noise)
        rolling_noise = max(int(sfreq * moving_average_length), 1)  # 0.05 second
        signal = (
            pd.DataFrame({"signal": signal})
            .rolling(rolling_noise, center=True)
            .mean()
            .signal.to_numpy()
        )
    if peak_enhancement is True:
        # Square signal (peak enhancement)
        signal = (np.asarray(signal) ** 2) * np.sign(signal)

    # Compute moving average and standard deviation
    signal_df = pd.DataFrame({"signal": signal})
    mean_signal = (
        signal_df.rolling(int(sfreq * win), center=True).mean().signal.to_numpy()
    )
    std_signal = (
        signal_df.rolling(int(sfreq * win), center=True).std().signal.to_numpy()
    )

    # Substract moving average + standard deviation
    signal -= mean_signal + std_signal

    # Find positive peaks
    peaks_idx = find_peaks(signal, height=0, distance=int(sfreq * distance))[0]

    return peaks_idx
