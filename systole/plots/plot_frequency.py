# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from bokeh.plotting._figure import figure
from matplotlib.axes import Axes
from scipy.interpolate import interp1d

from systole.hrv import psd
from systole.plots.utils import get_plotting_function
from systole.utils import input_conversion


def plot_frequency(
    rr: Union[np.ndarray, list],
    input_type: str = "peaks",
    fbands: Optional[Dict[str, Tuple[str, Tuple[float, float], str]]] = None,
    figsize: Optional[Union[List[int], Tuple[int, int], int]] = None,
    backend: str = "matplotlib",
    ax: Optional[Axes] = None,
    **kwargs
) -> Union[figure, Axes]:
    """Plot power spectral densty of RR time series.

    Parameters
    ----------
    rr :
        Boolean vector of peaks detection or RR intervals.
    input_type :
        The type of input vector. Default is `"peaks"` (a boolean vector where
        `1` represents the occurrence of R waves or systolic peaks). Can also be
        `"rr_s"` or `"rr_ms"` for vectors of RR intervals, or interbeat intervals
        (IBI), expressed in seconds or milliseconds (respectively).
    fbands :
        Dictionary containing the names of the frequency bands of interest (str), their
        range (tuples) and their color in the PSD plot. Default is::

           {
            'vlf': ('Very low frequency', (0.003, 0.04), 'b'),
            'lf': ('Low frequency', (0.04, 0.15), 'g'),
            'hf': ('High frequency', (0.15, 0.4), 'r')
            }

    figsize :
        Figure size. Default is `(13, 5)`.
    ax :
        Where to draw the plot. Default is `None` (create a new figure).
    backend :
        Select plotting backend (`"matplotlib"`, `"bokeh"`). Defaults to `"matplotlib"`.

    Returns
    -------
    plot :
        The matplotlib axes, or the boken figure containing the plot.

    See also
    --------
    plot_events, plot_ectopic, plot_shortlong, plot_subspaces, plot_frequency,
    plot_timedomain, plot_nonlinear

    Examples
    --------

    Visualizing HRV frequency domain from RR time series using Matplotlib as plotting
    backend.

    .. jupyter-execute::

        from systole import import_rr
        from systole.plots import plot_frequency
        # Import PPG recording as numpy array
        rr = import_rr().rr.to_numpy()
        plot_frequency(rr, input_type="rr_ms")

    Visualizing HRV frequency domain from RR time series using Bokeh as plotting
    backend.

    .. jupyter-execute::

        from systole import import_rr
        from systole.plots import plot_frequency
        from bokeh.io import output_notebook
        from bokeh.plotting import show
        output_notebook()

        show(
            plot_frequency(rr, input_type="rr_ms", backend="bokeh")
        )

    """
    # Define figure size
    if figsize is None:
        if backend == "matplotlib":
            figsize = (8, 6)
        elif backend == "bokeh":
            figsize = 600

    if input_type != "rr_ms":
        rr = input_conversion(rr, input_type=input_type, output_type="rr_ms")
    freq, power = psd(rr)

    # Interpolate PSD line for plotting
    f = interp1d(freq, power, kind="cubic")
    freq = np.arange(0.003, 0.4, 0.001)
    power = f(freq)

    # Clip power to avoid values < 0 before plotting
    power = np.clip(power, a_min=0, a_max=None)  # type: ignore

    plot_frequency_args = {
        "freq": freq,
        "power": power,
        "figsize": figsize,
        "fbands": fbands,
        "ax": ax,
    }

    plotting_function = get_plotting_function(
        "plot_frequency", "plot_frequency", backend
    )
    plot = plotting_function(**plot_frequency_args)

    return plot
