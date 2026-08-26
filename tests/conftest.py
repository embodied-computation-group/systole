"""Shared pytest configuration.

The plotting tests open a large number of Matplotlib figures. Left on an
interactive backend they accumulate live GUI windows, which exhausts Tk
resources partway through a full run and fails with a TclError -- the same
tests pass in isolation. CI is headless in any case, so force the
non-interactive Agg backend and close figures between tests.
"""

import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    """Close any figure a test leaves open."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
