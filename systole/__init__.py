# Raise a warning if a newer version of Systole is available
from outdated import warn_if_outdated

from .correction import *
from .datasets import *
from .detection import *
from .hrv import *
from .plotly import *
from .plotting import *  # type: ignore
from .utils import *

__version__ = "0.1.3"

warn_if_outdated("systole", __version__)
