""".. include:: ./doc.md"""  # noqa: D415

from .api import enhance_audios  # noqa: F401
from .residual import (  # noqa: F401
    ResidualComputation,
    align,
    band_energy_fractions,
    compute_residual,
    correlation,
    find_lag,
    fit_gain,
)

__all__ = [
    "enhance_audios",
    "ResidualComputation",
    "align",
    "band_energy_fractions",
    "compute_residual",
    "correlation",
    "fit_gain",
    "find_lag",
]
