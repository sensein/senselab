"""This module defines a pydra API for computing cross correlation between two signals."""

import pydra

from senselab.utils.tasks.cross_correlation import compute_normalized_cross_correlation

compute_normalized_cross_correlation_pt = pydra.mark.task(compute_normalized_cross_correlation)
