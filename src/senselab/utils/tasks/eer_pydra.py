"""This module defines a pydra API for computing EER."""

import pydra

from senselab.utils.tasks.eer import compute_eer

compute_eer_pt = pydra.mark.task(compute_eer)
