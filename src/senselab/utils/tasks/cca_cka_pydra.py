"""This module defines a pydra API for the CCA and CKA tasks."""

import pydra

from senselab.utils.tasks.cca_cka import compute_cca, compute_cka

compute_cca_pt = pydra.mark.task(compute_cca)
compute_cka_pt = pydra.mark.task(compute_cka)
