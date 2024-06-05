"""This module defines a pydra API for computing cosine similarity."""

import pydra

from senselab.utils.tasks.cosine_similarity import compute_cosine_similarity

cosine_similarity_pt = pydra.mark.task(compute_cosine_similarity)
