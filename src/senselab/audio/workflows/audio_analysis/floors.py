"""The one floor every withdrawn weight in this system stops at.

A leaf module by construction: it imports nothing from the package, because the modules that
apply the floor (``reliability``, ``support``, ``rounds``, ``influence``) are deliberately pure
and import-free, and a shared constant must not be the thing that inverts their dependency
direction.
"""

from __future__ import annotations

__all__ = ["MIN_EVIDENCE_WEIGHT"]

MIN_EVIDENCE_WEIGHT = 0.05
"""Floor on every weight this system withdraws, so a signal is attenuated rather than erased.

One number with one derivation, because the sites that apply it are applications of a single
argument: the dissenter may be the only source that noticed something. Measured attenuation is
coarse — perturbation stability is sampled over two passes, corroboration over one bucket's
evidence, physical support over one recording's frames — and a hard zero converts a coarse
measurement into a deletion. Recorded after a real recording where a zeroed source turned out to
be the one that matched the spoken speaker names.

The floor is also what keeps erasure from re-entering through configuration: aggregation drops
voters whose weight reaches zero, so any threshold that may be set to zero is a purge switch.
Functions taking a ``floor`` argument reject a non-positive value rather than honouring it.
"""
