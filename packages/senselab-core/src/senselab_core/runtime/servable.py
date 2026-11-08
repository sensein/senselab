"""The servable rule: what a run does with a stored entry its reuse policy admitted.

`ResultStore.admits` (probing) and `execute` (serving) both apply `servable`, so the license and verify-sampling
rules exist once.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from senselab_core.kernel.identity import ExecutionContext, ReusePolicy
from senselab_core.kernel.license import License
from senselab_core.runtime.equivalence import sampled

if TYPE_CHECKING:
    from senselab_core.runtime.call import RunConfig


@dataclass(frozen=True)
class StoredContext:
    """A stored entry a run's reuse rules admit: its fingerprint, its context and the policy that admitted it."""

    fingerprint: str
    context: ExecutionContext
    policy_used: ReusePolicy


class Servable(Enum):
    """What a run does with a hit: `serve` it, `verify` it (recompute and compare), or `refuse` it."""

    SERVE = "serve"
    VERIFY = "verify"
    REFUSE = "refuse"


def license_admits(run: "RunConfig | None", recorded: License | None) -> bool:
    """Whether a result whose context records env license `recorded` may be served under `run.licenses`.

    A result whose context records no env license is refused unless `run.licenses` allows unlicensed hits.
    """
    licenses = None if run is None else run.licenses
    if recorded is None:
        return licenses is not None and licenses.allow_unlicensed_hits
    return licenses is None or licenses.refusal("", recorded) is None


def servable(run: "RunConfig | None", key_digest: str, stored: StoredContext, current: ExecutionContext) -> Servable:
    """Return what `run` does with the entry `stored` for `current`: the license rule, then verify sampling.

    This is the one rule serving (`execute`) and probing (`ResultStore.admits`) share. Without a run, any entry
    the reuse policy admitted is served.
    """
    if run is None:
        return Servable.SERVE
    if not license_admits(run, stored.context.env_license):
        return Servable.REFUSE
    if stored.policy_used is not ReusePolicy.EXACT and sampled(run.verify, key_digest, stored.context, current):
        return Servable.VERIFY
    return Servable.SERVE
