"""The servable rule: what a run does with a stored entry its reuse policy admitted."""

from collections.abc import Collection, Sequence
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


def license_admits(run: "RunConfig", recorded: License | None) -> bool:
    """Whether a result whose context records env license `recorded` may be served under `run.licenses`.

    A result whose context records no env license is refused unless `run.licenses` allows unlicensed hits.
    """
    if recorded is None:
        return run.licenses is not None and run.licenses.allow_unlicensed_hits
    return run.licenses is None or run.licenses.refusal("", recorded) is None


def servable(run: "RunConfig", key_digest: str, stored: StoredContext, current: ExecutionContext) -> Servable:
    """Return what `run` does with the entry `stored` for `current`: the license rule, then verify sampling."""
    if not license_admits(run, stored.context.env_license):
        return Servable.REFUSE
    if stored.policy_used is not ReusePolicy.EXACT and sampled(run.verify, key_digest, stored.context, current):
        return Servable.VERIFY
    return Servable.SERVE


def choose(
    ranked: Sequence[StoredContext],
    run: "RunConfig | None",
    key_digest: str,
    current: ExecutionContext,
    exclude: Collection[str],
) -> tuple[StoredContext, Servable] | None:
    """Return the first of `ranked` not in `exclude` with what `run` does with it, or None.

    With no run (`ResultStore.get`) the reuse policy alone decides, and the entry is served.
    """
    for entry in ranked:
        if entry.fingerprint not in exclude:
            return entry, Servable.SERVE if run is None else servable(run, key_digest, entry, current)
    return None
