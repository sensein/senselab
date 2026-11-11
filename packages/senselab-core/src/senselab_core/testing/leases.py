"""A lease section for tests: the packaged profile's, with the bounds a test sets."""

from senselab_core.runtime.execution_settings import JobLease
from senselab_core.runtime.profiles import ExecutionProfile


def packaged_lease(
    stale_after_s: float, renew_fraction: float, attribute_delay_s: float = 0.0, **keys: float | int | str
) -> JobLease:
    """Return the packaged profile's `lease` with the given stale bound, renewal share and attribute delay.

    `keys` replace any other key of the section, such as `wait_s` or `missed_beats`.
    """
    return ExecutionProfile.packaged().lease.model_copy(
        update={
            "stale_after_s": stale_after_s,
            "renew_fraction": renew_fraction,
            "attribute_delay_s": attribute_delay_s,
            **keys,
        }
    )
