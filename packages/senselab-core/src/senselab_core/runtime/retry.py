"""Retry of a call on a transient fault (`faults.is_transient`), under an explicit `RetryPolicy`."""

import logging
import time
from collections.abc import Callable
from typing import Annotated, Self, TypeVar

from pydantic import Field

from senselab_core.kernel._base import Frozen
from senselab_core.runtime.faults import is_transient
from senselab_core.runtime.storage import load_policy_values

logger = logging.getLogger("senselab")

RETRY_POLICY_FORMAT = "retry-policy@1.0"

_T = TypeVar("_T")


class RetryPolicy(Frozen):
    """How often a call that raised a transient fault is tried, and how long to wait first.

    Attributes:
        attempts: Tries in all, the first included.
        first_backoff_s: Seconds before the second try; each further wait doubles.
    """

    attempts: Annotated[int, Field(strict=True, ge=1)]
    first_backoff_s: Annotated[float, Field(ge=0)]

    @classmethod
    def load_default(cls) -> Self:
        """Return the policy in `data/retry_policy.yaml`.

        Raises:
            ValueError: if the file lacks the format, an entry, or an entry's derivation.
        """
        return cls.model_validate(load_policy_values("senselab_core.runtime", "retry_policy.yaml", RETRY_POLICY_FORMAT))


def retry_on_transient_error(
    fn: Callable[[], _T], policy: RetryPolicy, *, sleep: Callable[[float], None] = time.sleep
) -> _T:
    """Return `fn()`, trying again after a transient fault while `policy` allows.

    The waits are `policy.first_backoff_s`, then doubling. A permanent fault, or a transient one on
    the last attempt, is raised as is.
    """
    for attempt in range(policy.attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt == policy.attempts - 1 or not is_transient(exc):
                raise
            wait = policy.first_backoff_s * 2**attempt
            logger.warning(
                "Transient error on attempt %d/%d: %s. Retrying in %ss...", attempt + 1, policy.attempts, exc, wait
            )
            sleep(wait)
    raise AssertionError("unreachable: RetryPolicy.attempts is at least 1")
