"""Machine-readable license records (P-23)."""

import datetime
import re
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import AfterValidator, StrictBool

from senselab_core.kernel._base import Frozen, NonEmptyStr, StrMap


class MissingLicense(LookupError):
    """A license record is required and absent."""


def _check_verified(v: Mapping[str, str]) -> Mapping[str, str]:
    if not {"date", "how"} <= set(v):
        raise ValueError(f"verified needs 'date' and 'how', got {sorted(v)}")
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", v["date"]):
        raise ValueError(f"verified date must be YYYY-MM-DD, got {v['date']!r}")
    datetime.date.fromisoformat(v["date"])
    if not v["how"]:
        raise ValueError("verified 'how' must name the method, such as hf-card, pep639 or manual")
    return v


Verified = Annotated[StrMap, AfterValidator(_check_verified)]
"""Read-only `{"date": "YYYY-MM-DD", "how": <method>, ...}`."""


class License(Frozen):
    """A license record: an SPDX `expression`, its derived flags, where it was read and how it was verified.

    `verified` holds at least `date` (ISO `YYYY-MM-DD`) and `how` (for example `hf-card`, `pep639`,
    `manual`); it is read-only.
    """

    expression: NonEmptyStr
    commercial_use: Literal["yes", "no", "unknown"]
    copyleft: Literal["none", "weak", "strong", "network"]
    attribution: StrictBool
    gated: StrictBool
    source: NonEmptyStr
    verified: Verified


def require_license(record: License | None, subject: str) -> License:
    """Return `record`, or raise `MissingLicense` naming `subject` when it is absent."""
    if record is None:
        raise MissingLicense(f"{subject} has no license record; add one or relax the policy to on_missing: warn")
    return record
