"""The `pii_spans` profile slice: the instruction, categories and per-task-family guidance the prompt is built from."""

from functools import cache
from importlib import resources

from pydantic import Field
from senselab_core.kernel.declare import DeclaredParams
from senselab_core.runtime.config import load_yaml

SLICE_FORMAT = "profile-slice@1.0"
SLICE_FILE = "pii_spans.yaml"


class Category(DeclaredParams):
    """One category a span can carry, and what it covers."""

    name: str = Field(min_length=1, pattern=r"^[A-Z][A-Z_]*$")
    description: str = Field(min_length=1)


class Family(DeclaredParams):
    """Guidance for the task families it names; a family matches a name it equals or starts with, then a hyphen."""

    names: tuple[str, ...] = Field(min_length=1)
    guidance: str = Field(min_length=1)


class PiiProfile(DeclaredParams):
    """The slice: `instruction`, the `categories` a span may carry and the `families` with guidance."""

    instruction: str = Field(min_length=1)
    categories: tuple[Category, ...] = Field(min_length=1)
    families: tuple[Family, ...] = ()


@cache
def packaged_profile() -> PiiProfile:
    """Return the packaged slice.

    Raises:
        ValueError: if the file is not a `profile-slice@1.0` of this slice.
    """
    raw = load_yaml(resources.files("senselab").joinpath("data", "profiles", SLICE_FILE), SLICE_FILE)
    if raw.get("format") != SLICE_FORMAT or raw.get("slice") != "pii_spans":
        raise ValueError(f"{SLICE_FILE} is not a {SLICE_FORMAT} of slice pii_spans: {raw.get('format')!r}")
    return PiiProfile.model_validate({k: v for k, v in raw.items() if k not in {"format", "slice"}})
