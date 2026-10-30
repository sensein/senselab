"""Format tags carried by every persisted artifact."""

import re
from typing import Annotated

from pydantic import StringConstraints, model_validator

from senselab_core.kernel._base import Frozen

FORMAT_PATTERN = r"^[a-z0-9_.-]+@\d+\.\d+$"

FormatTag = Annotated[str, StringConstraints(pattern=FORMAT_PATTERN)]
"""`<name>@<major>.<minor>`."""


def _name_and_major(tag: str) -> tuple[str, str]:
    name, version = tag.split("@")
    return name, version.split(".")[0]


class Formatted(Frozen):
    """A model carrying `format: "<name>@<major>.<minor>"`.

    A subclass declares the tag it writes as the default, `format: FormatTag = "<name>@<major>.<minor>"`,
    and then reads only tags with that name and major version; unknown fields are ignored.
    """

    format: FormatTag

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        """Refuse a subclass whose declared default tag is malformed."""
        super().__pydantic_init_subclass__(**kwargs)
        written = cls.model_fields["format"].default
        if isinstance(written, str) and not re.match(FORMAT_PATTERN, written):
            raise TypeError(f"{cls.__name__} declares a malformed format {written!r}")

    @model_validator(mode="after")
    def _check_readable(self) -> "Formatted":
        written = type(self).model_fields["format"].default
        if isinstance(written, str) and _name_and_major(self.format) != _name_and_major(written):
            raise ValueError(f"{type(self).__name__} reads {written} and its minor versions, got {self.format}")
        return self
