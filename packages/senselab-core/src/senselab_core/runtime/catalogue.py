"""The runtime catalogue: the models a task can name, each pinned by identity and carrying its license (`catalogue@2`).

An entry is `{id, revision, tasks, backends, stacks, license, default_for}`. `revision` is a 40-hex commit or a
64-hex weights digest, never a ref. `license.expression` is an SPDX expression; its flags are derived
(`senselab_core.runtime.licenses`) and a flag written in the file must equal the derived one. `Catalogue.load` takes
YAML text or a `Path`; the packaged catalogue belongs to `senselab`.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import field_validator, model_validator
from senselab_worker.digest import COMMIT_HEX, SHA256_HEX

from senselab_core.kernel._base import Frozen, NonEmptyStr, ignoring_unknown_fields
from senselab_core.kernel.declare import BackendSpec
from senselab_core.kernel.formats import (
    CATALOGUE_FORMAT,
    RETIRED_CATALOGUE_FORMATS,
    FormatTag,
    Formatted,
    FormatVerdict,
    check,
)
from senselab_core.kernel.identity import ModelRef
from senselab_core.kernel.license import License
from senselab_core.runtime.config import load_yaml
from senselab_core.runtime.licenses import license_record

_FLAGS = ("commercial_use", "copyleft", "attribution")


class UnknownModel(KeyError):
    """The catalogue has no such entry or default."""


class CatalogueFormatRetired(ValueError):
    """A catalogue in a retired format; its entries are rewritten as `catalogue@2.0`."""

    def __init__(self, found: str) -> None:
        """Name the declared format."""
        super().__init__(
            f"the catalogue declares the retired format {found}; it is not read. Rewrite it as {CATALOGUE_FORMAT}:"
            " drop each entry's inline resources"
        )
        self.found = found


def _license(raw: Mapping[str, Any]) -> License:
    given = dict(raw)
    try:
        expression, source, verified, gated = (given.pop(k) for k in ("expression", "source", "verified", "gated"))
    except KeyError as err:
        raise ValueError(f"a license needs expression, source, verified and gated; missing {err}") from err
    unknown = set(given) - set(_FLAGS)
    if unknown and not ignoring_unknown_fields():
        raise ValueError(f"a license holds no field {sorted(unknown)}")
    given = {k: v for k, v in given.items() if k not in unknown}
    return license_record(expression, source=source, verified=verified, gated=gated, flags=given)


class CatalogueEntry(Frozen):
    """One model: its hub id, pinned identity, the tasks, backends and stacks that serve it, and its license."""

    id: NonEmptyStr
    revision: NonEmptyStr
    tasks: tuple[NonEmptyStr, ...]
    backends: tuple[NonEmptyStr, ...]
    stacks: tuple[NonEmptyStr, ...]
    license: License
    default_for: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _derive_license(cls, data: Any) -> Any:  # noqa: ANN401
        if isinstance(data, Mapping) and isinstance(data.get("license"), Mapping):
            return {**data, "license": _license(data["license"])}
        return data

    @field_validator("revision")
    @classmethod
    def _check_revision(cls, v: str) -> str:
        if not (COMMIT_HEX.fullmatch(v) or SHA256_HEX.fullmatch(v)):
            raise ValueError(f"revision must be a 40-hex commit or a 64-hex weights digest, never a ref: got {v!r}")
        return v

    @property
    def ref(self) -> ModelRef:
        """The pinned reference `<id>@<revision>` of this entry."""
        return ModelRef.parse(f"{self.id}@{self.revision}")

    @model_validator(mode="after")
    def _check_entry(self) -> "CatalogueEntry":
        _ = self.ref
        if not set(self.default_for) <= set(self.tasks):
            raise ValueError(f"{self.id}: default_for {sorted(set(self.default_for) - set(self.tasks))} not in tasks")
        return self


class Catalogue(Formatted):
    """The entries, in file order."""

    format: FormatTag = CATALOGUE_FORMAT
    entries: tuple[CatalogueEntry, ...]

    @model_validator(mode="after")
    def _check_catalogue(self) -> "Catalogue":
        ids = [e.id for e in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError(f"catalogue ids repeat: {sorted({i for i in ids if ids.count(i) > 1})}")
        defaults = [t for e in self.entries for t in e.default_for]
        if len(set(defaults)) != len(defaults):
            raise ValueError(
                f"a task has two catalogue defaults: {sorted({t for t in defaults if defaults.count(t) > 1})}"
            )
        return self

    @classmethod
    def load(cls, text_or_path: str | Path) -> "Catalogue":
        """Return the catalogue in YAML `text_or_path` (the text, or the file when a `Path`).

        Raises:
            CatalogueFormatRetired: the document declares a retired format.
            ValueError: the document does not validate.
        """
        raw = load_yaml(text_or_path, "catalogue")
        if check(CATALOGUE_FORMAT, raw.get("format"), RETIRED_CATALOGUE_FORMATS) is FormatVerdict.RETIRED:
            raise CatalogueFormatRetired(str(raw.get("format")))
        return cls.model_validate(raw)

    def entry(self, model_id: str) -> CatalogueEntry:
        """Return the entry for `model_id`, or raise `UnknownModel`."""
        for e in self.entries:
            if e.id == model_id:
                return e
        raise UnknownModel(f"{model_id} is not in the catalogue")

    def default(self, task: str) -> CatalogueEntry:
        """Return the entry that is the default model of `task`, or raise `UnknownModel`."""
        for e in self.entries:
            if task in e.default_for:
                return e
        raise UnknownModel(f"the catalogue names no default model for task {task!r}")

    def backend_entry(self, backend: BackendSpec) -> "CatalogueEntry | None":
        """Return the entry `backend` runs: its first declared model, else its task's default, else None.

        Raises:
            UnknownModel: the backend's first declared model is not in the catalogue.
        """
        if backend.models:
            return self.entry(backend.models[0])
        try:
            return self.default(backend.task)
        except UnknownModel:
            return None

    def sources(self) -> tuple[str, ...]:
        """Return the sorted hub organisations of the entries."""
        return tuple(sorted({e.id.split("/")[0] for e in self.entries if "/" in e.id}))
