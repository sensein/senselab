"""Model resolution and the weights manifest (design 4.3, P-4, P-16, P-37)."""

import os
import re
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from types import MappingProxyType

import yaml
from senselab_worker.digest import file_digest
from senselab_worker.hashing import weights_identity

from senselab_core.kernel.identity import ModelRef, ResolvedModel
from senselab_core.runtime.revision import resolve_revision

WEIGHTS_FORMAT = "weights@1.0"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class WeightsMismatch(RuntimeError):
    """A weight file's sha256 differs from the manifest entry for `name`."""

    def __init__(self, name: str, expected: str, actual: str, path: Path) -> None:
        """Record the manifest `name`, the `expected` and `actual` sha256 and the file `path`."""
        super().__init__(f"weights {name!r} at {path}: sha256 {actual} != manifest {expected}")
        self.name = name
        self.expected = expected
        self.actual = actual
        self.path = path


class WeightsManifestError(ValueError):
    """`weights.yaml` is malformed; the message names the offending entry."""


class ModelResolutionError(RuntimeError):
    """A model reference cannot be bound to an identity: a missing or empty local path."""


def parse_weights(text: str) -> Mapping[str, str]:
    """Parse the text of a `weights@1.0` manifest into a read-only name-to-sha256 mapping.

    Raises:
        WeightsManifestError: on a wrong format, a missing or non-mapping `weights`, or an entry
            that is not a 64-hex sha256 string.
    """
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict) or doc.get("format") != WEIGHTS_FORMAT:
        raise WeightsManifestError(f"weights.yaml must carry format {WEIGHTS_FORMAT!r}")
    weights = doc.get("weights")
    if not isinstance(weights, dict):
        raise WeightsManifestError("weights.yaml needs a 'weights' mapping of name to sha256")
    for name, digest in weights.items():
        if not (isinstance(name, str) and isinstance(digest, str) and _SHA256.fullmatch(digest)):
            raise WeightsManifestError(f"weights.yaml entry {name!r} must map a name to a 64-hex sha256 string")
    return MappingProxyType(dict(weights))


def _load_weights() -> Mapping[str, str]:
    text = resources.files("senselab_core.runtime").joinpath("data", "weights.yaml").read_text(encoding="utf-8")
    return parse_weights(text)


WEIGHTS: Mapping[str, str] = _load_weights()
"""Weight name to the sha256 of that single weight file, loaded once from `weights.yaml` (read-only)."""


def verify_weights(name: str, path: Path) -> None:
    """Check that the file at `path` has the sha256 the manifest records for `name`.

    This is a single-file integrity check: the file's own sha256, streamed in chunks (never read
    whole). It is not the `weights_identity` of a weight set.

    Raises:
        KeyError: if `name` is not in `WEIGHTS`.
        WeightsMismatch: if the sha256 differs.
    """
    expected = WEIGHTS[name]
    actual = file_digest("file@1", path)
    if actual != expected:
        raise WeightsMismatch(name, expected, actual, Path(path))


def verify_model(model: ResolvedModel, path: Path) -> None:
    """Check downloaded bytes of a url model against the hash its reference declared.

    A single file is checked by its own sha256 (the pip-style `#sha256=` convention); a directory
    by `weights_identity` of the files under it.

    Raises:
        ValueError: if `model` is not a url model.
        WeightsMismatch: if the bytes do not hash to the declared value.
    """
    if model.ref.source != "url":
        raise ValueError(f"verify_model checks url models, got {model.ref.source}")
    path = Path(path)
    actual = file_digest("file@1", path) if path.is_file() else weights_identity(_files_under(path), root=path)
    if actual != model.identity:
        raise WeightsMismatch(str(model.ref), model.identity, actual, path)


def _files_under(path: Path) -> list[Path]:
    return sorted(Path(d) / f for d, _, names in os.walk(path) for f in names)


def _local_identity(path: Path) -> str:
    if path.is_file():
        return weights_identity([path], root=path.parent)
    if path.is_dir():
        files = _files_under(path)
        if files:
            return weights_identity(files, root=path)
        raise ModelResolutionError(f"local model directory {path} holds no files")
    raise ModelResolutionError(f"local model path {path} does not exist")


def resolve(ref: ModelRef, *, token: str | None = None) -> ResolvedModel:
    """Bind `ref` to the identity of its weights.

    - hf: the 40-hex commit from `resolve_revision` (its errors propagate).
    - local: `weights_identity` of the file, or of every file under the directory, rooted at it.
    - url: the hash declared in the reference (`url:<url>#sha256=<hex>`); nothing is downloaded.
      `verify_model` checks the downloaded bytes against it at load.

    Raises:
        ValueError: for a url reference with no declared hash.
        ModelResolutionError: for a missing local path or an empty directory.
    """
    if ref.source == "hf":
        return ResolvedModel(ref=ref, identity=resolve_revision(ref.id, ref.ref, token=token))
    if ref.source == "local":
        return ResolvedModel(ref=ref, identity=_local_identity(Path(ref.id)))
    if not ref.ref:
        raise ValueError(f"url model {ref.id!r} needs a declared hash: 'url:{ref.id}#sha256=<64 hex>'")
    return ResolvedModel(ref=ref, identity=ref.ref)
