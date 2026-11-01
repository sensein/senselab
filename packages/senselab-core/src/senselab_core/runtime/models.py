"""Model resolution and the weights manifest (design 4.3, P-4, P-16, P-37)."""

import os
import stat
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from types import MappingProxyType

import yaml
from senselab_worker.digest import SHA256_HEX, file_digest
from senselab_worker.hashing import weights_identity

from senselab_core.kernel.identity import ModelRef, ResolvedModel
from senselab_core.runtime.revision import PROCESS_RESOLVER, Resolver, resolve_revision

WEIGHTS_FORMAT = "weights@1.0"


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
    """A model reference cannot be bound to an identity: a missing, empty or unhashable local path."""


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
        if not (isinstance(name, str) and isinstance(digest, str) and SHA256_HEX.fullmatch(digest)):
            raise WeightsManifestError(f"weights.yaml entry {name!r} must map a name to a 64-hex sha256 string")
    return MappingProxyType(dict(weights))


def _load_weights() -> Mapping[str, str]:
    text = resources.files("senselab_core.runtime").joinpath("data", "weights.yaml").read_text(encoding="utf-8")
    return parse_weights(text)


WEIGHTS: Mapping[str, str] = _load_weights()
"""Weight name to the sha256 of that single weight file, loaded once from `weights.yaml` (read-only)."""


def verify_weights(name: str, path: Path, *, weights: Mapping[str, str] = WEIGHTS) -> None:
    """Check that the file at `path` has the sha256 the manifest `weights` records for `name`.

    This is a single-file integrity check: the file's own sha256, streamed in chunks (never read
    whole). It is not the `weights_identity` of a weight set.

    Raises:
        KeyError: if `name` is not in `weights`.
        WeightsMismatch: if the sha256 differs.
    """
    expected = weights[name]
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
        ModelResolutionError: for a directory entry `_files_under` refuses.
    """
    if model.ref.source != "url":
        raise ValueError(f"verify_model checks url models, got {model.ref.source}")
    path = Path(path)
    actual = file_digest("file@1", path) if path.is_file() else weights_identity(_files_under(path), root=path)
    if actual != model.identity:
        raise WeightsMismatch(str(model.ref), model.identity, actual, path)


def _files_under(path: Path) -> list[Path]:
    """Return every regular file reachable under the directory `path`, sorted, named by where it sits.

    Symlinks to files and to directories are followed; a file is named by its path under `path`, links unresolved.

    Raises:
        ModelResolutionError: for an entry that is neither a regular file nor a directory (FIFO, socket,
            device), a dangling symlink, or a directory symlink that leads back to a directory holding it.
    """
    files: list[Path] = []

    def walk(directory: Path, holding: frozenset[tuple[int, int]]) -> None:
        for entry in sorted(os.scandir(directory), key=lambda e: e.name):
            here = Path(entry.path)
            try:
                st = entry.stat(follow_symlinks=True)
            except FileNotFoundError:
                raise ModelResolutionError(f"local model entry {here} is a dangling symlink") from None
            if stat.S_ISREG(st.st_mode):
                files.append(here)
            elif stat.S_ISDIR(st.st_mode):
                if (st.st_dev, st.st_ino) in holding:
                    raise ModelResolutionError(f"local model entry {here} links back to a directory holding it")
                walk(here, holding | {(st.st_dev, st.st_ino)})
            else:
                raise ModelResolutionError(f"local model entry {here} is not a regular file or a directory")

    top = path.stat()
    walk(path, frozenset({(top.st_dev, top.st_ino)}))
    return sorted(files)


def _local_identity(path: Path) -> str:
    try:
        st = path.stat()
    except FileNotFoundError:
        raise ModelResolutionError(f"local model path {path} does not exist") from None
    if stat.S_ISREG(st.st_mode):
        return file_digest("file@1", path)
    if not stat.S_ISDIR(st.st_mode):
        raise ModelResolutionError(f"local model path {path} is not a regular file or a directory")
    files = _files_under(path)
    if files:
        return weights_identity(files, root=path)
    raise ModelResolutionError(f"local model directory {path} holds no files")


def resolve(ref: ModelRef, *, token: str | None = None, resolver: Resolver = PROCESS_RESOLVER) -> ResolvedModel:
    """Bind `ref` to the identity of its weights.

    - hf: the 40-hex commit from `resolve_revision` through `resolver` (its errors propagate).
    - local: the file's `file_digest`, or `weights_identity` of every regular file reachable under the directory
      (symlinks followed), rooted at it. Every file counts, including editor and download metadata.
    - url: the hash declared in the reference (`url:<url>#sha256=<hex>`); nothing is downloaded.
      `verify_model` checks the downloaded bytes against it at load.

    Raises:
        ValueError: for a url reference with no declared hash.
        ModelResolutionError: for a missing local path, an empty directory, or an entry `_files_under` refuses.
    """
    if ref.source == "hf":
        return ResolvedModel(ref=ref, identity=resolve_revision(ref.id, ref.ref, token=token, resolver=resolver))
    if ref.source == "local":
        return ResolvedModel(ref=ref, identity=_local_identity(Path(ref.id)))
    if not ref.ref:
        raise ValueError(f"url model {ref.id!r} needs a declared hash: 'url:{ref.id}#sha256=<64 hex>'")
    return ResolvedModel(ref=ref, identity=ref.ref)
