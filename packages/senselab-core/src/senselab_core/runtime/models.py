"""Model resolution and the weights manifest (design 4.3, P-4, P-16, P-37)."""

import os
import re
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from types import MappingProxyType

import yaml
from senselab_worker.hashing import sha256_file, weights_identity

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


def _load_weights() -> Mapping[str, str]:
    text = resources.files("senselab_core.runtime").joinpath("data", "weights.yaml").read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict) or doc.get("format") != WEIGHTS_FORMAT:
        raise ValueError(f"weights.yaml must carry format {WEIGHTS_FORMAT!r}")
    weights = doc["weights"]
    bad = [k for k, v in weights.items() if not (isinstance(v, str) and _SHA256.fullmatch(v))]
    if bad:
        raise ValueError(f"weights.yaml entries must be 64-hex sha256 strings: {bad}")
    return MappingProxyType({str(k): v for k, v in weights.items()})


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
    actual = sha256_file(path)
    if actual != expected:
        raise WeightsMismatch(name, expected, actual, Path(path))


def _local_identity(path: Path) -> str | None:
    if path.is_file():
        return weights_identity([path], root=path.parent)
    if path.is_dir():
        files = sorted(Path(d) / f for d, _, names in os.walk(path) for f in names)
        return weights_identity(files, root=path) if files else None
    return None


def resolve(ref: ModelRef, *, token: str | None = None) -> ResolvedModel:
    """Bind `ref` to the identity of its weights.

    - hf: the 40-hex commit from `resolve_revision` (its errors propagate).
    - local: `weights_identity` of the file, or of every file under the directory, rooted at it.
    - url: its bytes are not fetched here, so it is unresolved with a reason.
    - a missing local path or an empty directory: unresolved with a reason.
    """
    if ref.source == "hf":
        return ResolvedModel(ref=ref, identity=resolve_revision(ref.id, ref.ref, token=token))
    if ref.source == "local":
        identity = _local_identity(Path(ref.id))
        if identity is not None:
            return ResolvedModel(ref=ref, identity=identity)
        return ResolvedModel(ref=ref, identity=str(ref), unresolved_reason=f"no weight files at {ref.id}")
    return ResolvedModel(
        ref=ref, identity=str(ref), unresolved_reason="url weights are not fetched, so their bytes are not hashed"
    )
