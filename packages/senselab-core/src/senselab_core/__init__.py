"""Framework-free core of senselab."""

import json
from importlib import metadata


def _is_editable(distribution: metadata.Distribution) -> bool:
    """Whether the distribution is an editable install, per its PEP 610 ``direct_url.json``."""
    text = distribution.read_text("direct_url.json")
    if not text:
        return False
    try:
        return bool(json.loads(text).get("dir_info", {}).get("editable", False))
    except (ValueError, AttributeError):
        return False


def _require_matching_worker() -> None:
    """Raise ``ImportError`` naming both versions when two non-editable installs of core and worker differ.

    An editable workspace install is pinned by the lock, not by version, so it is exempt.
    """
    core = metadata.distribution("senselab-core")
    try:
        worker = metadata.distribution("senselab-worker")
    except metadata.PackageNotFoundError:
        raise ImportError(
            f"senselab-core {core.version} needs senselab-worker {core.version}, which is not installed"
        ) from None
    if _is_editable(core) or _is_editable(worker):
        return
    if worker.version != core.version:
        raise ImportError(
            f"senselab-core {core.version} needs senselab-worker {core.version}, found senselab-worker {worker.version}"
        )


_require_matching_worker()
