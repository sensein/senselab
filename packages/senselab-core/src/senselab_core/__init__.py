"""Framework-free core of senselab."""

from importlib import metadata


def _require_matching_worker() -> None:
    """Raise ``ImportError`` naming both versions unless senselab-worker is installed at core's own version."""
    core = metadata.version("senselab-core")
    try:
        worker = metadata.version("senselab-worker")
    except metadata.PackageNotFoundError:
        worker = "not installed"
    if worker != core:
        raise ImportError(f"senselab-core {core} needs senselab-worker {core}, found senselab-worker {worker}")


_require_matching_worker()
