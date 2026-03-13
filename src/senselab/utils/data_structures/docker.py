"""Docker status utilities.

This module provides a lightweight check to determine whether the Docker CLI
is available **and** the Docker daemon is reachable on the current system.

How it works:
    1) Verifies that the ``docker`` executable exists on the PATH via
       ``shutil.which("docker")``.
    2) Executes ``docker info`` and returns ``True`` **only** if it exits
       successfully (exit code 0). All command output is suppressed.

Notes:
    - This does **not** validate Docker version, Compose availability, or any
      specific container runtime configuration.
    - The check may block briefly while the Docker client contacts the daemon.
    - Permission/configuration issues (e.g., missing group membership) will
      cause the function to return ``False``.
"""

import shutil
import subprocess


def docker_is_running() -> bool:
    """Return whether the Docker CLI is present and the daemon is reachable.

    The function returns:
      * ``True``  → ``docker`` is on PATH **and** ``docker info`` succeeds.
      * ``False`` → CLI missing, the daemon is not running/unreachable, or the
        command exits with a non-zero status.

    Returns:
        bool: ``True`` if Docker appears operational; otherwise ``False``.

    Example:
        >>> from senselab.utils.data_structures.docker import docker_is_running
        >>> if docker_is_running():
        ...     print("Docker is up!")
        ... else:
        ...     print("Docker is not available or the daemon is down.")
    """
    if not shutil.which("docker"):
        return False
    try:
        subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False
