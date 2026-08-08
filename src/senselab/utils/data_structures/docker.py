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
    - The check is bounded by ``DOCKER_INFO_TIMEOUT_S``. It has to be: on a machine where the
      ``docker`` CLI is installed but no daemon is listening — every macOS GitHub runner — ``docker
      info`` does not fail fast, it blocks on the socket. Unbounded, that turned one skipped
      visualization test into a **5.5-hour hang** that only ended when GitHub cancelled the job at
      its 6-hour ceiling (macOS-tests, run 31218624423). Linux CI never saw it because a daemon
      answers there.
    - Permission/configuration issues (e.g., missing group membership) will
      cause the function to return ``False``.
"""

import shutil
import subprocess

DOCKER_INFO_TIMEOUT_S = 20
"""Seconds to wait for ``docker info``.

Generous rather than tight: a *working* daemon answered in 11.9 s on a warm laptop
(Docker Desktop, macOS), so anything much lower would report a healthy Docker as absent. The point
is only to bound the wait, and a daemon that cannot answer in twenty seconds is not usable by the
callers of this function either.
"""


def docker_is_running() -> bool:
    """Return whether the Docker CLI is present and the daemon is reachable.

    The function returns:
      * ``True``  → ``docker`` is on PATH **and** ``docker info`` succeeds.
      * ``False`` → CLI missing, the daemon is not running/unreachable, the command exits with a
        non-zero status, or it does not answer within ``DOCKER_INFO_TIMEOUT_S``.

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
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=DOCKER_INFO_TIMEOUT_S,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        # A daemon that will not answer is indistinguishable from an absent one *to every caller of
        # this function*, all of which use it to decide whether to skip Docker work. OSError covers
        # the CLI vanishing between the `which` above and this call.
        return False
