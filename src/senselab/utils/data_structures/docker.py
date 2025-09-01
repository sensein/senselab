"""Utility functions for checking Docker status."""

import shutil
import subprocess


def docker_is_running() -> bool:
    """Return True if Docker CLI exists and daemon is running, else False."""
    if not shutil.which("docker"):
        return False
    try:
        subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False
