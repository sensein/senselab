"""This is the launcher for the AI Agent extension."""
import subprocess
import sys

from .senselab_agent.install import ensure_installed


def senselab_ai():
    # Install the example AI Agent extension
    ensure_installed()
    # pass through args to jupyter
    sys.exit(subprocess.call(["jupyter", "lab", *sys.argv[1:]]))
