"""This is the launcher for the AI Agent extension."""

import subprocess
import sys

from .senselab_agent.install import ensure_installed


def senselab_ai() -> None:
    """Launch the senselab AI agent.

    Ensures the AI Agent extension is installed and ready to use.
    Then, launches Jupyter Lab with the appropriate arguments.
    """
    # Install the example AI Agent extension
    ensure_installed()
    # pass through args to jupyter
    sys.exit(subprocess.call(["jupyter", "lab", *sys.argv[1:]]))
