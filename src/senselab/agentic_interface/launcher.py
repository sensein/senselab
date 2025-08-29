"""This is the launcher for the AI Agent extension."""
import subprocess
import sys

from .senselab_agent.install import ensure_installed


def senselab_ai():
    # Install the example AI Agent extension
    print("Ensuring AI Agent Example extension is installed...")
    ensure_installed()
    # pass through args to jupyter
    print("Launching Jupyter Lab...")
    sys.exit(subprocess.call(["jupyter", "lab", *sys.argv[1:]]))
