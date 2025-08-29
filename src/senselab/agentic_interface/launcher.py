"""This is the launcher for the AI Agent extension."""
import subprocess
import sys

from .senselab_agent.install import install


def senselab_ai():
    # Install the example AI Agent extension
    print("Installing AI Agent Example extension...")
    install(force=True)
    # pass through args to jupyter
    print("Launching Jupyter Lab...")
    sys.exit(subprocess.call(["jupyter", "lab", *sys.argv[1:]]))
