"""Triage one recording through the senselab triage graph.

Moved into the package as ``senselab.audio.workflows.triage.drivers.triage_audio``.
This file keeps ``python scripts/triage_audio.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.triage_audio``.
"""

from senselab.audio.workflows.triage.drivers.triage_audio import main

if __name__ == "__main__":
    raise SystemExit(main())
