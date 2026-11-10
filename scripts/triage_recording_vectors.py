"""Write one compact row per recording from a tree of finished triage stores.

Moved into the package as ``senselab.audio.workflows.triage.drivers.triage_recording_vectors``.
This file keeps ``python scripts/triage_recording_vectors.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.triage_recording_vectors``.
"""

from senselab.audio.workflows.triage.drivers.triage_recording_vectors import main

if __name__ == "__main__":
    raise SystemExit(main())
