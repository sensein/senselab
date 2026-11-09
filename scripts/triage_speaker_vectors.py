"""Build one speaker embedding per subject from a tree of finished triage runs.

Moved into the package as ``senselab.audio.workflows.triage.drivers.triage_speaker_vectors``.
This file keeps ``python scripts/triage_speaker_vectors.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.triage_speaker_vectors``.
"""

from senselab.audio.workflows.triage.drivers.triage_speaker_vectors import main

if __name__ == "__main__":
    raise SystemExit(main())
