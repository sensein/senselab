"""Extend finished triage runs with the PREPROCESS diarization block, in place.

Moved into the package as ``senselab.audio.workflows.triage.drivers.extend_diarization``.
This file keeps ``python scripts/extend_diarization.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.extend_diarization``.
"""

from senselab.audio.workflows.triage.drivers.extend_diarization import main

if __name__ == "__main__":
    raise SystemExit(main())
