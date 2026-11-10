"""Extend finished triage runs with the posteriorgram and Praat blocks, in place.

Moved into the package as ``senselab.audio.workflows.triage.drivers.extend_ppg_praat``.
This file keeps ``python scripts/extend_ppg_praat.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.extend_ppg_praat``.
"""

from senselab.audio.workflows.triage.drivers.extend_ppg_praat import main

if __name__ == "__main__":
    raise SystemExit(main())
