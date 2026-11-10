"""Give finished triage runs PREPROCESS's own-floor measurement, the first half of the session floor.

Moved into the package as ``senselab.audio.workflows.triage.drivers.extend_background_floor``.
This file keeps ``python scripts/extend_background_floor.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.extend_background_floor``.
"""

from senselab.audio.workflows.triage.drivers.extend_background_floor import main

if __name__ == "__main__":
    raise SystemExit(main())
