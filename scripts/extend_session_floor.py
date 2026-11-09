"""Write SESSION's floor into finished triage runs, one session at a time.

Moved into the package as ``senselab.audio.workflows.triage.drivers.extend_session_floor``.
This file keeps ``python scripts/extend_session_floor.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.extend_session_floor``.
"""

from senselab.audio.workflows.triage.drivers.extend_session_floor import main

if __name__ == "__main__":
    raise SystemExit(main())
