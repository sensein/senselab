"""Measure the session background against the session floor over a sample of finished runs, read-only.

Moved into the package as ``senselab.audio.workflows.triage.drivers.measure_session_background``.
This file keeps ``python scripts/measure_session_background.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.measure_session_background``.
"""

from senselab.audio.workflows.triage.drivers.measure_session_background import main

if __name__ == "__main__":
    raise SystemExit(main())
