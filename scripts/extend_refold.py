"""Decide every recording in a finished corpus again, reading nothing new.

Moved into the package as ``senselab.audio.workflows.triage.drivers.extend_refold``.
This file keeps ``python scripts/extend_refold.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.extend_refold``.
"""

from senselab.audio.workflows.triage.drivers.extend_refold import main

if __name__ == "__main__":
    raise SystemExit(main())
