"""Score triage decisions against the owner's listening labels.

Moved into the package as ``senselab.audio.workflows.triage.drivers.evaluate_task_events``.
This file keeps ``python scripts/evaluate_task_events.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.evaluate_task_events``.
"""

from senselab.audio.workflows.triage.drivers.evaluate_task_events import main

if __name__ == "__main__":
    raise SystemExit(main())
