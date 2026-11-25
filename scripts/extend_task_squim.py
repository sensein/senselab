"""Read SQUIM over fixed windows of every finished run's task extent, on the plain stream.

Moved into the package as ``senselab.audio.workflows.triage.drivers.extend_task_squim``.
This file keeps ``python scripts/extend_task_squim.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.extend_task_squim``.
"""

from senselab.audio.workflows.triage.drivers.extend_task_squim import main

if __name__ == "__main__":
    raise SystemExit(main())
