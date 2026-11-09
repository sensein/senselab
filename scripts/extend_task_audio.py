"""Cut every finished run's plain, enhanced and released redacted streams to its task extent.

Moved into the package as ``senselab.audio.workflows.triage.drivers.extend_task_audio``.
This file keeps ``python scripts/extend_task_audio.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.extend_task_audio``.
"""

from senselab.audio.workflows.triage.drivers.extend_task_audio import main

if __name__ == "__main__":
    raise SystemExit(main())
