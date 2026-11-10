"""Compare the per-span SQUIM columns of one ``triage.tsv`` with the task-extent ones of another.

Moved into the package as ``senselab.audio.workflows.triage.drivers.verify_task_squim``.
This file keeps ``python scripts/verify_task_squim.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.verify_task_squim``.
"""

from senselab.audio.workflows.triage.drivers.verify_task_squim import main

if __name__ == "__main__":
    raise SystemExit(main())
