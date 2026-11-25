"""Bring a finished triage run's derivable PREPROCESS and TAXONOMY outputs up to date, in place.

Moved into the package as ``senselab.audio.workflows.triage.drivers.extend_reprocessed_outputs``.
This file keeps ``python scripts/extend_reprocessed_outputs.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.extend_reprocessed_outputs``.
"""

from senselab.audio.workflows.triage.drivers.extend_reprocessed_outputs import main

if __name__ == "__main__":
    raise SystemExit(main())
