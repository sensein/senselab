"""Run COHORT over finished triage runs: corpus quantiles, then speaker enrollment and readings.

Moved into the package as ``senselab.audio.workflows.triage.drivers.extend_cohort``.
This file keeps ``python scripts/extend_cohort.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.extend_cohort``.
"""

from senselab.audio.workflows.triage.drivers.extend_cohort import main

if __name__ == "__main__":
    raise SystemExit(main())
