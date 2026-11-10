"""Give a finished triage run the QUALITY verdict its graph pass never reached, in place.

Moved into the package as ``senselab.audio.workflows.triage.drivers.extend_quality``.
This file keeps ``python scripts/extend_quality.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.extend_quality``.
"""

from senselab.audio.workflows.triage.drivers.extend_quality import main

if __name__ == "__main__":
    raise SystemExit(main())
