"""Check that a derivatives copy holds what its source tree holds: one threaded pass per tree.

Moved into the package as ``senselab.audio.workflows.triage.drivers.verify_derivatives_copy``.
This file keeps ``python scripts/verify_derivatives_copy.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.verify_derivatives_copy``.
"""

from senselab.audio.workflows.triage.drivers.verify_derivatives_copy import main

if __name__ == "__main__":
    raise SystemExit(main())
