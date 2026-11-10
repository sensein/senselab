"""Check that a corpus's declared identities hold together, before a run or over its corpus table.

Moved into the package as ``senselab.audio.workflows.triage.drivers.check_identity``.
This file keeps ``python scripts/check_identity.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.check_identity``.
"""

from senselab.audio.workflows.triage.drivers.check_identity import main

if __name__ == "__main__":
    raise SystemExit(main())
