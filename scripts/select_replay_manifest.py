"""Write the replay manifest: every recording of the AIRWAY families, and every recording whose run is incomplete.

Moved into the package as ``senselab.audio.workflows.triage.drivers.select_replay_manifest``.
This file keeps ``python scripts/select_replay_manifest.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.select_replay_manifest``.
"""

from senselab.audio.workflows.triage.drivers.select_replay_manifest import main

if __name__ == "__main__":
    raise SystemExit(main())
