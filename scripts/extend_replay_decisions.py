"""Re-decide a finished triage run from TAXONOMY, over the PREPROCESS output its store holds.

Moved into the package as ``senselab.audio.workflows.triage.drivers.extend_replay_decisions``.
This file keeps ``python scripts/extend_replay_decisions.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.extend_replay_decisions``.
"""

from senselab.audio.workflows.triage.drivers.extend_replay_decisions import main

if __name__ == "__main__":
    raise SystemExit(main())
