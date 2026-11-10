"""Say what a replay over a finished triage corpus changed, store by store and then corpus-wide.

Moved into the package as ``senselab.audio.workflows.triage.drivers.triage_replay_diff``.
This file keeps ``python scripts/triage_replay_diff.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.triage_replay_diff``.
"""

from senselab.audio.workflows.triage.drivers.triage_replay_diff import main

if __name__ == "__main__":
    raise SystemExit(main())
