"""Withdraw the clip spans a finished triage run's own amplitudes contradict, in place.

Moved into the package as ``senselab.audio.workflows.triage.drivers.extend_withdraw_clips``.
This file keeps ``python scripts/extend_withdraw_clips.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.extend_withdraw_clips``.
"""

from senselab.audio.workflows.triage.drivers.extend_withdraw_clips import main

if __name__ == "__main__":
    raise SystemExit(main())
