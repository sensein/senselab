"""Build every subprocess venv the alternative diarization backends need, one at a time.

Each backend installs its own torch stack, so this is minutes-to-tens-of-minutes and a
failure in one must not stop the others — a backend that cannot be installed on this host
is a result the comparison has to report, not a crash.
"""

from __future__ import annotations

import time
import traceback

from senselab.audio.tasks.speaker_diarization import child_adult, diarizen, moss, nvidia
from senselab.utils.subprocess_venv import ensure_venv

TARGETS = (
    ("sortformer", nvidia._NEMO_VENV, nvidia._NEMO_REQUIREMENTS, nvidia._NEMO_PYTHON),
    ("diarizen", diarizen._DIARIZEN_VENV, diarizen._DIARIZEN_REQUIREMENTS, diarizen._DIARIZEN_PYTHON),
    ("moss", moss._MOSS_VENV, moss._MOSS_REQUIREMENTS, moss._MOSS_PYTHON),
    (
        "child_adult",
        child_adult._CHILD_ADULT_VENV,
        child_adult._CHILD_ADULT_REQUIREMENTS,
        child_adult._CHILD_ADULT_PYTHON,
    ),
)


def main() -> int:
    """Build each venv and print one line per backend saying whether it came up."""
    for label, name, reqs, py in TARGETS:
        print(f"=== {label} venv={name} python={py}", flush=True)
        t0 = time.time()
        try:
            path = ensure_venv(name, reqs, python_version=py)
            print(f"=== {label} OK in {time.time() - t0:.0f}s at {path}", flush=True)
        except Exception:  # noqa: BLE001
            print(f"=== {label} FAILED after {time.time() - t0:.0f}s", flush=True)
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
