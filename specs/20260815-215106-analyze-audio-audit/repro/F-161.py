"""Reproduction for F-161 (raised-by B-shadow).

`src/senselab/audio/workflows/audio_analysis/types.py` is named `types`, shadowing the stdlib
`types` module for any Python process whose current working directory is that package directory
(Python 3's implicit relative-import-via-sys.path[0] behavior). This is directly reproduced by
actually invoking Python with a cwd inside `audio_analysis/` and importing `ast` (which pulls in
`weakref`/`_weakrefset`, which import the stdlib `types` module) -- exactly the scenario the
project's own CLAUDE.md warns never to do ("never with cwd inside `audio_analysis/`").

This script itself is run from the repository root (per the rules for every other script here);
it shells out ONE subprocess with a deliberately different cwd to demonstrate the shadowing,
which is the whole point of the finding.

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-161.py
(from the repository root)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
audio_analysis_dir = repo_root / "src/senselab/audio/workflows/audio_analysis"
assert (audio_analysis_dir / "types.py").is_file(), "expected types.py to exist at this path"

result = subprocess.run(
    [sys.executable, "-c", "import ast"],
    cwd=str(audio_analysis_dir),
    capture_output=True,
    text=True,
)

print(f"cwd = {audio_analysis_dir}")
print(f"command: python -c 'import ast'")
print(f"returncode = {result.returncode}")
print(f"stderr tail:\n{result.stderr.strip().splitlines()[-1] if result.stderr.strip() else '(none)'}")

shadowed = (
    result.returncode != 0
    and "GenericAlias" in result.stderr
    and "types.py" in result.stderr
    and "cannot import name" in result.stderr
)

if shadowed:
    print(
        "DEFECT REPRODUCED: running `python -c 'import ast'` with cwd inside "
        "audio_analysis/ fails with an ImportError naming 'types.py' (this package's own "
        "types.py, not the stdlib) and blaming 'weakref' in the traceback rather than the real "
        "cause -- the package-local types.py silently shadows the stdlib types module for any "
        "process launched from that directory. Wrong: importing `ast` should always succeed. "
        "Right: this package must not be named `types`, or every subprocess/script invocation "
        "must run from the repository root (as this reproduction and CLAUDE.md's rule both do)."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
