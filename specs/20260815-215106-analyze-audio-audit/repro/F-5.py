"""Reproduce F-5: adaptive/provenance.py's module docstring claims "every state change
in a mutually-influencing loop is attributable ... via RevisionRecord/classify_resolution".
An AST sweep over the audio_analysis package's real orchestration files (loop.py,
interventions.py, belief.py) counts zero call sites of either name -- the mechanism is
defined and unit-tested (influence_test.py) but never wired into a real run.

No model load, no network: pure AST parsing of source files already on disk.

Run from the repository root:
    uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-5.py
"""

import ast
import inspect
import sys
from pathlib import Path

import senselab.audio.workflows.audio_analysis.adaptive.provenance as provenance

doc = inspect.getdoc(provenance) or ""
claimed = "Every state change"
assert claimed in doc, "provenance.py module docstring wording changed"
assert "classify_resolution" in doc, "provenance.py module docstring wording changed"
assert hasattr(provenance, "RevisionRecord") and hasattr(provenance, "classify_resolution")

pkg_dir = Path(provenance.__file__).resolve().parent


def call_names(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


orchestration_files = ["loop.py", "interventions.py", "belief.py"]
call_sites = {}
for fname in orchestration_files:
    path = pkg_dir / fname
    tree = ast.parse(path.read_text(), filename=str(path))
    names = call_names(tree)
    hits = {"RevisionRecord", "classify_resolution"} & names
    call_sites[fname] = sorted(hits)

total_calls = sum(len(v) for v in call_sites.values())

print("claimed (provenance.py docstring): every revision is attributed via "
      "RevisionRecord/classify_resolution")
for fname, hits in call_sites.items():
    print(f"actual  call sites in {fname}: {hits or '[]'}")

if total_calls == 0:
    print("DEFECT REPRODUCED: RevisionRecord/classify_resolution have zero call sites across "
          f"{orchestration_files} -- the attribution mechanism the docstring describes is never "
          "invoked by the real loop.")
    sys.exit(0)
else:
    print("NOT REPRODUCED: found live call sites.")
    sys.exit(1)
