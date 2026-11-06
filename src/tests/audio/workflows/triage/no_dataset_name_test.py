"""No triage test names the dataset its fixture profile was taken from (plan step 8).

The tests run on a fixture profile and a made-up one, recordings carry plain names and declare their identity, so
nothing in them needs the source dataset's name. The tests of the two modules plan step 7 hands to that dataset's
own package are the exception, and so is the fixture's task guidance, which reviewer readings hash byte for byte.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS = Path(__file__).resolve().parents[3]
"""``src/tests``."""

SCANNED = (TESTS / "audio" / "workflows" / "triage", TESTS / "scripts", TESTS / "text")
"""The trees whose tests run triage."""

EXEMPT = {
    TESTS / "audio" / "workflows" / "triage" / "b2ai_hints_test.py",
    TESTS / "audio" / "workflows" / "triage" / "task_registry_test.py",
    TESTS / "audio" / "workflows" / "triage" / "profiles" / "adult_speech_fixture" / "task_guidance.yaml",
    Path(__file__).resolve(),
}
"""The tests of the modules plan step 7 moves out, the byte-hashed task guidance, and this file."""

NAME = re.compile("b2" + "ai|bridge" + "2ai", re.IGNORECASE)
"""The source dataset's name, as any test might spell it."""

PACKAGED = "b2" + "ai_task_registry"
"""The vendored task registry's directory under ``src/senselab``, which a test may name until plan step 7 moves it."""


def test_no_test_names_the_source_dataset() -> None:
    """Every file under the scanned trees, but the exempt ones, is free of the name."""
    named = [
        f"{path.relative_to(TESTS)}:{number}"
        for tree in SCANNED
        for path in sorted(tree.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".txt", ".mjs", ".json"} and path not in EXEMPT
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1)
        if NAME.search(line.replace(PACKAGED, ""))
    ]
    assert named == [], f"{len(named)} lines name the source dataset: {named[:20]}"
