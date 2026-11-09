"""No triage test, and no driver under ``scripts/``, names the dataset its fixture profile was taken from (plan step 8).

The tests run on a fixture profile and a made-up one, recordings carry plain names and declare their identity, so
nothing in them needs the source dataset's name. The fixture's task guidance is the exception, which reviewer
readings hash byte for byte.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS = Path(__file__).resolve().parents[3]
"""``src/tests``."""

SCRIPTS = TESTS.parents[1] / "scripts"
"""The repository's ``scripts/``: the drivers take every dataset value from the profile they are given."""

SCANNED = (TESTS / "audio" / "workflows" / "triage", TESTS / "scripts", TESTS / "text", SCRIPTS)
"""The trees whose tests run triage, and the drivers they run."""

EXEMPT = {
    TESTS / "audio" / "workflows" / "triage" / "profiles" / "adult_speech_fixture" / "task_guidance.yaml",
    Path(__file__).resolve(),
}
"""The byte-hashed task guidance, and this file."""

NAME = re.compile("b2" + "ai|bridge" + "2ai", re.IGNORECASE)
"""The source dataset's name, as any test might spell it."""


def test_no_test_names_the_source_dataset() -> None:
    """Every file under the scanned trees, but the exempt ones, is free of the name."""
    named = [
        f"{path.relative_to(TESTS.parents[1])}:{number}"
        for tree in SCANNED
        for path in sorted(tree.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".txt", ".mjs", ".json"} and path not in EXEMPT
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1)
        if NAME.search(line)
    ]
    assert named == [], f"{len(named)} lines name the source dataset: {named[:20]}"
