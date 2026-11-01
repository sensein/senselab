"""Ask a small model to review a pull request diff against the 2.x principles and engineering checklist.

Reads the diff from the file named by ``DIFF_FILE`` and the checklist from ``CHECKLIST_FILES`` (colon separated),
calls the Anthropic Messages API with ``ANTHROPIC_API_KEY``, and writes the comment to ``COMMENT_FILE``.
Any failure writes a comment saying so and exits 0: the review is advice and never blocks.
"""

import json
import os
import urllib.request
from pathlib import Path

MODEL = "claude-haiku-4-5"
DIFF_LIMIT = 150_000
PROMPT = (
    "You review a pull request against the project's principles and engineering checklist, given first. "
    "Report only departures the diff itself shows, each with the file, the principle it departs from and a one-line "
    "fix. Skip style that a linter catches. If the diff departs from none, say so in one line. "
    "Be brief: at most ten items."
)


def review(diff: str, checklist: str) -> str:
    """Return the model's review of ``diff`` against ``checklist``."""
    body = {
        "model": MODEL,
        "max_tokens": 1500,
        "system": PROMPT,
        "messages": [{"role": "user", "content": f"CHECKLIST\n{checklist}\n\nDIFF\n{diff[:DIFF_LIMIT]}"}],
    }
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        return "".join(block.get("text", "") for block in json.load(response)["content"])


def main() -> None:
    """Write the review, or the reason there is none, to ``COMMENT_FILE``."""
    try:
        diff = Path(os.environ["DIFF_FILE"]).read_text()
        checklist = "\n\n".join(Path(p).read_text() for p in os.environ["CHECKLIST_FILES"].split(":"))
        text = review(diff, checklist)
    except Exception as exc:  # noqa: BLE001
        text = f"The prose review could not run: {type(exc).__name__}."
    Path(os.environ["COMMENT_FILE"]).write_text(f"**Prose review** (advisory, `{MODEL}`)\n\n{text}\n")


if __name__ == "__main__":
    main()
