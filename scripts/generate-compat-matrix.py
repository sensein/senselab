#!/usr/bin/env python3
"""Generate docs/compatibility-matrix.md from the compatibility module."""

from pathlib import Path

from senselab.utils.compatibility import generate_matrix_markdown

output = Path(__file__).parent.parent / "docs" / "compatibility-matrix.md"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(generate_matrix_markdown())
print(f"Generated {output}")
