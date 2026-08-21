#!/usr/bin/env python3
"""Generate the per-function dependency table from the compatibility module.

The path is ``senselab.utils.compatibility.GENERATED_DOC``. It is deliberately not
``docs/compatibility-matrix.md``, which is a hand-maintained document this script must not touch.
"""

from pathlib import Path

from senselab.utils.compatibility import GENERATED_DOC, generate_matrix_markdown

output = Path(__file__).parent.parent / GENERATED_DOC
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(generate_matrix_markdown())
print(f"Generated {output}")
