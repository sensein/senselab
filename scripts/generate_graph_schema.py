"""Generate `runtime/data/graph@1.0.schema.json`, the published JSON Schema of a graph document.

uv run python scripts/generate_graph_schema.py          # write the schema
uv run python scripts/generate_graph_schema.py --check  # exit 1 if the committed schema differs
"""

import argparse
import json
import sys
from pathlib import Path

from senselab_core.kernel.formats import GRAPH_FORMAT
from senselab_core.kernel.graph import graph_schema

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "senselab-core"
    / "src"
    / "senselab_core"
    / "runtime"
    / "data"
    / f"{GRAPH_FORMAT}.schema.json"
)


def render() -> str:
    """Return the text of the published schema: `graph_schema()` as 4-space indented JSON."""
    return json.dumps(graph_schema(), indent=4) + "\n"


def is_current(output: Path) -> bool:
    """Whether `output` holds exactly the schema `render` writes."""
    return output.exists() and output.read_text() == render()


def main(argv: list[str] | None = None) -> int:
    """Write the schema, or with `--check` compare it with the committed one."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the schema differs")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="the schema file to write or check")
    args = parser.parse_args(argv)
    if args.check:
        if not is_current(args.output):
            print(f"{args.output} is out of date; run scripts/generate_graph_schema.py", file=sys.stderr)
            return 1
        return 0
    args.output.write_text(render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
