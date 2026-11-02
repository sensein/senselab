"""Readers of the checkout's spec files that the status page shows: tables, bullets and the phase row."""

import re
from pathlib import Path

SPEC_DIR = "specs/20261030-140022-senselab-2x-api"
_CELL = re.compile(r"(?<!\\)\|")


def checkout_root() -> Path | None:
    """Return the checkout above this package that holds the 2.x specs, or None, as in an installed wheel."""
    for parent in Path(__file__).resolve().parents:
        if (parent / SPEC_DIR).is_dir():
            return parent
    return None


def read_spec(name: str, root: Path | None = None) -> str:
    """Return the text of spec file `name`, or "" when the checkout or the file is absent."""
    root = root or checkout_root()
    path = None if root is None else root / SPEC_DIR / name
    return path.read_text("utf-8") if path is not None and path.is_file() else ""


def tables(text: str) -> list[list[list[str]]]:
    """Return each markdown table of `text` as its rows of cells, header first, separator row dropped."""
    found: list[list[list[str]]] = []
    block: list[list[str]] = []
    for line in [*text.splitlines(), ""]:
        if line.startswith("|"):
            cells = [c.strip().replace("\\|", "|") for c in _CELL.split(line.strip())[1:-1]]
            if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                block.append(cells)
        elif block:
            found.append(block)
            block = []
    return found


def table_after(text: str, heading: str, first_cell: str | None = None) -> list[dict[str, str]]:
    """Return the rows, as header-keyed dicts, of the first table after the line starting with `heading`.

    With `first_cell`, the first table whose header's first cell is that.
    """
    start = re.search(rf"^{re.escape(heading)}", text, re.MULTILINE)
    if start is None:
        return []
    for rows in tables(text[start.start() :]):
        if first_cell is None or rows[0][0] == first_cell:
            return [dict(zip(rows[0], row, strict=False)) for row in rows[1:]]
    return []


def ledger_marks(ledger: str) -> list[dict[str, str]]:
    """Return the migration ledger's per-area marks: `Area (1.x paths)`, `Where`, `Mark`, `Reconciled changes ...`."""
    return table_after(ledger, "| Area (1.x paths)")


def moved_in_phase(ledger: str, phase: int) -> list[dict[str, str]]:
    """Return the ledger's rows of 1.x modules moved in Phase `phase`; empty when it has no such section."""
    return table_after(ledger, f"## Moved in Phase {phase}")


def open_register_rows(register: str) -> list[dict[str, str]]:
    """Return the review register's rows whose status begins `open` or `scheduled`."""
    rows = next((t for t in tables(register) if t[0][:2] == ["Id", "Src"]), [])
    keyed = (dict(zip(rows[0], r, strict=False)) for r in rows[1:]) if rows else iter(())
    return [r for r in keyed if r.get("Status", "").lower().startswith(("open", "scheduled"))]


_ID = re.compile(
    r"\s*\(§[\d.]+\)|\s*\((?:[^()]*\b(?:ADR|[PRHD])-?\d[^()]*)\)|\b(?:ADR|P2|[PRHD])-\d+\w*\b|\bH\d+[a-z]?\b"
)


def scrub(text: str) -> str:
    """Return `text` without register, principle, hardening and ADR ids, and without the parentheses that held them."""
    return " ".join(_ID.sub("", text).split())


def phase_exits(design: str, phase: int) -> list[str]:
    """Return the exit bullets of Phase `phase` in design section 6.2, each as one line."""
    block = re.search(rf"^\*\*Phase {phase}:.*?(?=^\*\*Phase \d|^###|\Z)", design, re.MULTILINE | re.DOTALL)
    exits = re.search(r"^- Exit:\n(.*)", block.group(0), re.MULTILINE | re.DOTALL) if block else None
    if exits is None:
        return []
    bullets = re.findall(r"^ {2}- (.*?)(?=^ {2}- |\Z)", exits.group(1), re.MULTILINE | re.DOTALL)
    return [scrub(b).rstrip(";.") for b in bullets]


def consolidation_inventory(design: str) -> list[dict[str, str]]:
    """Return the design's consolidation inventory (P-37): `Behaviour`, `Copies today`, `One implementation in 2.x`."""
    return table_after(design, "**Consolidation inventory", "Behaviour")


def phase_states(state: str) -> dict[str, str]:
    """Return the phase table of the working state file as `{phase: done | in progress | not started}`."""
    rows = next((t for t in tables(state) if t[0][:2] == ["Phase", "State"]), [])
    return {r[0]: _phase_word(r[1]) for r in rows[1:] if len(r) > 1}


def _phase_word(cell: str) -> str:
    lowered = cell.lower()
    if "not started" in lowered:
        return "not started"
    return "done" if lowered.startswith("done") or re.search(r"phase \d+ done", lowered) else "in progress"


def open_by_phase(register: str) -> dict[str, int]:
    """Return the count of open register rows by the phase that owns them, `unassigned` when a row names none."""
    counts: dict[str, int] = {}
    for row in open_register_rows(register):
        owner = re.search(r"owner: ([^;,(]*?Phase \d|first 2\.x release)", row["Status"])
        key = re.sub(r".*(Phase \d)", r"\1", owner.group(1)) if owner else "unassigned"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
