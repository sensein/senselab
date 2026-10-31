"""The first-party top-level packages of the workspace, discovered from the source trees."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def first_party_packages() -> list[str]:
    """Every top-level package under ``src/`` and ``packages/*/src/``, sorted."""
    roots = [REPO / "src", *sorted(REPO.glob("packages/*/src"))]
    return sorted(
        {
            p.name
            for root in roots
            for p in root.iterdir()
            if p.is_dir() and ((p / "__init__.py").exists() or (p / "__init__.pyi").exists())
        }
    )
