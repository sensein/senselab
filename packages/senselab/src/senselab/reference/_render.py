"""Markdown of the reference and status pages, from the declarations, the catalogue and the checkout's specs."""

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

import yaml
from senselab_core.kernel.declare import REGISTRY, Registry
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.fetch import DataManifest
from senselab_core.runtime.tolerance import tolerance_profile_path

from senselab._api.describe import describe
from senselab._api.packaged import CATALOGUE
from senselab.reference import _specs

BADGES = {
    "2x": '<span class="badge badge-2x">available in 2.x</span>',
    "1x": '<span class="badge badge-1x">1.x only</span>',
    "planned": '<span class="badge badge-planned">planned</span>',
}
"""The status badge of a page or an area, by kind."""

_NONE = "none yet"


def _cell(value: object, limit: int | None = None) -> str:
    text = " ".join(str(value).split()).replace("|", "\\|")
    return text if limit is None or len(text) <= limit else text[: limit - 1].rstrip() + "…"


def table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Return a markdown table, or the line `none yet` when `rows` is empty."""
    body = [f"| {' | '.join(_cell(c) for c in row)} |" for row in rows]
    if not body:
        return _NONE
    return "\n".join([f"| {' | '.join(header)} |", f"|{'---|' * len(header)}", *body])


def _type(prop: Mapping[str, Any]) -> str:
    if "enum" in prop:
        return " or ".join(f"`{v}`" for v in prop["enum"])
    if "anyOf" in prop:
        return " or ".join(_type(p) for p in prop["anyOf"])
    kind = str(prop.get("type", "any"))
    bounds = ", ".join(f"{k} {prop[k]}" for k in ("minimum", "maximum") if k in prop)
    return f"{kind} ({bounds})" if bounds else kind


def _params(schema: Mapping[str, Any]) -> str:
    rows = [
        (f"`{name}`", _type(prop), f"`{prop.get('default', 'required')}`", prop.get("description", ""))
        for name, prop in schema.get("properties", {}).items()
    ]
    return table(["Param", "Type", "Default", "Meaning"], rows) if rows else "no params"


def _ports(schema: Mapping[str, Any]) -> str:
    required = set(schema.get("required", ()))
    rows = [
        (f"`{name}`", prop.get("$ref", "").rsplit("/", 1)[-1] or "path or URI", "yes" if name in required else "no")
        for name, prop in schema["properties"].items()
    ]
    return table(["Port", "Type", "Required"], rows)


def _model_row(model: Mapping[str, Any]) -> tuple[object, ...]:
    lic = model["license"]
    return (f"`{model['id']}`", f"`{model['revision'][:12]}`", ", ".join(model["backends"]), lic["expression"])


def _task_section(name: str, registry: Registry, catalogue: Catalogue) -> str:
    d = describe(name, registry, catalogue)
    function = d["callable"]["function"]
    backends = table(
        ["Backend", "Version", "Stack", "Devices"],
        [(f"`{b['name']}`", b["version"], b["stack"], ", ".join(b["devices"])) for b in d["backends"]],
    )
    backend_params = "\n\n".join(
        f"`{b}` backend params\n\n{_params(schema)}" for b, schema in d["backend_params"].items()
    )
    models = table(["Model", "Revision", "Backends", "License"], [_model_row(m) for m in d["models"]])
    default = f"`{d['default_model']}`" if d["default_model"] else "none"
    return f"""### `{name}`

Version {d["version"]}, kind {d["kind"]}. Default model: {default}.

Inputs

{_ports(d["inputs"])}

Outputs

{_ports(d["outputs"])}

Params

{_params(d["params"])}

{backend_params}

Backends are code paths declared with the task.

{backends}

Models are catalogue entries.

{models}

Call it as a function, `senselab.{function}(...)`, a class, `senselab.{d["callable"]["class"]}(...)`, or as the
JSON call `{{"task": "{name}", "inputs": ..., "params": ..., "model": ..., "backend_params": ...}}`;
`senselab.describe("{name}")` returns all of the above as JSON Schema.
"""


def tasks_page(registry: Registry = REGISTRY, catalogue: Catalogue | None = None) -> str:
    """Return the task reference: one section per declared task, from `describe`."""
    catalogue = catalogue or CATALOGUE
    described = {s.name: describe(s.name, registry, catalogue) for s in registry.tasks()}
    summary = table(
        ["Task", "Kind", "Backends", "Default model"],
        [
            (
                f"`{s.name}`",
                s.kind,
                ", ".join(b.name for b in registry.backends(s.name)),
                f"`{d}`" if (d := described[s.name]["default_model"]) else "none",
            )
            for s in registry.tasks()
        ],
    )
    sections = "\n".join(_task_section(s.name, registry, catalogue) for s in registry.tasks())
    return f"""{BADGES["2x"]}

The tasks 2.x declares, generated from their declarations and the packaged catalogue. A task names what is
computed: typed inputs, outputs and params. A backend is a code path that runs it, in an environment. A model is
a catalogue entry, pinned by commit, that backends load.

{summary}

## Tasks

{sections}"""


def models_page(catalogue: Catalogue | None = None) -> str:
    """Return the model reference: each catalogue entry with its pinned revision, backends and license."""
    entries = (catalogue or CATALOGUE).entries
    rows = [
        (
            f"`{e.id}`",
            f"`{e.revision[:12]}`",
            ", ".join(e.tasks),
            ", ".join(e.backends),
            ", ".join(e.default_for) or "-",
            e.license.expression,
            e.license.commercial_use,
            e.license.copyleft,
            "yes" if e.license.gated else "no",
        )
        for e in entries
    ]
    header = ["Model", "Revision", "Tasks", "Backends", "Default for", "License", "Commercial use", "Copyleft", "Gated"]
    return f"""{BADGES["2x"]}

The packaged catalogue. A model is a catalogue entry, data and not code: a hub id, the commit its weights are
loaded at, the tasks and backends that serve it, and its license record. Every load passes the commit. Check a
model's terms before using it on clinical or commercial work; a gated model needs its terms accepted on the hub.

{table(header, rows)}
"""


UNAVAILABLE = "Page unavailable:"
"""The start of a page built outside a checkout; the docs job fails when it finds it in the output."""


def or_unavailable(page: Callable[[], str]) -> str:
    """Return `page()`, or a notice starting `UNAVAILABLE` when no checkout holds its sources."""
    try:
        return page()
    except FileNotFoundError as err:
        return f"{UNAVAILABLE} {err}"


def _require_root(root: Path | None) -> Path:
    found = root or _specs.checkout_root()
    if found is None:
        raise FileNotFoundError("the reference pages are built from a senselab checkout; none lies above this package")
    return found


def datasets_page(root: Path | None = None) -> str:
    """Return the public datasets of the benchmark manifests with their licenses and how to fetch them.

    Raises:
        FileNotFoundError: no checkout holds the manifests.
    """
    root = _require_root(root)
    manifests = sorted((root / "benchmarks" / "manifests").glob("*.yaml"))
    loaded = [(m.relative_to(root), DataManifest.load(m)) for m in manifests]
    if not loaded:
        raise FileNotFoundError(f"no data manifests under {root / 'benchmarks' / 'manifests'}")
    rows = [
        (
            f"`{m.name}`",
            m.version,
            m.license.expression,
            m.license.commercial_use,
            "yes" if m.license.gated else "no",
            len(m.files),
            f"`senselab data fetch {path}`",
        )
        for path, m in loaded
    ]
    terms = "\n\n".join(f"`{m.name}`: {' '.join(m.terms.split())}" for _, m in loaded)
    return f"""{BADGES["2x"]}

Examples and benchmarks run on public datasets. A data manifest pins each file by sha256 with its license and
terms; `senselab data fetch` downloads into a cache outside the repository and refuses a hash mismatch. No dataset
is committed; an unavailable source fails the build.

{table(["Dataset", "Version", "License", "Commercial use", "Gated", "Files", "Fetch"], rows)}

## Example

Fetch LibriSpeech test-clean, then transcribe one utterance on the default model of `speech_to_text`:

```bash
uv run senselab data fetch benchmarks/manifests/librispeech-test-clean.yaml
```

```python
import senselab

result = senselab.speech_to_text("path/to/an/extracted/utterance.flac", timestamps="word")
```

## Terms

{terms}
"""


def _smoke_tasks(root: Path | None) -> set[str]:
    suites = sorted((root / "benchmarks" / "suites").glob("*.yaml")) if root else []
    return {b["task"] for s in suites for b in yaml.safe_load(s.read_text("utf-8")).get("benchmarks", ())}


def tolerance_dir() -> Traversable:
    """Return the folder `scripts/generate_tolerance.py` writes profiles to, `<task>/<backend>.yaml` under it."""
    return resources.files("senselab").joinpath("data", "tolerance")


def _has_tolerance(registry: Registry, task: str, folder: Traversable) -> bool:
    backends = registry.backends(task)
    return bool(backends) and all(tolerance_profile_path(folder, task, b.name).is_file() for b in backends)


def task_states(
    registry: Registry = REGISTRY, ledger: str = "", root: Path | None = None, tolerance: Traversable | None = None
) -> list[dict[str, Any]]:
    """Return each declared task's state: `{task, declared, implemented, ported, benchmarked, parity}`.

    Declared: in `registry`. Implemented: a backend is declared. Ported: a row of the ledger's moved-in-Phase-2 table
    names the task by name. Benchmarked: a benchmark suite names it.
    Parity: every backend has a tolerance profile in `tolerance`, the folder the generator writes (the packaged one).
    """
    moved = " ".join(" ".join(r.values()) for r in _specs.moved_in_phase(ledger, 2))
    benchmarked = _smoke_tasks(root)
    return [
        {
            "task": s.name,
            "declared": True,
            "implemented": bool(registry.backends(s.name)),
            "ported": re.search(rf"[`/]{re.escape(s.name)}[`/]", moved) is not None,
            "benchmarked": s.name in benchmarked,
            "parity": _has_tolerance(registry, s.name, tolerance or tolerance_dir()),
        }
        for s in registry.tasks()
    ]


def _mark(flag: bool) -> str:  # noqa: FBT001
    return "yes" if flag else "-"


def status_page(registry: Registry = REGISTRY, root: Path | None = None) -> str:
    """Return the 2.x status page: tasks, phases and exits, consolidation, ledger marks and open review items.

    Raises:
        FileNotFoundError: no checkout holds the specs.
    """
    root = _require_root(root)
    ledger, register, design, state = (
        _specs.read_spec(n, root) for n in ("migration-ledger.md", "review-register.md", "design.md", "STATE.md")
    )
    tasks = table(
        ["Task", "Declared", "Implemented", "Ported", "Benchmarked", "At parity"],
        [
            (
                f"`{t['task']}`",
                *(_mark(t[k]) for k in ("declared", "implemented", "ported", "benchmarked")),
                "measured" if t["parity"] else "not measured",
            )
            for t in task_states(registry, ledger, root)
        ],
    )
    phases = table(["Phase", "State"], [(p, s) for p, s in _specs.phase_states(state).items() if p[0].isdigit()])
    exits = "\n".join(f"- {e}" for e in _specs.phase_exits(design, 2)) or _NONE
    inventory = table(
        ["Behaviour", "Copies today", "One implementation in 2.x"],
        [
            tuple(_specs.scrub(r[k]) for k in ("Behaviour", "Copies today", "One implementation in 2.x"))
            for r in _specs.consolidation_inventory(design)
        ],
    )
    marks = table(
        ["Area (1.x paths)", "Where", "Mark", "Status"],
        [
            (r["Area (1.x paths)"], r["Where"], r["Mark"], BADGES["1x" if r["Where"].startswith("on alpha") else "2x"])
            for r in _specs.ledger_marks(ledger)
        ],
    )
    moved = table(
        ["1.x module", "Moved to"],
        [tuple(_specs.scrub(v) for v in list(r.values())[:2]) for r in _specs.moved_in_phase(ledger, 2)],
    )
    open_rows = table(["Owner", "Open items"], _specs.open_by_phase(register).items())
    return f"""{BADGES["2x"]}

What 2.x has built so far. Generated from the declarations and the spec files of the checkout.

## Tasks

A task is declared when its declaration is in the registry, implemented when a backend is declared, ported when
the ledger lists the 1.x module it came from, benchmarked when a benchmark suite runs it, and at parity when every
backend has a measured tolerance profile.

{tasks}

## Phases

{phases}

Exit requirements of Phase 2:

{exits}

## Consolidation inventory

Each phase removes the duplicates it touches; Phase 5 exits with this list empty.

{inventory}

## Migration ledger

The last `alpha` commit reconciled into 2.x, per 1.x area.

{marks}

Moved in Phase 2:

{moved}

## Open review items

Items found in review and not yet resolved, by the phase that owns them.

{open_rows}
"""
