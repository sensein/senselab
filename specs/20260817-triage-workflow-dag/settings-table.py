"""Emit the per-family settings table: what each task asks for, grouped by identical settings."""

import collections
import dataclasses

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.nodes.branches import EXPECTATIONS
from senselab.audio.workflows.triage.nodes.gates import (
    DEFAULT_LAYER,
    FAMILY_LAYER,
    GATE_SECTION,
    GROUP_LAYER,
    Pattern,
)

cfg = load_triage_config(None)
SETTINGS = sorted(cfg.require("branch").keys()) if isinstance(cfg.get("branch"), dict) else []

out = []
out.append("# What each task asks for, and what every task shares\n")
out.append(
    "Generated from `EXPECTATIONS` and the packaged config by "
    "`specs/20260817-triage-workflow-dag/settings-table.py`. Regenerate it rather than editing it.\n"
)
out.append("## The instrument settings are global\n")
out.append(
    "Every setting below says how a reading is taken and applies identically to every task. "
    "`branch:` carries no per-family override and holds no gate.\n"
)
out.append("| instrument setting | value |\n| --- | --- |")
for k in SETTINGS:
    v = cfg.get(f"branch.{k}")
    if isinstance(v, (int, float, str, bool)) or v is None:
        out.append(f"| `{k}` | `{v}` |")

out.append(f"\n## The gates resolve family, then group, then default, in `{GATE_SECTION}`\n")
out.append(
    "A gate says what reading is good enough. A layer that names no value for a gate does not "
    "apply it, which is why `GLIDE` carries no `f0_spread_max_semitones`. A family overrides its "
    "group key by key.\n"
)
default = cfg.get(f"{GATE_SECTION}.{DEFAULT_LAYER}") or {}
by_family = cfg.get(f"{GATE_SECTION}.{FAMILY_LAYER}") or {}
out.append(
    f"`{DEFAULT_LAYER}` carries {len(default) or 'nothing'}"
    + (": " + ", ".join(f"`{k}` = `{v}`" for k, v in sorted(default.items())) if default else "")
    + f" — no gate reaches every group. `{FAMILY_LAYER}` carries "
    + (", ".join(f"`{k}`" for k in sorted(by_family)) if by_family else "nothing")
    + ": no per-family difference has been derived yet.\n"
)
groups = [p.name for p in Pattern]
gate_names = sorted({name for group in groups for name in (cfg.get(f"{GATE_SECTION}.{GROUP_LAYER}.{group}") or {})})
out.append("| gate | " + " | ".join(f"`{group}`" for group in groups) + " |")
out.append("| --- | " + " | ".join("---" for _ in groups) + " |")
for name in gate_names:
    cells = []
    for group in groups:
        table = cfg.get(f"{GATE_SECTION}.{GROUP_LAYER}.{group}") or {}
        cells.append("—" if name not in table else f"`{table[name]}`")
    out.append(f"| `{name}` | " + " | ".join(cells) + " |")

out.append("\n## What else differs is the expectation row\n")

by_settings = collections.defaultdict(list)
for branch, table in EXPECTATIONS.items():
    for fam, e in table.items():
        d = dataclasses.asdict(e) if dataclasses.is_dataclass(e) else dict(vars(e))
        live = tuple(sorted((k, str(v)) for k, v in d.items() if v not in (None, (), [], {}, False)))
        by_settings[(branch, live)].append(fam)

out.append(
    f"**{len(by_settings)} distinct settings groups over {sum(len(v) for v in by_settings.values())} families.**\n"
)
for branch in sorted({b for b, _ in by_settings}):
    out.append(f"\n### {branch}\n")
    out.append("| pattern | settings | families |\n| --- | --- | --- |")
    rows = [(k, f) for k, f in by_settings.items() if k[0] == branch]
    for (_, key), fams in sorted(rows, key=lambda kv: -len(kv[1])):
        d = dict(key)
        pat = d.pop("pattern", "?").split(".")[-1]
        s = "; ".join(f"`{k}`={v}" for k, v in sorted(d.items())) or "—"
        out.append(f"| `{pat}` | {s[:160]} | {', '.join(sorted(fams))} |")
print("\n".join(out))
