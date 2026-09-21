"""Emit the per-family settings table: what each task asks for, grouped by identical settings."""
import collections, dataclasses
from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.nodes.branches import EXPECTATIONS

cfg = load_triage_config(None)
GATES = [k for k in sorted(cfg.require("branch").keys())] if isinstance(cfg.get("branch"), dict) else []

out = []
out.append("# What each task asks for, and what every task shares\n")
out.append("Generated from `EXPECTATIONS` and the packaged config by "
           "`specs/20260817-triage-workflow-dag/settings-table.py`. Regenerate it rather than editing it.\n")
out.append("## The operating points are global\n")
out.append("Every gate below applies identically to every task. `branch:` carries no per-family "
           "override, and VERDICT's only per-family hook, `conformance_flags_by_family`, ships empty. "
           "So a 0.5 s `production_min_s` governs a glide, a sustained vowel, a cough and a DDK train alike.\n")
out.append("| operating point | value |\n| --- | --- |")
for k in GATES:
    v = cfg.get(f"branch.{k}")
    if isinstance(v, (int, float, str, bool)) or v is None:
        out.append(f"| `{k}` | `{v}` |")
out.append("\n## What differs is the expectation row\n")

groups = collections.defaultdict(list)
for branch, table in EXPECTATIONS.items():
    for fam, e in table.items():
        d = dataclasses.asdict(e) if dataclasses.is_dataclass(e) else dict(vars(e))
        live = tuple(sorted((k, str(v)) for k, v in d.items() if v not in (None, (), [], {}, False)))
        groups[(branch, live)].append(fam)

out.append(f"**{len(groups)} distinct settings groups over "
           f"{sum(len(v) for v in groups.values())} families.**\n")
for branch in sorted({b for b, _ in groups}):
    out.append(f"\n### {branch}\n")
    out.append("| pattern | settings | families |\n| --- | --- | --- |")
    rows = [(k, f) for k, f in groups.items() if k[0] == branch]
    for (_, key), fams in sorted(rows, key=lambda kv: -len(kv[1])):
        d = dict(key)
        pat = d.pop("pattern", "?").split(".")[-1]
        s = "; ".join(f"`{k}`={v}" for k, v in sorted(d.items())) or "—"
        out.append(f"| `{pat}` | {s[:160]} | {', '.join(sorted(fams))} |")
print("\n".join(out))
