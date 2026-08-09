#!/usr/bin/env python3
"""Generate model_registry.md from model_registry.yaml."""

from pathlib import Path

import yaml


def main() -> None:
    """Read YAML and output Markdown table."""
    registry_path = Path(__file__).parent.parent / "src" / "senselab" / "model_registry.yaml"
    with open(registry_path) as f:
        models = yaml.safe_load(f)

    # Group by task
    tasks: dict = {}
    for model in models:
        task = model["task"]
        if task not in tasks:
            tasks[task] = []
        tasks[task].append(model)

    # Accumulate sections and join with a single blank line between them, rather than
    # printing a trailing blank after each section: the latter left a blank line at
    # end-of-file that `end-of-file-fixer` always stripped before commit, so the
    # generator's own output never actually matched the committed file — `git diff
    # --exit-code` after a regeneration would report a diff forever.
    sections = ["# Senselab Model Registry\n\nAll models supported by senselab, organized by task."]

    for task, task_models in tasks.items():
        lines = []
        title = task.replace("_", " ").title()
        lines.append(f"## {title}\n")
        # `license` is an optional key: only models whose weights carry a restriction
        # narrower than every other entry's default (e.g. DiariZen's CC BY-NC 4.0) set
        # it, so most tasks render without the column at all rather than a page of "—".
        has_license = any("license" in m for m in task_models)
        # `capabilities` is an optional key, present only on diarization entries.
        # Gate the columns on the section, like `license` above, so unrelated task
        # tables are not widened with columns that would be empty in every row.
        has_caps = any("capabilities" in m for m in task_models)

        header = "| Model | Source | Model ID | Embedding Dim | Parameters |"
        separator = "|-------|--------|----------|---------------|------------|"
        if has_license:
            header += " License |"
            separator += "---------|"
        if has_caps:
            header += " Speakers | Text |"
            separator += "---|---|"
        header += " Recommended For |"
        separator += "-----------------|"
        lines.append(header)
        lines.append(separator)

        for m in task_models:
            name = m["name"]
            source = m["source"]
            model_id = f"`{m['model_id']}`"
            emb = m.get("embedding_dim", "—")
            params = m.get("parameters", "—")
            rec = m.get("recommended_for", "—")
            row = f"| {name} | {source} | {model_id} | {emb} | {params} |"
            if has_license:
                license_ = m.get("license", "—")
                row += f" {license_} |"
            if has_caps:
                # Distinguish "key absent" from "key present but empty": today every
                # diarization entry is required (by the registry test) to declare
                # capabilities, so `caps` is never actually `{}` in practice — but the
                # moment another task section adds `capabilities` to only one entry,
                # `.get("capabilities", {})` would render that entry's missing block
                # identically to a backend that declares max_speakers/populates_text
                # as falsy, i.e. a silent "no" instead of "not declared".
                if "capabilities" in m:
                    caps = m["capabilities"]
                    max_spk = caps.get("max_speakers")
                    # An em dash, never "unlimited": null means nobody has measured
                    # the ceiling. Rendering it as unlimited would invent a capability.
                    speakers = "—" if max_spk is None else str(max_spk)
                    text_col = "yes" if caps.get("populates_text") else "no"
                else:
                    speakers = "?"
                    text_col = "?"
                row += f" {speakers} | {text_col} |"
            row += f" {rec} |"
            lines.append(row)

        sections.append("\n".join(lines))

    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
