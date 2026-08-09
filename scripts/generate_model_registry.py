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

    print("# Senselab Model Registry\n")
    print("All models supported by senselab, organized by task.\n")

    for task, task_models in tasks.items():
        title = task.replace("_", " ").title()
        print(f"## {title}\n")
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
        print(header)
        print(separator)

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
                caps = m.get("capabilities", {})
                max_spk = caps.get("max_speakers")
                # An em dash, never "unlimited": null means nobody has measured the
                # ceiling. Rendering it as unlimited would invent a capability.
                speakers = "—" if max_spk is None else str(max_spk)
                text_col = "yes" if caps.get("populates_text") else "no"
                row += f" {speakers} | {text_col} |"
            row += f" {rec} |"
            print(row)
        print()


if __name__ == "__main__":
    main()
