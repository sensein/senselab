#!/usr/bin/env python3
"""Generate model_registry.md from model_registry.yaml."""

from pathlib import Path

import yaml


def main() -> None:
    """Read YAML and output Markdown table."""
    registry_path = Path(__file__).parent.parent / "docs" / "model_registry.yaml"
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
        print("| Model | Source | Model ID | Embedding Dim | Parameters | Recommended For |")
        print("|-------|--------|----------|---------------|------------|-----------------|")
        for m in task_models:
            name = m["name"]
            source = m["source"]
            model_id = f'`{m["model_id"]}`'
            emb = m.get("embedding_dim", "—")
            params = m.get("parameters", "—")
            rec = m.get("recommended_for", "—")
            print(f"| {name} | {source} | {model_id} | {emb} | {params} | {rec} |")
        print()


if __name__ == "__main__":
    main()
