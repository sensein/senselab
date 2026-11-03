"""A new dataset's profile, written as a template: every value it must supply blank, beside what the value means.

A dataset that wants triage needs a profile (plan step 5): its task list, its policy, the statistics of its own
recordings and its per-task values, and, unless it builds on a profile that carries them, the detector tuning.
:func:`write_template` writes that profile with each value the dataset must still supply left empty, preceded by its
entry in ``data/profile_values.yaml``: what it means, its unit, and how a dataset sets it. By default the template
extends senselab's reference profile (``senselab:adult_clinical_speech``), so only the dataset's own values are
asked for; ``extends=None`` asks for every value, and ``start_from`` copies another dataset's profile instead.

The template is a starting point, not a profile a run accepts: :func:`~senselab.audio.workflows.triage.
dataset_profile.missing_dataset_values` names whatever is still blank.
"""

from __future__ import annotations

import datetime
import shutil
import textwrap
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml  # type: ignore[import-untyped]

from senselab.audio.workflows.triage import dataset_profile
from senselab.audio.workflows.triage.dataset_profile import (
    CONFIG_FILE,
    MANIFEST,
    PACKAGED_DATA,
    REQUIRED_DATASET_PATHS,
    data_values,
    profile_values,
)

TEMPLATE_DIR = PACKAGED_DATA / "profile_template"
"""Skeletons of the dataset files every run reads (:data:`~senselab.audio.workflows.triage.dataset_profile.
REQUIRED_DATASET_PATHS`), with what each holds."""

REFERENCE_PROFILE = "senselab:adult_clinical_speech"
"""The profile a template extends by default: senselab's reference profile for adult clinical speech recordings."""

KIND_LABELS = {
    "a": "decide: your dataset's policy",
    "b": "compute from your own recordings (a pilot run)",
    "c": "tune by listening or labelled recordings",
    "d": "per task in your task list",
}
"""How each kind of profile value is set (``data/profile_values.yaml``)."""


def _lookup(document: Any, path: list[str]) -> Any:  # noqa: ANN401 -- a YAML value
    for part in path:
        if not isinstance(document, Mapping) or part not in document:
            return None
        document = document[part]
    return document


def values_to_supply(extends: str | Path | None = REFERENCE_PROFILE) -> dict[str, list[dict[str, Any]]]:
    """The required profile values a dataset must supply when it builds on ``extends``.

    Args:
        extends: The profile the new one extends (``senselab:<name>`` or a directory), or None for none.

    Returns:
        ``{data file: [entry, ...]}``, each entry as ``data/profile_values.yaml`` gives it; the configuration's
        under :data:`~senselab.audio.workflows.triage.dataset_profile.CONFIG_FILE`. The active profile and the
        run's data section are as they were before the call.
    """
    from senselab.audio.workflows.triage.config import load_triage_config  # noqa: PLC0415

    before, run_data = dataset_profile.active_profile(), dataset_profile.run_data()
    dataset_profile.deactivate_profile()
    try:
        config = load_triage_config(profile=extends) if extends is not None else load_triage_config()
        out: dict[str, list[dict[str, Any]]] = {}
        for relative, entries in profile_values().items():
            document = config.values if relative == CONFIG_FILE else data_values(relative)
            blank = [e for e in entries if e.get("required") and _lookup(document, list(e["path"])) is None]
            if blank:
                out[relative] = blank
        return out
    finally:
        dataset_profile.deactivate_profile()
        dataset_profile.set_run_data(run_data)
        if before is not None:
            dataset_profile.activate_profile(before)


def _comment(entry: Mapping[str, Any]) -> str:
    unit = entry.get("unit")
    means = str(entry.get("means") or "").strip()
    head = f"{means} ({unit})" if means and unit and unit != "none" else means
    kind = KIND_LABELS.get(str(entry.get("kind")), str(entry.get("kind")))
    fit = str(entry.get("fit") or "").strip()
    return " ".join(part for part in (head, f"[{kind}]", fit) if part)


def _emit(entries: list[dict[str, Any]]) -> Iterator[str]:
    """YAML lines nesting each entry's path, its value blank, its comment on the line above."""
    tree: dict[str, Any] = {}
    for entry in entries:
        node = tree
        *parents, last = [str(part) for part in entry["path"]]
        for part in parents:
            node = node.setdefault(part, {})
        node[last] = entry

    def walk(node: Mapping[str, Any], depth: int) -> Iterator[str]:
        pad = "  " * depth
        for key, child in node.items():
            name = yaml.safe_dump(key, default_style=None).strip().removesuffix("...").strip()
            if isinstance(child, Mapping) and "path" in child and "kind" in child:
                for line in textwrap.wrap(_comment(child), width=max(40, 116 - len(pad))):
                    yield f"{pad}# {line}"
                yield f"{pad}{name}:"
            else:
                yield f"{pad}{name}:"
                yield from walk(child, depth + 1)

    yield from walk(tree, 0)


def write_template(
    out: str | Path,
    *,
    name: str,
    extends: str | Path | None = REFERENCE_PROFILE,
    start_from: str | Path | None = None,
) -> Path:
    """Write a new dataset's profile with every value it must still supply blank, each beside its entry.

    Args:
        out: The new profile's directory; it must not exist yet.
        name: The new profile's name.
        extends: The profile it builds on: senselab's reference profile by default, another ``senselab:<name>`` or
            a directory, or None to supply every value itself.
        start_from: Copy this profile directory instead (a dataset that starts from another's values); the copy is
            renamed and records where it came from.

    Returns:
        The profile directory.

    Raises:
        FileExistsError: If ``out`` exists.
    """
    root = Path(out)
    if root.exists():
        raise FileExistsError(f"{root} exists; a template is written into a new directory")
    written = datetime.date.today().isoformat()
    if start_from is not None:
        shutil.copytree(Path(start_from), root)
        manifest = yaml.safe_load((root / MANIFEST).read_text()) or {}
        copied = manifest.get("name")
        manifest["name"] = name
        manifest["version"] = "0"
        manifest["provenance"] = {"copied_from": str(start_from), "copied_from_profile": copied, "on": written}
        (root / MANIFEST).write_text(yaml.safe_dump(manifest, sort_keys=False))
        return root
    root.mkdir(parents=True)
    files: dict[str, str] = {}
    for relative in REQUIRED_DATASET_PATHS:
        shutil.copy(TEMPLATE_DIR / relative, root / relative)
        files[relative] = relative
    config = None
    for relative, entries in values_to_supply(extends).items():
        target = root / ("config.yaml" if relative == CONFIG_FILE else relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        header = [
            f"# {name}: the values of senselab's {relative} this dataset supplies (template written {written}).",
            "# Fill in every blank; a run refuses a profile that leaves a required value empty, naming it.",
        ]
        target.write_text("\n".join([*header, *_emit(entries)]) + "\n")
        if relative == CONFIG_FILE:
            config = "config.yaml"
        else:
            top = relative.split("/")[0]
            files[top] = top
    manifest: dict[str, Any] = {"name": name, "version": "0"}
    if extends is not None:
        manifest["extends"] = str(extends)
    manifest["provenance"] = {"template": "senselab triage profile template", "written": written}
    if config is not None:
        manifest["config"] = config
    manifest["files"] = dict(sorted(files.items()))
    head = (
        f"# The {name} dataset profile, written from senselab's template. Fill in tasks.yaml and every blank value;\n"
        "# each blank carries what it means and how to set it.\n"
    )
    (root / MANIFEST).write_text(head + yaml.safe_dump(manifest, sort_keys=False))
    return root
