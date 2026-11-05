"""A new dataset's profile, written as a template: every value it must supply blank, beside what the value means.

A dataset that wants triage needs a profile (plan step 5): its task list, its policy, the statistics of its own
recordings and its per-task values, and, unless it builds on a profile that carries them, the detector tuning.
:func:`write_template` writes that profile with each value the dataset must still supply left empty, preceded by its
entry in ``data/profile_values.yaml``: what it means, its unit, and how a dataset sets it; a corpus statistic (kind b)
is written commented out, for ``scripts/fit_profile.py`` to fill in ``fitted.yaml``. By default the template
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
from typing import Any, Collection, Iterator, Mapping, Sequence

import yaml  # type: ignore[import-untyped]

from senselab.audio.workflows.triage import dataset_profile
from senselab.audio.workflows.triage.dataset_profile import (
    CONFIG_FILE,
    MANIFEST,
    PACKAGED_DATA,
    PILOT_KEY,
    REQUIRED_DATASET_PATHS,
    SENSELAB_PROFILE,
    DatasetDataMissing,
    data_values,
    load_profile,
    needed,
    profile_values,
    stage_dataset_paths,
)
from senselab.audio.workflows.triage.workflow import STAGE_NAMES, check_stages, listed_stages, running_stages

TEMPLATE_DIR = PACKAGED_DATA / "profile_template"
"""Skeletons of the dataset files every run reads (:data:`~senselab.audio.workflows.triage.dataset_profile.
REQUIRED_DATASET_PATHS`), with what each holds."""

REFERENCE_PROFILE = "senselab:adult_clinical_speech"
"""The profile a template extends by default: senselab's reference profile for adult clinical speech recordings."""

KIND_LABELS = {
    "a": "decide: your dataset's policy",
    "b": "compute from your own recordings: left commented out; run a pilot, then scripts/fit_profile.py fits it",
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


def values_to_supply(
    extends: str | Path | None = REFERENCE_PROFILE, *, off: Collection[str] = ()
) -> dict[str, list[dict[str, Any]]]:
    """The required profile values a dataset must supply when it builds on ``extends``.

    Args:
        extends: The profile the new one extends (``senselab:<name>`` or a directory), or None for none.
        off: Stages the new profile's workflow leaves off, beside those ``extends`` already does; a value only they
            read is not asked for.

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
        running = running_stages(_with_off(config.values, off))
        out: dict[str, list[dict[str, Any]]] = {}
        for relative, entries in profile_values().items():
            try:
                document = config.values if relative == CONFIG_FILE else data_values(relative)
            except DatasetDataMissing:
                document = {}  # a dataset file the base does not supply: every value of it is the new profile's
            blank = [e for e in entries if needed(e, running) and _lookup(document, list(e["path"])) is None]
            if blank:
                out[relative] = blank
        return out
    finally:
        dataset_profile.deactivate_profile()
        dataset_profile.set_run_data(run_data)
        if before is not None:
            dataset_profile.activate_profile(before)


def _with_off(config_values: Mapping[str, Any], off: Collection[str]) -> dict[str, Any]:
    """The configuration with ``off`` also left out of its workflow.

    Raises:
        ValueError: If ``off`` names a stage that is not one, or one that may not be off.
    """
    listed = [stage for stage in listed_stages(config_values) if stage not in set(off)]
    unknown = sorted(set(off) - set(STAGE_NAMES))
    if unknown:
        raise ValueError(f"{unknown} are not triage stages; the stages are {STAGE_NAMES}")
    check_stages(listed)
    return {**config_values, "workflow": {**(config_values.get("workflow") or {}), "stages": listed}}


def _comment(entry: Mapping[str, Any]) -> str:
    unit = entry.get("unit")
    means = str(entry.get("means") or "").strip()
    head = f"{means} ({unit})" if means and unit and unit != "none" else means
    kind = KIND_LABELS.get(str(entry.get("kind")), str(entry.get("kind")))
    fit = str(entry.get("fit") or "").strip()
    return " ".join(part for part in (head, f"[{kind}]", fit) if part)


def _emit(entries: list[dict[str, Any]]) -> Iterator[str]:
    """YAML lines nesting each entry's path, its value blank, its comment on the line above.

    A corpus statistic (kind ``b``) is written commented out: ``scripts/fit_profile.py`` fills it in ``fitted.yaml``,
    which a blank in the profile's own file would override. A mapping holding only such values is commented out too,
    so it does not blank the profile it extends.
    """
    tree: dict[str, Any] = {}
    for entry in entries:
        node = tree
        *parents, last = [str(part) for part in entry["path"]]
        for part in parents:
            node = node.setdefault(part, {})
        node[last] = entry

    def is_entry(node: Any) -> bool:  # noqa: ANN401 -- a tree node
        return isinstance(node, Mapping) and "path" in node and "kind" in node

    def active(node: Any) -> bool:  # noqa: ANN401 -- a tree node
        if is_entry(node):
            return node.get("kind") != dataset_profile.FITTED_KIND
        return any(active(child) for child in node.values())

    def walk(node: Mapping[str, Any], depth: int) -> Iterator[str]:
        pad = "  " * depth
        for key, child in node.items():
            name = yaml.safe_dump(key, default_style=None).strip().removesuffix("...").strip()
            off = "" if active(child) else "# "
            if is_entry(child):
                for line in textwrap.wrap(_comment(child), width=max(40, 116 - len(pad))):
                    yield f"{pad}# {line}"
                yield f"{pad}{off}{name}:"
            else:
                yield f"{pad}{off}{name}:"
                yield from (
                    line if not off or line.lstrip().startswith("#") else f"{pad}# {line[len(pad) :]}"
                    for line in walk(child, depth + 1)
                )

    yield from walk(tree, 0)


def _base_config_values(reference: str | None) -> dict[str, Any]:
    """The configuration a template's base resolves to, leaving the active profile and run data as they were."""
    from senselab.audio.workflows.triage.config import load_triage_config  # noqa: PLC0415

    before, run_data = dataset_profile.active_profile(), dataset_profile.run_data()
    try:
        config = load_triage_config(profile=reference) if reference is not None else load_triage_config()
        return dict(config.values)
    finally:
        dataset_profile.deactivate_profile()
        dataset_profile.set_run_data(run_data)
        if before is not None:
            dataset_profile.activate_profile(before)


def _workflow_lines(listed: Sequence[str], off: Collection[str]) -> list[str]:
    """The ``workflow`` section a template's config carries when its workflow leaves stages off."""
    return [
        f"# The stages this dataset runs; {', '.join(sorted(off))} are off, and no value only they read is asked for.",
        "workflow:",
        f"  stages: [{', '.join(listed)}]",
    ]


def _absolute(reference: str | Path | None, relative_to: Path | None = None) -> str | None:
    """A profile reference as the new profile records it: ``senselab:<name>`` unchanged, a path made absolute."""
    if reference is None:
        return None
    text = str(reference)
    if text.startswith(SENSELAB_PROFILE):
        return text
    path = Path(text)
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    return str(path.resolve())


def write_template(
    out: str | Path,
    *,
    name: str,
    extends: str | Path | None = REFERENCE_PROFILE,
    start_from: str | Path | None = None,
    supply: Mapping[str, list[dict[str, Any]]] | None = None,
    off: Collection[str] = (),
) -> Path:
    """Write a new dataset's profile with every value it must still supply blank, each beside its entry.

    Args:
        out: The new profile's directory; it must not exist yet.
        name: The new profile's name.
        extends: The profile it builds on: senselab's reference profile by default, another ``senselab:<name>`` or
            a directory (recorded as an absolute path), or None to supply every value itself.
        start_from: Copy this profile directory instead (a dataset that starts from another's values); the copy is
            renamed, records where it came from, and keeps building on what the original built on.
        supply: :func:`values_to_supply` for ``extends`` and ``off``, where the caller has it already.
        off: Stages the new profile's workflow leaves off: its config lists the rest under ``workflow.stages``, and
            a value only those stages read is not asked for. Not with ``start_from``.

    Returns:
        The profile directory.

    Raises:
        FileExistsError: If ``out`` exists.
        ValueError: If ``extends`` names no profile; nothing is written then.
    """
    if off and start_from is not None:
        raise ValueError("off writes a new template; a copy (start_from) keeps the original's workflow")
    root = Path(out)
    if root.exists():
        raise FileExistsError(f"{root} exists; a template is written into a new directory")
    written = datetime.date.today().isoformat()
    if start_from is not None:
        source = Path(start_from).resolve()
        load_profile(source)
        shutil.copytree(source, root)
        manifest = yaml.safe_load((root / MANIFEST).read_text()) or {}
        copied = manifest.get("name")
        manifest["name"] = name
        manifest["version"] = "0"
        if manifest.get("extends"):
            manifest["extends"] = _absolute(manifest["extends"], relative_to=source)
        manifest["provenance"] = {"copied_from": str(source), "copied_from_profile": copied, "on": written}
        (root / MANIFEST).write_text(yaml.safe_dump(manifest, sort_keys=False))
        return root
    reference = _absolute(extends)
    base = load_profile(reference) if reference is not None else None
    blanks = values_to_supply(reference, off=off) if supply is None else supply
    workflow_values = _with_off(_base_config_values(reference), off)
    workflow = _workflow_lines(workflow_values["workflow"]["stages"], off) if off else []
    root.mkdir(parents=True)
    files: dict[str, str] = {}
    for relative in (*REQUIRED_DATASET_PATHS, *stage_dataset_paths(running_stages(workflow_values))):
        skeleton = TEMPLATE_DIR / relative
        if (base is not None and base.resolve(relative) is not None) or not skeleton.is_file():
            continue
        shutil.copy(skeleton, root / relative)
        files[relative] = relative
    config = None
    for relative, entries in blanks.items():
        target = root / ("config.yaml" if relative == CONFIG_FILE else relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        header = [
            f"# {name}: the values of senselab's {relative} this dataset supplies (template written {written}).",
            "# Fill in every blank; a run refuses a profile that leaves a required value empty, naming it.",
        ]
        lines = list(_emit(entries))
        if relative == CONFIG_FILE:
            lines, workflow = [*workflow, *lines], []
        target.write_text("\n".join([*header, *lines]) + "\n")
        if relative == CONFIG_FILE:
            config = "config.yaml"
        else:
            top = relative.split("/")[0]
            files[top] = top
    if workflow:  # no config value was blank: the workflow is the config's only content
        header = [f"# {name}: the senselab configuration this dataset changes (template written {written})."]
        (root / "config.yaml").write_text("\n".join([*header, *workflow]) + "\n")
        config = "config.yaml"
    manifest: dict[str, Any] = {"name": name, "version": "0", PILOT_KEY: True}
    if reference is not None:
        manifest["extends"] = reference
    manifest["provenance"] = {"template": "senselab triage profile template", "written": written}
    if config is not None:
        manifest["config"] = config
    manifest["files"] = dict(sorted(files.items()))
    head = (
        f"# The {name} dataset profile, written from senselab's template. Fill in tasks.yaml and every blank value;\n"
        "# each blank carries what it means and how to set it. It starts as a pilot (pilot: true): a run may leave\n"
        "# the values computed from your own recordings (kind b) out. Run a pilot, the COHORT pass and the corpus\n"
        "# table, then scripts/fit_profile.py writes them into fitted.yaml; run the COHORT pass again, replay the\n"
        "# pilot and remove pilot: true.\n"
    )
    (root / MANIFEST).write_text(head + yaml.safe_dump(manifest, sort_keys=False))
    return root
