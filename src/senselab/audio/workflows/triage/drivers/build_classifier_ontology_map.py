#!/usr/bin/env python
"""Generate the classifier-ontology profile the triage workflow's airway branch corroborates with.

Fetches three pinned artifacts -- the AudioSet ontology, AudioSet's released 527-class index and
YAMNet's own 521-class map -- resolves each HeAR event label onto AudioSet node identifiers, expands
every mapping to its ontology subtree, and writes one JSON profile under the triage module's
``data/classifier_ontology/``.

Run it with::

    uv run python scripts/build_classifier_ontology_map.py

Nothing it writes is hand-edited afterwards. Re-running it with the same pins reproduces the file
byte for byte; changing a pin is the only way its content changes.

The reasoning, the fetched versions and what the mapping fixed are in
``specs/20260910-classifier-ontology-mapping/design.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from senselab.audio.tasks.health_acoustics.hear import HEAR_EVENT_LABELS
from senselab.audio.workflows.triage.classifier_ontology import PROFILE_DIR, PROFILE_VERSION

REQUEST_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class PinnedSource:
    """One fetched artifact, addressed by commit rather than by branch.

    Attributes:
        key: The name this source is recorded under in the profile.
        repository: The repository the artifact is served from.
        commit: The 40-hex commit the raw URL resolves against.
        path: The artifact's path inside that repository.
        retrieved: ISO date the recorded digest was observed.
        sha256: The artifact's expected digest; a mismatch aborts the build.
        licence: The artifact's licence identifier.
        attribution: The attribution line the licence requires.
    """

    key: str
    repository: str
    commit: str
    path: str
    retrieved: str
    sha256: str
    licence: str
    attribution: str

    @property
    def url(self) -> str:
        """The raw URL this source is fetched from.

        Returns:
            A ``raw.githubusercontent.com`` URL carrying the pinned commit, never a branch name.
        """
        owner_repo = self.repository.removeprefix("https://github.com/")
        return f"https://raw.githubusercontent.com/{owner_repo}/{self.commit}/{self.path}"

    def record(self) -> dict[str, str]:
        """The provenance block written into the profile for this source.

        Returns:
            Every field a reader needs to re-fetch and re-verify the artifact.
        """
        return {
            "repository": self.repository,
            "commit": self.commit,
            "path": self.path,
            "url": self.url,
            "sha256": self.sha256,
            "retrieved": self.retrieved,
            "licence": self.licence,
            "attribution": self.attribution,
        }


AUDIOSET_ONTOLOGY = PinnedSource(
    key="audioset_ontology",
    repository="https://github.com/audioset/ontology",
    commit="d417d32bf59c711abb5910fd2f76a0eb44697991",
    path="ontology.json",
    retrieved="2026-09-10",
    sha256="9c685f4403eecc3ca9be37fd7285cf212feaaea6ff7229d3e7ca89e0d1f2d15d",
    licence="CC-BY-4.0",
    attribution="AudioSet ontology, Google LLC, licensed CC BY 4.0",
)

AUDIOSET_CLASS_INDEX = PinnedSource(
    key="audioset_class_index",
    repository="https://github.com/YuanGongND/ast",
    commit="9fc5b67075f6c59a84c7931c4b5a0bf60c1416c6",
    path="egs/audioset/data/class_labels_indices.csv",
    retrieved="2026-09-10",
    sha256="cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429",
    licence="CC-BY-4.0",
    attribution="AudioSet released class index, Google LLC, licensed CC BY 4.0",
)

YAMNET_CLASS_MAP = PinnedSource(
    key="yamnet_class_map",
    repository="https://github.com/tensorflow/models",
    commit="dfffd623b6be8d1d9744b8e261fbac370d17c46d",
    path="research/audioset/yamnet/yamnet_class_map.csv",
    retrieved="2026-09-10",
    sha256="cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2",
    licence="Apache-2.0",
    attribution="YAMNet class map, TensorFlow Model Garden, licensed Apache 2.0",
)

SOURCES: tuple[PinnedSource, ...] = (AUDIOSET_ONTOLOGY, AUDIOSET_CLASS_INDEX, YAMNET_CLASS_MAP)

CROSSWALK: dict[str, dict[str, Any]] = {
    "Cough": {"group": "cough", "roots": ["Cough"], "note": ""},
    "Snore": {"group": "breath", "roots": ["Snoring"], "note": ""},
    "Baby Cough": {
        "group": "cough",
        "roots": ["Cough"],
        "note": (
            "AudioSet has no infant-specific cough class -- its only infant classes are "
            "'Baby cry, infant cry' and 'Baby laughter' -- so this maps onto the adult 'Cough' "
            "node rather than being dropped."
        ),
    },
    "Breathe": {"group": "breath", "roots": ["Breathing"], "note": ""},
    "Sneeze": {"group": "cough", "roots": ["Sneeze"], "note": ""},
    "Throat Clear": {"group": "cough", "roots": ["Throat clearing"], "note": ""},
    "Laugh": {"group": "laugh", "roots": ["Laughter"], "note": ""},
    "Speech": {"group": "speech", "roots": ["Speech"], "note": ""},
}
"""HeAR label -> the AudioSet class names it denotes, before subtree expansion.

HeAR publishes no ontology, so this crosswalk is the one authored input the profile carries. Every
other field the profile holds is derived from a pinned source. The eight keys are checked against
:data:`~senselab.audio.tasks.health_acoustics.hear.HEAR_EVENT_LABELS` at build time.
"""

GROUP_FAMILY: dict[str, str] = {"cough": "airway", "breath": "airway", "laugh": "voice", "speech": "speech"}
"""Which triage family each crosswalk group belongs to."""


def _fetch(source: PinnedSource) -> bytes:
    """Fetch one pinned artifact and verify its digest.

    Args:
        source: The artifact to fetch.

    Returns:
        The artifact's bytes.

    Raises:
        ValueError: If the fetched bytes do not hash to the recorded digest, which means the pin no
            longer names what it named when the digest was recorded.
    """
    with urllib.request.urlopen(source.url, timeout=REQUEST_TIMEOUT_S) as response:  # noqa: S310 — https, pinned
        payload: bytes = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != source.sha256:
        raise ValueError(f"{source.key}: fetched {digest}, pinned {source.sha256} at {source.url}")
    return payload


def _read_mids(payload: bytes) -> list[str]:
    """The machine ids a released class list carries, in index order.

    Args:
        payload: The CSV bytes, whose header names an ``mid`` column.

    Returns:
        The machine ids.
    """
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    return [row["mid"] for row in reader]


def _subtree(node_id: str, children: dict[str, list[str]], seen: list[str] | None = None) -> list[str]:
    """The node and every descendant of it, depth-first, root first, each once.

    Args:
        node_id: The subtree root.
        children: ``{node id: child ids}`` over the whole ontology.
        seen: Accumulator; callers pass nothing.

    Returns:
        The subtree's node ids.
    """
    order = [] if seen is None else seen
    if node_id in order:
        return order
    order.append(node_id)
    for child in children.get(node_id, []):
        if child in children:
            _subtree(child, children, order)
    return order


def _classes(ontology: list[dict[str, Any]], released: Iterable[str], emittable: Iterable[str]) -> dict[str, Any]:
    """The ontology node table the profile ships.

    Args:
        ontology: The parsed ``ontology.json``.
        released: Machine ids in AudioSet's released 527-class index.
        emittable: Machine ids in YAMNet's 521-class map.

    Returns:
        ``{machine id: {name, child_ids, restrictions, in_audioset_527, in_yamnet_521}}``.
    """
    released_set, emittable_set = set(released), set(emittable)
    return {
        node["id"]: {
            "name": node["name"],
            "child_ids": list(node.get("child_ids") or []),
            "restrictions": list(node.get("restrictions") or []),
            "in_audioset_527": node["id"] in released_set,
            "in_yamnet_521": node["id"] in emittable_set,
        }
        for node in ontology
    }


def build() -> dict[str, Any]:
    """Fetch the pinned taxonomies and resolve the crosswalk into a complete profile.

    Returns:
        The profile mapping, ready to be written as JSON.

    Raises:
        ValueError: If the crosswalk does not name exactly HeAR's eight event labels, or names an
            AudioSet class the ontology does not contain.
    """
    ontology = json.loads(_fetch(AUDIOSET_ONTOLOGY).decode("utf-8"))
    released = _read_mids(_fetch(AUDIOSET_CLASS_INDEX))
    emittable = _read_mids(_fetch(YAMNET_CLASS_MAP))

    classes = _classes(ontology, released, emittable)
    children = {node_id: list(entry["child_ids"]) for node_id, entry in classes.items()}
    by_name = {entry["name"]: node_id for node_id, entry in classes.items()}
    if len(by_name) != len(classes):
        raise ValueError("the AudioSet ontology no longer has unique class names; the crosswalk keys on them")

    if set(CROSSWALK) != set(HEAR_EVENT_LABELS):
        raise ValueError(f"crosswalk covers {sorted(CROSSWALK)}, HeAR emits {sorted(HEAR_EVENT_LABELS)}")

    mapping: dict[str, Any] = {}
    for label in HEAR_EVENT_LABELS:
        entry = CROSSWALK[label]
        missing = [name for name in entry["roots"] if name not in by_name]
        if missing:
            raise ValueError(f"{label}: {missing} are not AudioSet class names")
        roots = [by_name[name] for name in entry["roots"]]
        resolved: list[str] = []
        for root in roots:
            for node_id in _subtree(root, children):
                if node_id not in resolved:
                    resolved.append(node_id)
        mapping[label] = {
            "group": entry["group"],
            "family": GROUP_FAMILY[entry["group"]],
            "audioset_ids": roots,
            "audioset_names": [classes[node_id]["name"] for node_id in roots],
            "corroborating_ids": resolved,
            "corroborating_names": [classes[node_id]["name"] for node_id in resolved],
            "unreleased": [classes[i]["name"] for i in resolved if not classes[i]["in_audioset_527"]],
            "not_emittable_by_yamnet": [classes[i]["name"] for i in resolved if not classes[i]["in_yamnet_521"]],
            "note": entry["note"],
        }

    labels = list(HEAR_EVENT_LABELS)
    overlaps = [
        {
            "labels": [left, right],
            "shared_names": [classes[i]["name"] for i in mapping[left]["corroborating_ids"] if i in shared],
        }
        for index, left in enumerate(labels)
        for right in labels[index + 1 :]
        if (shared := set(mapping[left]["corroborating_ids"]) & set(mapping[right]["corroborating_ids"]))
    ]

    return {
        "profile_version": PROFILE_VERSION,
        "generated": date.today().isoformat(),
        "generator": "scripts/build_classifier_ontology_map.py",
        "spec": "specs/20260910-classifier-ontology-mapping/design.md",
        "sources": {source.key: source.record() for source in SOURCES},
        "hear_labels": labels,
        "audioset": {
            "ontology_class_count": len(classes),
            "released_class_count": len(released),
            "yamnet_class_count": len(emittable),
            "released_not_in_yamnet": sorted(classes[m]["name"] for m in set(released) - set(emittable)),
            "classes": classes,
        },
        "mapping": mapping,
        "overlaps": overlaps,
    }


def main() -> None:
    """Build the profile and write it under the triage module's ``data/classifier_ontology/``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="output path; default is today's dated profile")
    arguments = parser.parse_args()

    profile = build()
    out = arguments.out or PROFILE_DIR / f"{profile['generated']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2, sort_keys=False) + "\n")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    for label, entry in profile["mapping"].items():
        print(f"  {label:<12} -> {', '.join(entry['corroborating_names'])}")


if __name__ == "__main__":
    main()
