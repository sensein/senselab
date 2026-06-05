"""Ranking-store layout, immutable metric-version management, and manifest.

A store is a directory::

    <store>/
        manifest.json            # index of versions, lineage, corpus/unit
        metric_versions/<vN>.json   # immutable metric versions
        rankings/<vN>.parquet       # one ranking per version
        annotations.json            # managed by annotate.py
        movement/<vA>__<vB>.json    # managed by movement.py

Metric versions are immutable: re-writing an existing ``version_id`` raises.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from senselab.audio.workflows.ranking import io
from senselab.audio.workflows.ranking.types import (
    MetricDefinition,
    MetricVersion,
    RecalibrationResult,
    SignalTerm,
)


def _defn_to_dict(defn: MetricDefinition) -> dict:
    """Serialize a metric definition to a plain dict (stable key order)."""
    return {
        "name": defn.name,
        "direction": defn.direction,
        "combine": defn.combine,
        "notes": defn.notes,
        "terms": [asdict(t) for t in defn.terms],
    }


def _defn_from_dict(data: dict) -> MetricDefinition:
    """Reconstruct a metric definition from a dict."""
    terms = [
        SignalTerm(
            signal=t["signal"],
            weight=float(t["weight"]),
            transform=t.get("transform", "identity"),
            transform_params=dict(t.get("transform_params", {})),
            missing=t.get("missing", "unscorable"),
        )
        for t in data["terms"]
    ]
    return MetricDefinition(
        name=data["name"],
        terms=terms,
        direction=data.get("direction", "higher_is_better"),
        combine=data.get("combine", "weighted_sum"),
        notes=data.get("notes", ""),
    )


def metric_definition_hash(defn: MetricDefinition) -> str:
    """Stable content hash of a metric definition (for ranking provenance / identity)."""
    blob = json.dumps(_defn_to_dict(defn), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class RankingStore:
    """Filesystem-backed store for metric versions, rankings, and reports."""

    def __init__(self, root: Path | str) -> None:
        """Open (and create if needed) a ranking store rooted at ``root``."""
        self.root = Path(root)
        self.versions_dir = self.root / "metric_versions"
        self.rankings_dir = self.root / "rankings"
        self.movement_dir = self.root / "movement"
        self.manifest_path = self.root / "manifest.json"
        self.annotations_path = self.root / "annotations.json"
        for d in (self.versions_dir, self.rankings_dir, self.movement_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── manifest ───────────────────────────────────────────────────────────

    def _manifest(self) -> dict:
        if self.manifest_path.exists():
            return io.load_json(self.manifest_path)
        return {"unit": None, "versions": []}

    def list_versions(self) -> list[str]:
        """Return version ids in creation order."""
        return [v["version_id"] for v in self._manifest().get("versions", [])]

    def unit(self) -> str | None:
        """Return the store's fixed unit, or None if no version exists yet."""
        return self._manifest().get("unit")

    def next_version_id(self) -> str:
        """Return the next monotonic version id (``v1``, ``v2`` …)."""
        existing = self.list_versions()
        return f"v{len(existing) + 1}"

    # ── metric versions ──────────────────────────────────────────────────--

    def version_path(self, version_id: str) -> Path:
        """Path to a metric-version JSON file."""
        return self.versions_dir / f"{version_id}.json"

    def ranking_path(self, version_id: str) -> Path:
        """Path to a ranking parquet file."""
        return self.rankings_dir / f"{version_id}.parquet"

    def write_metric_version(self, version: MetricVersion, unit: str) -> None:
        """Persist an immutable metric version and update the manifest.

        Raises if ``version_id`` already exists (immutability — FR-018) or if
        ``unit`` conflicts with the store's established unit.
        """
        path = self.version_path(version.version_id)
        if path.exists():
            raise FileExistsError(f"metric version {version.version_id} already exists (immutable)")

        manifest = self._manifest()
        if manifest.get("unit") not in (None, unit):
            raise ValueError(f"store unit is {manifest['unit']!r}; cannot add unit {unit!r}")

        recal = version.recal
        payload = {
            "version_id": version.version_id,
            "origin": version.origin,
            "parent_version_id": version.parent_version_id,
            "created_at": version.created_at,
            "definition": _defn_to_dict(version.definition),
            "recal": (
                {
                    "status": recal.status,
                    "n_annotations_used": recal.n_annotations_used,
                    "n_pairs": recal.n_pairs,
                    "n_distinct_levels": recal.n_distinct_levels,
                    "agreement_before": recal.agreement_before,
                    "agreement_after": recal.agreement_after,
                    "message": recal.message,
                }
                if recal is not None
                else None
            ),
        }
        io.save_json(path, payload)

        manifest["unit"] = unit
        manifest.setdefault("versions", []).append(
            {
                "version_id": version.version_id,
                "origin": version.origin,
                "parent_version_id": version.parent_version_id,
                "created_at": version.created_at,
            }
        )
        io.save_json(self.manifest_path, manifest)

    def read_metric_version(self, version_id: str) -> MetricVersion:
        """Load a previously written metric version."""
        data = io.load_json(self.version_path(version_id))
        recal_data = data.get("recal")
        recal = (
            RecalibrationResult(
                status=recal_data["status"],
                proposed_definition=None,
                n_annotations_used=recal_data["n_annotations_used"],
                n_pairs=recal_data["n_pairs"],
                n_distinct_levels=recal_data["n_distinct_levels"],
                agreement_before=recal_data["agreement_before"],
                agreement_after=recal_data["agreement_after"],
                message=recal_data.get("message", ""),
            )
            if recal_data is not None
            else None
        )
        return MetricVersion(
            version_id=data["version_id"],
            definition=_defn_from_dict(data["definition"]),
            origin=data["origin"],
            parent_version_id=data.get("parent_version_id"),
            created_at=data.get("created_at", ""),
            recal=recal,
        )
