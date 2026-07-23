"""Concrete metadata provider for the Bridge2AI-Voice (b2aiprep) BIDS dataset.

This is intentionally **specific to the b2ai-voice v3.x layout**, not a general BIDS parser.
Given an incoming audio reference, it resolves the recording's ``recording_id`` / task, and the
speaker's age and gold-standard diagnosis (GSD), from the standardized dataset:

```
<root>/
  phenotype/
    demographics/demographics.tsv        # participant_id, age, ...
    diagnosis/<condition>.tsv            # participant_id, ..._gold_standard_diagnosis
  sub-<uuid>/ses-<UUID>/audio/
    sub-<uuid>_ses-<UUID>_task-<Name>.wav
    sub-<uuid>_ses-<UUID>_task-<Name>_recording-metadata.json   # recording_id, task_name, prompts
```

Field facts confirmed against the dataset schema (v3.1 adult):

* ``participant_id`` is the bare UUID (no ``sub-`` prefix) -- the join key.
* the GSD column in each ``diagnosis/<condition>.tsv`` ends in ``gold_standard_diagnosis``
  (stems vary per condition; ``control`` has none).
* the sidecar JSON carries ``recording_id``, ``task_name`` and ``prompts`` (a list).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Optional

from senselab_ls.common.audio_plus import AudioPlusMetadata, SpeakerInfo, TaskInfo

GSD_COLUMN_SUFFIX = "gold_standard_diagnosis"
PARTICIPANT_ID_COLUMN = "participant_id"
AGE_COLUMN = "age"
RECORDING_SIDECAR_SUFFIX = "_recording-metadata.json"
# Values in a GSD cell that mean "not this diagnosis" (documented assumption; raw values are
# also preserved in SpeakerInfo.metadata for the caller to interpret).
_NEGATIVE_GSD_VALUES = {"", "0", "no", "n", "false", "na", "n/a", "none", "unknown", "not applicable"}

_ENTITY_RE = re.compile(r"sub-(?P<sub>[^_/]+)_ses-(?P<ses>[^_/]+)_task-(?P<task>.+?)(?:_recording-metadata)?\.[^.]+$")


class B2AIMetadataProvider:
    """Resolve b2ai-voice recording/speaker metadata for an incoming audio reference.

    Args:
        dataset_root: Path to the BIDS dataset root (the directory holding ``phenotype/`` and
            the ``sub-*`` folders).
        include_related: When ``True`` (default), populate ``related_audio_refs`` with the
            speaker's other recordings.
    """

    def __init__(self, dataset_root: str, *, include_related: bool = True) -> None:
        """Store the dataset root and prepare lazy phenotype caches."""
        self.root = Path(dataset_root)
        self.include_related = include_related
        self._age_by_participant: Optional[dict[str, Optional[str]]] = None
        self._gsd_by_participant: Optional[dict[str, list[str]]] = None

    def lookup(self, ref: str) -> AudioPlusMetadata:
        """Return the joined metadata for the recording referenced by ``ref``.

        Args:
            ref: The audio reference (path, ``s3://`` key, or bare filename). Only its basename
                is parsed for the ``sub-``/``ses-``/``task-`` entities; files are read from
                ``dataset_root``.

        Returns:
            An :class:`AudioPlusMetadata` with recording_id, task, speaker (age + GSD), and
            related recordings. Missing pieces are left as ``None`` / empty rather than raising.
        """
        match = _ENTITY_RE.search(Path(ref).name)
        if match is None:
            return AudioPlusMetadata()
        sub, ses, task_entity = match.group("sub"), match.group("ses"), match.group("task")

        sidecar = self._read_sidecar(sub, ses, task_entity)
        recording_id = sidecar.get("recording_id")
        task = TaskInfo(
            name=sidecar.get("task_name") or task_entity,
            content=self._join_prompts(sidecar.get("prompts")),
        )
        gsd_labels = self._gsd_labels().get(sub, [])
        speaker = SpeakerInfo(
            speaker_id=sub,
            age=self._parse_age(self._age_map().get(sub)),
            gsd=", ".join(gsd_labels) or None,
            metadata={"gsd_conditions": gsd_labels},
        )
        related = self._related_refs(sub, ref) if self.include_related else []
        return AudioPlusMetadata(recording_id=recording_id, task=task, speaker=speaker, related_audio_refs=related)

    # -- sidecar --------------------------------------------------------------------------

    def _read_sidecar(self, sub: str, ses: str, task_entity: str) -> dict:
        """Read the recording sidecar JSON for one recording; empty dict when absent.

        Args:
            sub: Subject UUID (no ``sub-`` prefix).
            ses: Session id.
            task_entity: The ``task-`` label from the filename.

        Returns:
            The parsed sidecar dict, or ``{}`` if the file is missing/unreadable.
        """
        stem = f"sub-{sub}_ses-{ses}_task-{task_entity}"
        path = self.root / f"sub-{sub}" / f"ses-{ses}" / "audio" / f"{stem}{RECORDING_SIDECAR_SUFFIX}"
        if not path.is_file():
            return {}
        with path.open() as handle:
            return json.load(handle)

    @staticmethod
    def _join_prompts(prompts: object) -> Optional[str]:
        """Join the sidecar ``prompts`` list into a single content string.

        Args:
            prompts: The sidecar ``prompts`` value (usually a list of strings).

        Returns:
            The joined prompt text, or ``None`` when empty.
        """
        if isinstance(prompts, list):
            text = " ".join(str(p) for p in prompts if p)
            return text or None
        if isinstance(prompts, str):
            return prompts or None
        return None

    # -- demographics (age) ---------------------------------------------------------------

    def _age_map(self) -> dict[str, Optional[str]]:
        """Return a cached ``participant_id -> raw age`` map from ``demographics.tsv``."""
        if self._age_by_participant is None:
            self._age_by_participant = self._read_column_map(
                self.root / "phenotype" / "demographics" / "demographics.tsv", AGE_COLUMN
            )
        return self._age_by_participant

    def _read_column_map(self, tsv_path: Path, column: str) -> dict[str, Optional[str]]:
        """Map ``participant_id -> value of column`` from a phenotype TSV.

        Args:
            tsv_path: Path to the TSV file.
            column: The column to extract.

        Returns:
            A dict keyed by participant id; empty if the file/column is absent.
        """
        out: dict[str, Optional[str]] = {}
        if not tsv_path.is_file():
            return out
        with tsv_path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or column not in reader.fieldnames:
                return out
            for row in reader:
                pid = row.get(PARTICIPANT_ID_COLUMN)
                if pid:
                    out[pid] = row.get(column)
        return out

    @staticmethod
    def _parse_age(raw: Optional[str]) -> Optional[float]:
        """Coerce a raw age cell to ``float``; ``None`` when blank/non-numeric.

        Args:
            raw: The raw ``age`` cell value.

        Returns:
            The age as a float, or ``None``.
        """
        if raw is None or not str(raw).strip():
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    # -- diagnosis (GSD) ------------------------------------------------------------------

    def _gsd_labels(self) -> dict[str, list[str]]:
        """Return a cached ``participant_id -> [condition labels]`` map for positive GSDs."""
        if self._gsd_by_participant is None:
            self._gsd_by_participant = self._build_gsd_map()
        return self._gsd_by_participant

    def _build_gsd_map(self) -> dict[str, list[str]]:
        """Scan ``phenotype/diagnosis/*.tsv`` and collect each participant's positive GSDs.

        For every ``<condition>.tsv`` the column ending in ``gold_standard_diagnosis`` is read;
        a participant is credited with that condition when the cell is affirmative.

        Returns:
            A dict keyed by participant id whose values are condition-file stems (e.g.
            ``"parkinsons_disease"``).
        """
        out: dict[str, list[str]] = {}
        diagnosis_dir = self.root / "phenotype" / "diagnosis"
        if not diagnosis_dir.is_dir():
            return out
        for tsv_path in sorted(diagnosis_dir.glob("*.tsv")):
            condition = tsv_path.stem
            with tsv_path.open(newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                fields = reader.fieldnames or []
                gsd_columns = [c for c in fields if c.endswith(GSD_COLUMN_SUFFIX)]
                if not gsd_columns:
                    continue
                gsd_column = gsd_columns[0]
                for row in reader:
                    pid = row.get(PARTICIPANT_ID_COLUMN)
                    if pid and self._is_affirmative(row.get(gsd_column)):
                        out.setdefault(pid, []).append(condition)
        return out

    @staticmethod
    def _is_affirmative(value: Optional[str]) -> bool:
        """Whether a GSD cell counts as a positive diagnosis (documented heuristic).

        Args:
            value: The raw GSD cell value.

        Returns:
            ``True`` when the cell is non-empty and not a known negative sentinel.
        """
        if value is None:
            return False
        return value.strip().lower() not in _NEGATIVE_GSD_VALUES

    # -- related recordings ---------------------------------------------------------------

    def _related_refs(self, sub: str, current_ref: str) -> list[str]:
        """Return the speaker's other recording paths (for profile building).

        Args:
            sub: Subject UUID (no ``sub-`` prefix).
            current_ref: The reference being looked up, excluded from the result.

        Returns:
            Absolute paths of the participant's other ``.wav`` recordings.
        """
        subject_dir = self.root / f"sub-{sub}"
        if not subject_dir.is_dir():
            return []
        current_name = Path(current_ref).name
        return [str(p) for p in sorted(subject_dir.glob("ses-*/audio/*.wav")) if p.name != current_name]
