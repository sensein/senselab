"""Concrete metadata provider for the Bridge2AI-Voice (b2aiprep) BIDS dataset.

Intentionally **specific to the b2ai-voice v3.x layout**, not a general BIDS parser. Given an
incoming audio reference, it resolves the recording's ``recording_id`` / task, and the speaker's
age and gold-standard diagnosis (GSD), from the standardized dataset:

```
<root>/
  phenotype/
    demographics/demographics.tsv        # participant_id, age, ...
    diagnosis/<condition>.tsv            # participant_id, ..._gold_standard_diagnosis
    diagnosis/<condition>.json           # data dictionary (documents the GSD value vocab)
  sub-<uuid>/ses-<UUID>/audio/
    sub-<uuid>_ses-<UUID>_task-<Name>.wav
    sub-<uuid>_ses-<UUID>_task-<Name>_recording-metadata.json   # recording_id, task_name, prompts
```

Schema facts confirmed against v3.1 adult:

* ``participant_id`` is the bare UUID (no ``sub-`` prefix) -- the join key.
* the GSD column in each ``diagnosis/<condition>.tsv`` ends in ``gold_standard_diagnosis``
  (stems vary per condition; ``control`` has none).
* GSD values are a controlled vocab (from the ``.json`` data dictionaries): ``yes`` / ``no`` /
  ``notCertain`` for most conditions, and ``copd`` / ``asthma`` / ``bothCopdAsthma`` /
  ``neitherCopdAsthma`` / ``notCertain`` for ``copd_and_asthma``. "Affirmative" = present and not
  one of the negative/uncertain values (so ``copd``/``asthma``/``bothCopdAsthma`` count).
* the sidecar JSON carries ``recording_id``, ``task_name`` and ``prompts`` (a list).

The dataset root may be a local directory or an ``s3://bucket/prefix`` location (read via boto3).
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from senselab_ls.common.audio_plus import AudioPlusMetadata, SpeakerInfo, TaskInfo

GSD_COLUMN_SUFFIX = "gold_standard_diagnosis"
PARTICIPANT_ID_COLUMN = "participant_id"
AGE_COLUMN = "age"
RECORDING_SIDECAR_SUFFIX = "_recording-metadata.json"
# GSD values that mean "not (confidently) this diagnosis" -- the vocab is documented in the
# phenotype diagnosis ``.json`` data dictionaries. Anything else non-empty counts as affirmative,
# which correctly keeps the copd_and_asthma positives (copd / asthma / bothCopdAsthma).
_GSD_NEGATIVE_VALUES = {"", "no", "notcertain", "neithercopdasthma", "unknown", "na", "n/a", "none"}

_ENTITY_RE = re.compile(r"sub-(?P<sub>[^_/]+)_ses-(?P<ses>[^_/]+)_task-(?P<task>.+?)(?:_recording-metadata)?\.[^.]+$")


class _FileStore:
    """Read text and list files under a dataset root that is a local dir or an ``s3://`` prefix."""

    def __init__(self, root: str) -> None:
        """Parse ``root`` and pick the local or S3 backend."""
        self.root = root
        self.is_s3 = root.startswith("s3://")
        if self.is_s3:
            parsed = urlparse(root)
            self._bucket = parsed.netloc
            self._prefix = parsed.path.strip("/")
        else:
            self._base = Path(root)

    def read_text(self, relpath: str) -> Optional[str]:
        """Return the UTF-8 contents of ``relpath`` under the root, or ``None`` if missing.

        Args:
            relpath: Path relative to the dataset root.

        Returns:
            File contents, or ``None`` when the file does not exist.
        """
        if self.is_s3:
            import boto3

            try:
                obj = boto3.client("s3").get_object(Bucket=self._bucket, Key=self._key(relpath))
            except Exception:  # noqa: BLE001 -- any S3/client error means "absent" here
                return None
            return obj["Body"].read().decode("utf-8")
        path = self._base / relpath
        return path.read_text() if path.is_file() else None

    def list(self, prefix: str, suffix: str) -> list[str]:
        """List files (recursively) under ``prefix`` whose name ends with ``suffix``.

        Args:
            prefix: Directory relative to the root to search under.
            suffix: Filename suffix filter (e.g. ``".tsv"``).

        Returns:
            Sorted root-relative paths.
        """
        if self.is_s3:
            import boto3

            client = boto3.client("s3")
            full_prefix = f"{self._key(prefix).rstrip('/')}/"
            out: list[str] = []
            token: Optional[str] = None
            while True:
                kwargs = {"Bucket": self._bucket, "Prefix": full_prefix}
                if token:
                    kwargs["ContinuationToken"] = token
                resp = client.list_objects_v2(**kwargs)
                for item in resp.get("Contents", []):
                    key = item["Key"]
                    if key.endswith(suffix):
                        out.append(key[len(self._prefix) + 1 :] if self._prefix else key)
                if resp.get("IsTruncated"):
                    token = resp.get("NextContinuationToken")
                else:
                    break
            return sorted(out)
        base = self._base / prefix
        if not base.is_dir():
            return []
        return sorted(str(p.relative_to(self._base)) for p in base.rglob("*") if p.name.endswith(suffix))

    def join(self, relpath: str) -> str:
        """Return a loadable reference (absolute path or ``s3://`` URI) for ``relpath``."""
        if self.is_s3:
            return f"s3://{self._bucket}/{self._key(relpath)}"
        return str(self._base / relpath)

    def _key(self, relpath: str) -> str:
        """Join the S3 prefix with ``relpath`` into a full object key."""
        return f"{self._prefix}/{relpath}" if self._prefix else relpath


class B2AIMetadataProvider:
    """Resolve b2ai-voice recording/speaker metadata for an incoming audio reference.

    Args:
        dataset_root: Local directory or ``s3://bucket/prefix`` holding ``phenotype/`` and the
            ``sub-*`` folders.
        include_related: When ``True`` (default), populate ``related_audio_refs`` with the
            speaker's other recordings.
    """

    def __init__(self, dataset_root: str, *, include_related: bool = True) -> None:
        """Store the root (via a local/S3 file store) and prepare lazy phenotype caches."""
        self.store = _FileStore(dataset_root)
        self.include_related = include_related
        self._age_by_participant: Optional[dict[str, Optional[str]]] = None
        self._gsd_by_participant: Optional[dict[str, dict[str, Optional[str]]]] = None

    def lookup(self, ref: str) -> AudioPlusMetadata:
        """Return the joined metadata for the recording referenced by ``ref``.

        Args:
            ref: Audio reference (path, ``s3://`` key, or bare filename). Only its basename is
                parsed for the ``sub-``/``ses-``/``task-`` entities; files are read from the root.

        Returns:
            An :class:`AudioPlusMetadata` with recording_id, task, speaker (age + GSD) and related
            recordings. Missing pieces are left as ``None`` / empty rather than raising.
        """
        match = _ENTITY_RE.search(Path(ref).name)
        if match is None:
            return AudioPlusMetadata()
        sub, ses, task_entity = match.group("sub"), match.group("ses"), match.group("task")

        sidecar = self._read_sidecar(sub, ses, task_entity)
        task = TaskInfo(
            name=sidecar.get("task_name") or task_entity,
            content=self._join_prompts(sidecar.get("prompts")),
        )
        gsd_details = self._gsd_map().get(sub, {})
        conditions = sorted(gsd_details.keys())
        speaker = SpeakerInfo(
            speaker_id=sub,
            age=self._parse_age(self._age_map().get(sub)),
            gsd=", ".join(conditions) or None,
            metadata={"gsd_conditions": conditions, "gsd_details": gsd_details},
        )
        related = self._related_refs(sub, ref) if self.include_related else []
        return AudioPlusMetadata(
            recording_id=sidecar.get("recording_id"), task=task, speaker=speaker, related_audio_refs=related
        )

    # -- sidecar --------------------------------------------------------------------------

    def _read_sidecar(self, sub: str, ses: str, task_entity: str) -> dict:
        """Read the recording sidecar JSON for one recording; empty dict when absent.

        Args:
            sub: Subject UUID (no ``sub-`` prefix).
            ses: Session id.
            task_entity: The ``task-`` label from the filename.

        Returns:
            The parsed sidecar dict, or ``{}`` if missing.
        """
        stem = f"sub-{sub}_ses-{ses}_task-{task_entity}"
        text = self.store.read_text(f"sub-{sub}/ses-{ses}/audio/{stem}{RECORDING_SIDECAR_SUFFIX}")
        return json.loads(text) if text else {}

    @staticmethod
    def _join_prompts(prompts: object) -> Optional[str]:
        """Join the sidecar ``prompts`` list into a single content string.

        Args:
            prompts: The sidecar ``prompts`` value (usually a list of strings).

        Returns:
            The joined prompt text, or ``None`` when empty.
        """
        if isinstance(prompts, list):
            return " ".join(str(p) for p in prompts if p) or None
        if isinstance(prompts, str):
            return prompts or None
        return None

    # -- demographics (age) ---------------------------------------------------------------

    def _age_map(self) -> dict[str, Optional[str]]:
        """Return a cached ``participant_id -> raw age`` map from ``demographics.tsv``."""
        if self._age_by_participant is None:
            text = self.store.read_text("phenotype/demographics/demographics.tsv")
            self._age_by_participant = self._column_map_from_text(text, AGE_COLUMN) if text else {}
        return self._age_by_participant

    @staticmethod
    def _column_map_from_text(text: str, column: str) -> dict[str, Optional[str]]:
        """Map ``participant_id -> value of column`` from TSV ``text``.

        Args:
            text: The TSV contents.
            column: Column to extract.

        Returns:
            A dict keyed by participant id; empty if the column is absent.
        """
        out: dict[str, Optional[str]] = {}
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
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

    def _gsd_map(self) -> dict[str, dict[str, Optional[str]]]:
        """Return a cached ``participant_id -> {condition: raw GSD value}`` for positive GSDs."""
        if self._gsd_by_participant is None:
            self._gsd_by_participant = self._build_gsd_map()
        return self._gsd_by_participant

    def _build_gsd_map(self) -> dict[str, dict[str, Optional[str]]]:
        """Scan ``phenotype/diagnosis/*.tsv`` and collect each participant's positive GSDs.

        For every ``<condition>.tsv`` the column ending in ``gold_standard_diagnosis`` is read; a
        participant is credited with that condition when the cell is affirmative (see
        :meth:`_is_affirmative`).

        Returns:
            ``{participant_id: {condition_stem: raw_value}}`` for affirmative diagnoses.
        """
        out: dict[str, dict[str, Optional[str]]] = {}
        for relpath in self.store.list("phenotype/diagnosis", ".tsv"):
            condition = Path(relpath).stem
            text = self.store.read_text(relpath)
            if not text:
                continue
            reader = csv.DictReader(io.StringIO(text), delimiter="\t")
            gsd_columns = [c for c in (reader.fieldnames or []) if c.endswith(GSD_COLUMN_SUFFIX)]
            if not gsd_columns:
                continue
            gsd_column = gsd_columns[0]
            for row in reader:
                pid = row.get(PARTICIPANT_ID_COLUMN)
                value = row.get(gsd_column)
                if pid and self._is_affirmative(value):
                    out.setdefault(pid, {})[condition] = value
        return out

    @staticmethod
    def _is_affirmative(value: Optional[str]) -> bool:
        """Whether a GSD cell counts as a positive diagnosis (vocab-driven).

        Args:
            value: The raw GSD cell value.

        Returns:
            ``True`` when non-empty and not one of the documented negative/uncertain values.
        """
        if value is None:
            return False
        return value.strip().lower() not in _GSD_NEGATIVE_VALUES

    # -- related recordings ---------------------------------------------------------------

    def _related_refs(self, sub: str, current_ref: str) -> list[str]:
        """Return the speaker's other recording references (for profile building).

        Args:
            sub: Subject UUID (no ``sub-`` prefix).
            current_ref: The reference being looked up, excluded from the result.

        Returns:
            Loadable references (paths or ``s3://`` URIs) of the participant's other ``.wav``s.
        """
        current_name = Path(current_ref).name
        return [self.store.join(r) for r in self.store.list(f"sub-{sub}", ".wav") if Path(r).name != current_name]
