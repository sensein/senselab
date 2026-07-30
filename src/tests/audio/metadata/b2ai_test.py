"""Unit tests for B2AIMetadataProvider against a synthetic b2ai-like fixture.

No real dataset is touched: a tiny directory mirroring the b2ai-voice v3.x layout is built in
a tmp path (empty .wav placeholders; the provider never loads audio).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from senselab.audio.metadata.b2ai import B2AIMetadataProvider

_SUB = "abc12345-0000-0000-0000-000000000001"
_SES = "SES-0001"


def _write(path: Path, text: str) -> None:
    """Create parent dirs and write ``text`` to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    """Build a minimal b2ai-like dataset and return its root."""
    root = tmp_path / "bids"
    audio = root / f"sub-{_SUB}" / f"ses-{_SES}" / "audio"
    stem = f"sub-{_SUB}_ses-{_SES}_task-Prolonged-vowel"
    other = f"sub-{_SUB}_ses-{_SES}_task-Caterpillar-Passage"

    _write(audio / f"{stem}.wav", "")
    _write(audio / f"{other}.wav", "")
    _write(
        audio / f"{stem}_recording-metadata.json",
        json.dumps(
            {
                "recording_id": "REC-123",
                "task_name": "prolonged-vowel",
                "session_id": _SES,
                "prompts": ["Say ahh for as long as you can"],
            }
        ),
    )
    _write(
        root / "phenotype" / "demographics" / "demographics.tsv",
        f"participant_id\tage\tgender_identity\n{_SUB}\t71\tfemale\n",
    )
    _write(
        root / "phenotype" / "diagnosis" / "parkinsons_disease.tsv",
        f"participant_id\tdiagnosis_parkinsons_gold_standard_diagnosis\n{_SUB}\tyes\n",
    )
    _write(
        root / "phenotype" / "diagnosis" / "depression.tsv",
        f"participant_id\tdiagnosis_dmdd_gold_standard_diagnosis\n{_SUB}\tno\n",
    )
    _write(
        root / "phenotype" / "diagnosis" / "anxiety.tsv",
        f"participant_id\tdiagnosis_ad_gold_standard_diagnosis\n{_SUB}\tnotCertain\n",
    )
    _write(
        root / "phenotype" / "diagnosis" / "copd_and_asthma.tsv",
        f"participant_id\tdiagnosis_ca_copd_asthma_gold_standard_diagnosis\n{_SUB}\tbothCopdAsthma\n",
    )
    _write(root / "phenotype" / "diagnosis" / "control.tsv", f"participant_id\tdiagnosis_c_ac\n{_SUB}\t0\n")
    return root


def test_lookup_resolves_recording_and_task(dataset_root: Path) -> None:
    """recording_id, task name and prompt content come from the sidecar."""
    provider = B2AIMetadataProvider(str(dataset_root))
    ref = dataset_root / f"sub-{_SUB}" / f"ses-{_SES}" / "audio" / f"sub-{_SUB}_ses-{_SES}_task-Prolonged-vowel.wav"

    meta = provider.lookup(str(ref))

    assert meta.recording_id == "REC-123"
    assert meta.task.name == "prolonged-vowel"
    assert meta.task.content == "Say ahh for as long as you can"


def test_lookup_resolves_age_and_positive_gsd(dataset_root: Path) -> None:
    """Age parses to float; affirmative GSDs (incl. multi-category copd) reported, negatives not."""
    provider = B2AIMetadataProvider(str(dataset_root))
    ref = f"sub-{_SUB}_ses-{_SES}_task-Prolonged-vowel.wav"  # bare basename also works

    meta = provider.lookup(ref)

    assert meta.speaker.speaker_id == _SUB
    assert meta.speaker.age == 71.0
    # parkinsons ("yes") and copd_and_asthma ("bothCopdAsthma") are positive;
    # depression ("no"), anxiety ("notCertain") and control (no GSD col) are not.
    assert meta.speaker.metadata["gsd_conditions"] == ["copd_and_asthma", "parkinsons_disease"]
    assert meta.speaker.gsd == "copd_and_asthma, parkinsons_disease"
    assert meta.speaker.metadata["gsd_details"]["copd_and_asthma"] == "bothCopdAsthma"


def test_lookup_lists_related_recordings(dataset_root: Path) -> None:
    """Related refs are the speaker's other recordings, excluding the current one."""
    provider = B2AIMetadataProvider(str(dataset_root))
    ref = f"sub-{_SUB}_ses-{_SES}_task-Prolonged-vowel.wav"

    meta = provider.lookup(ref)

    assert len(meta.related_audio_refs) == 1
    assert meta.related_audio_refs[0].endswith("task-Caterpillar-Passage.wav")


def test_unparseable_ref_returns_empty_metadata(dataset_root: Path) -> None:
    """A ref without BIDS entities yields empty (non-raising) metadata."""
    meta = B2AIMetadataProvider(str(dataset_root)).lookup("not-a-bids-name.wav")
    assert meta.recording_id is None
    assert meta.speaker.age is None
