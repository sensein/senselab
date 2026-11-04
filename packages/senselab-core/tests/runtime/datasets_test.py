"""Manifests built from datasets: identity from content only, sidecars outside it."""

from pathlib import Path

import pytest
import soundfile
from bids_tree_fixture import make_tree
from senselab_core.kernel.dataset import InputManifest
from senselab_core.runtime.bids import bids_dataset
from senselab_core.runtime.datasets import build_manifest


def _manifest(root: Path) -> InputManifest:
    return build_manifest(bids_dataset(root), {"audio": "audio"})


@pytest.mark.logic
def test_manifest_digest_changes_with_content_only(tmp_path: Path) -> None:
    """Moving the root leaves the digest; rewriting one sample changes it."""
    root = make_tree(tmp_path / "a")
    before = _manifest(root).digest()
    moved = root.rename(tmp_path / "b")
    assert _manifest(moved).digest() == before
    target = moved / "sub-02" / "ses-1" / "audio" / "sub-02_ses-1_task-read_run-1_audio.wav"
    data, rate = soundfile.read(target, dtype="float32")
    data[0] += 0.25
    soundfile.write(target, data, rate, subtype="FLOAT")
    assert _manifest(moved).digest() != before


@pytest.mark.logic
def test_sidecar_not_in_manifest_identity(tmp_path: Path) -> None:
    """Editing a sidecar changes the records and not the manifest."""
    root = make_tree(tmp_path / "a")
    before = _manifest(root).digest()
    (root / "task-read_audio.json").write_text('{"Instruction": "other"}', encoding="utf-8")
    (root / "participants.tsv").write_text("participant_id\tage\nsub-01\t99\nsub-02\t98\n", encoding="utf-8")
    assert _manifest(root).digest() == before
