"""Manifests built from datasets: identity from content only, binding by name, sidecars outside it."""

from pathlib import Path

import pytest
import soundfile
from bids_tree_fixture import make_tree, write_audio
from senselab_core.kernel.dataset import InputManifest
from senselab_core.runtime.bids import bids_dataset
from senselab_core.runtime.datasets import build_manifest, from_records, record_bind


def _manifest(root: Path) -> InputManifest:
    return build_manifest(bids_dataset(root), {"audio": "Audio"}, bind={"audio": "audio"})


def _rewrite_sample(target: Path) -> None:
    data, rate = soundfile.read(target, dtype="float32")
    data[0] += 0.25
    soundfile.write(target, data, rate, subtype="FLOAT")


@pytest.mark.logic
def test_manifest_digest_changes_with_content_only(tmp_path: Path) -> None:
    """Moving the root leaves the digest, whatever sits at the root; rewriting one sample changes it."""
    root = make_tree(tmp_path / "a")
    write_audio(root / "task-read_audio.wav", 5)
    before = _manifest(root).digest()
    moved = root.rename(tmp_path / "b")
    assert _manifest(moved).digest() == before
    _rewrite_sample(moved / "sub-02" / "ses-1" / "audio" / "sub-02_ses-1_task-read_run-1_audio.wav")
    assert _manifest(moved).digest() != before


@pytest.mark.logic
def test_in_memory_file_rows_are_stamped_from_content(tmp_path: Path) -> None:
    """A path in an in-memory row, as `Path` or text, is read for an `Audio` port: rewriting it changes the digest."""
    write_audio(tmp_path / "a.wav", 1)
    write_audio(tmp_path / "b.wav", 2)
    for ref in (tmp_path / "a.wav", str(tmp_path / "a.wav")):
        data = from_records(
            [{"id": "r", "inputs": {"audio": ref}}, {"id": "s", "inputs": {"audio": tmp_path / "b.wav"}}]
        )
        before = build_manifest(data, {"audio": "Audio"}).digest()
        _rewrite_sample(tmp_path / "a.wav")
        assert build_manifest(data, {"audio": "Audio"}).digest() != before
        write_audio(tmp_path / "a.wav", 1)


@pytest.mark.logic
def test_binding_by_name_and_override(tmp_path: Path) -> None:
    """A port binds to the input of its name; `bind` names another; a missing input names the port and the record."""
    write_audio(tmp_path / "a.wav", 1)
    write_audio(tmp_path / "b.wav", 2)
    data = from_records([{"id": "r", "inputs": {"audio": tmp_path / "a.wav", "other": tmp_path / "b.wav"}}])
    by_name = build_manifest(data, {"audio": "Audio"})
    other = build_manifest(data, {"audio": "Audio"}, bind={"audio": "other"})
    assert by_name.digest() != other.digest()
    assert by_name.rows[0].inputs["audio"] != other.rows[0].inputs["audio"]
    with pytest.raises(ValueError, match=r"record 'r'.*'speech'.*\['audio', 'other'\]"):
        build_manifest(data, {"speech": "Audio"})


@pytest.mark.logic
def test_a_single_port_binds_to_the_records_single_input(tmp_path: Path) -> None:
    """One port named for no record input reads the one input every record has, as `bind` naming it would."""
    data = bids_dataset(make_tree(tmp_path / "a"))
    assert record_bind(data, ["series"]) == {"series": "audio"}
    auto = build_manifest(data, {"series": "RegularSeries"})
    assert auto.digest() == build_manifest(data, {"series": "RegularSeries"}, bind={"series": "audio"}).digest()


@pytest.mark.logic
def test_an_ambiguous_binding_names_the_bind_to_pass(tmp_path: Path) -> None:
    """Two unbound ports, or a record with two inputs, bind nothing and name the `bind` that would."""
    write_audio(tmp_path / "a.wav", 1)
    single = from_records([{"id": "r", "inputs": {"audio": tmp_path / "a.wav"}}])
    double = from_records([{"id": "r", "inputs": {"audio": tmp_path / "a.wav", "other": tmp_path / "a.wav"}}])
    with pytest.raises(
        ValueError, match=r"no input 'a', 'b'; record inputs are \['audio'\]; pass bind=\{'a': 'audio', 'b': 'audio'\}"
    ):
        build_manifest(single, {"a": "Audio", "b": "Audio"})
    with pytest.raises(
        ValueError, match=r"no input 'series'; record inputs are \['audio', 'other'\]; pass bind=\{'series': 'audio'\}"
    ):
        build_manifest(double, {"series": "Audio"})


@pytest.mark.logic
def test_sidecar_not_in_manifest_identity(tmp_path: Path) -> None:
    """Editing a sidecar changes the records and not the manifest."""
    root = make_tree(tmp_path / "a")
    before = _manifest(root).digest()
    sidecars = [r.sidecar for r in bids_dataset(root).records()]
    (root / "task-read_audio.json").write_text('{"Instruction": "other"}', encoding="utf-8")
    (root / "participants.tsv").write_text("participant_id\tage\nsub-01\t99\nsub-02\t98\n", encoding="utf-8")
    assert [r.sidecar for r in bids_dataset(root).records()] != sidecars
    assert _manifest(root).digest() == before
