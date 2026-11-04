"""The BIDS adapter: entities, sidecars, ignored files, collisions and links."""

import os
from pathlib import Path

import pytest
from bids_tree_fixture import make_tree, write_audio
from senselab_core.runtime.bids import BidsCollision, BidsDataset, BidsError, bids_dataset


@pytest.mark.logic
def test_bids_entities_parsed_once(tmp_path: Path) -> None:
    """Entities come from the file name; datatype and suffix from the path; the sidecar nests, nearer winning."""
    data = bids_dataset(make_tree(tmp_path / "bids"))
    records = {r.id: r for r in data.records()}
    first = records["sub-01/ses-1/audio/sub-01_ses-1_task-read_run-1_audio.wav"]
    assert dict(first.entities) == {
        "sub": "01",
        "ses": "1",
        "task": "read",
        "run": "1",
        "datatype": "audio",
        "suffix": "audio",
    }
    assert list(first.inputs) == ["audio"]
    assert dict(first.sidecar) == {"age": "30", "Instruction": "read", "Gain": 2}
    other = records["sub-01/ses-1/audio/sub-01_ses-1_task-read_run-2_audio.wav"]
    assert other.sidecar["Gain"] == 1
    assert data.entities()["sub"] == ("01", "02")
    assert len(list(data.filter(sub="01", run=["1"]).records())) == 2


@pytest.mark.logic
def test_bids_ignores_non_bids_files(tmp_path: Path) -> None:
    """An audio file outside the naming grammar is skipped and counted; a text file is not read."""
    data = bids_dataset(make_tree(tmp_path / "bids"))
    assert isinstance(data, BidsDataset)
    assert data.ignored == 1
    assert len(list(data.records())) == 4
    assert len(list(bids_dataset(tmp_path / "bids", suffix="other").records())) == 0


@pytest.mark.logic
def test_bids_caseless_collision_refused(tmp_path: Path) -> None:
    """Two files differing only in case are refused."""
    root = make_tree(tmp_path / "bids")
    write_audio(root / "sub-02" / "sub-02_task-a_audio.wav", 5)
    write_audio(root / "sub-02" / "SUB-02_task-a_audio.wav", 6)
    names = {p.name for p in (root / "sub-02").iterdir()}
    if len(names & {"sub-02_task-a_audio.wav", "SUB-02_task-a_audio.wav"}) < 2:
        pytest.skip("the file system folds case")
    with pytest.raises(BidsCollision):
        bids_dataset(root)


@pytest.mark.logic
def test_bids_symlink_out_of_root_refused(tmp_path: Path) -> None:
    """A link whose target leaves the root is refused."""
    root = make_tree(tmp_path / "bids")
    outside = tmp_path / "outside.wav"
    write_audio(outside, 7)
    os.symlink(outside, root / "sub-02" / "sub-02_task-b_audio.wav")
    with pytest.raises(BidsError, match="links outside"):
        bids_dataset(root)
