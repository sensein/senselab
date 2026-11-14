"""The BIDS adapter: entities, sidecars, ignored files, collisions and links."""

import os
from collections.abc import Iterator
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
        "extension": ".wav",
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


def _listing(monkeypatch: pytest.MonkeyPatch, root: Path, names: list[str]) -> None:
    """Make the walk of `root` report one `sub-01/audio` directory holding `names`."""
    resolved = root.resolve()

    def walk(top: Path, **_: object) -> Iterator[tuple[str, list[str], list[str]]]:
        yield str(resolved), ["sub-01"], []
        yield str(resolved / "sub-01"), ["audio"], []
        yield str(resolved / "sub-01" / "audio"), [], names

    monkeypatch.setattr(os, "walk", walk)


@pytest.mark.logic
@pytest.mark.parametrize(
    "names",
    [
        ["sub-01_task-a_audio.wav", "SUB-01_task-a_audio.wav"],
        ["sub-01_task-caf\u00e9_audio.wav", "sub-01_task-cafe\u0301_audio.wav"],
    ],
)
def test_bids_caseless_collision_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, names: list[str]) -> None:
    """Names differing only in case or Unicode form are refused, on any file system."""
    _listing(monkeypatch, tmp_path, names)
    with pytest.raises(BidsCollision):
        bids_dataset(tmp_path)


@pytest.mark.logic
def test_bids_symlink_out_of_root_refused(tmp_path: Path) -> None:
    """A link whose target leaves the root is refused."""
    root = make_tree(tmp_path / "bids")
    outside = tmp_path / "outside.wav"
    write_audio(outside, 7)
    os.symlink(outside, root / "sub-02" / "sub-02_task-b_audio.wav")
    with pytest.raises(BidsError, match="links outside"):
        bids_dataset(root)


@pytest.mark.logic
def test_bids_looping_or_dangling_link_refused(tmp_path: Path) -> None:
    """A link to itself or to nothing is a `BidsError`."""
    root = make_tree(tmp_path / "bids")
    link = root / "sub-02" / "ses-1" / "audio" / "sub-02_ses-1_task-b_audio.wav"
    os.symlink(link.name, link)
    with pytest.raises(BidsError, match="broken or looping"):
        bids_dataset(root)
    link.unlink()
    os.symlink("missing.wav", link)
    with pytest.raises(BidsError, match="broken or looping"):
        bids_dataset(root)


@pytest.mark.logic
def test_bids_reads_the_raw_tree_only(tmp_path: Path) -> None:
    """Derivatives, sourcedata and dot directories are not read, and a link out of one does not refuse the tree."""
    root = make_tree(tmp_path / "bids")
    write_audio(root / "derivatives" / "p" / "sub-01" / "audio" / "sub-01_desc-x_audio.wav", 11)
    write_audio(root / ".hidden" / "sub-01" / "audio" / "sub-01_task-h_audio.wav", 12)
    os.symlink(tmp_path, root / "derivatives" / "out")
    assert len(list(bids_dataset(root).records())) == 4


@pytest.mark.logic
def test_bids_datatype_follows_the_grammar_position(tmp_path: Path) -> None:
    """A grammar-named file outside `sub-*/[ses-*/]<datatype>/` is ignored; labels must match the directories."""
    root = make_tree(tmp_path / "bids")
    write_audio(root / "task-read_audio.wav", 1)
    write_audio(root / "sub-01" / "sub-01_task-x_audio.wav", 2)
    write_audio(root / "sub-02" / "audio" / "sub-03_task-x_audio.wav", 3)
    write_audio(root / "sub-02" / "audio" / "sub-02_task-x_audio.WAV", 4)
    data = bids_dataset(root)
    assert isinstance(data, BidsDataset)
    assert data.ignored == 4
    assert {r.entities["datatype"] for r in data.records()} == {"audio"}
    assert len(list(bids_dataset(root).records())) == 5


@pytest.mark.logic
def test_bids_ids_do_not_depend_on_creation_order(tmp_path: Path) -> None:
    """Two trees with the same files created in opposite orders give the same records."""
    names = [f"sub-0{i}/audio/sub-0{i}_task-t_audio.wav" for i in (1, 2, 3)]
    for tree, order in (("a", names), ("b", names[::-1])):
        for i, name in enumerate(order):
            write_audio(tmp_path / tree / name, i)
    one = [r.id for r in bids_dataset(tmp_path / "a").records()]
    two = [r.id for r in bids_dataset(tmp_path / "b").records()]
    assert one == two == sorted(names)


@pytest.mark.logic
def test_bids_malformed_sidecar_refused(tmp_path: Path) -> None:
    """A sidecar that is not a JSON object is a `BidsError` naming the file."""
    root = make_tree(tmp_path / "bids")
    (root / "task-read_audio.json").write_text("[1]", encoding="utf-8")
    with pytest.raises(BidsError, match="task-read_audio.json"):
        bids_dataset(root)


@pytest.mark.logic
@pytest.mark.parametrize("name", ["task-read_audio.json", "participants.tsv"])
def test_bids_metadata_link_out_of_root_refused(tmp_path: Path, name: str) -> None:
    """A sidecar or participants table linked from outside the root is refused, as an audio file is."""
    root = make_tree(tmp_path / "bids")
    outside = tmp_path / "outside"
    outside.write_text("{}" if name.endswith("json") else "participant_id\tage\n", encoding="utf-8")
    (root / name).unlink()
    os.symlink(outside, root / name)
    with pytest.raises(BidsError, match="links outside"):
        bids_dataset(root)
