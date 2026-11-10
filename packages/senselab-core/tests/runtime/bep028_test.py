"""Export redaction and the BEP028 profile: no document, export or provenance error holds a path."""

import json
import os
import re
import shutil
import stat
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from lease_time_fixture import install, on_create
from senselab_core.kernel import prov
from senselab_core.kernel.prov import Association, Generation, Usage
from senselab_core.runtime.bep028 import (
    LABELS_NAME,
    PRIVATE_DIR,
    LabelMap,
    LabelMapRetired,
    ProfileViolation,
    check_profile,
    export,
    file_entity,
    layout_name,
)
from senselab_core.runtime.execution_settings import JobLease
from senselab_core.runtime.fetch import DataManifest
from senselab_core.runtime.provenance import PROV_DIR, MalformedDocument, ProvStore, RunIOError
from senselab_core.runtime.vocabulary import core_vocabulary
from senselab_worker.canonical import MAX_DEPTH, jcs_dumps, jcs_loads
from syscall_faults_fixture import displace_before_proof

pytestmark = pytest.mark.logic

VOCAB = core_vocabulary()
T0 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
OUTPUT = "output:" + "2" * 64
COMPONENT = st.text(alphabet="ghjkmnpqrstuvwxyz", min_size=10, max_size=20)
LICENSE = {
    "expression": "CC-BY-4.0",
    "commercial_use": "yes",
    "copyleft": "none",
    "attribution": True,
    "gated": False,
    "source": "https://example.org/license",
    "verified": {"date": "2026-10-01", "how": "manual"},
}


BEP028_NAME = re.compile(r"prov-[0-9a-f]{32}_all\.jsonld")
DRAFT_NAME = re.compile(r"prov-[A-Za-z0-9]+_(act|ent|agt|soft|env|base|all)\.(json|jsonld)")
DRAFT_REQUIRED = ("@context", "BIDSProvVersion", "Records")
"""The top-level keys of a BEP028 draft document (bids-standard/BEP028_BIDSprov @162bea57, `bep028spec.md` 1.4)."""


def _manifest(*paths: str) -> DataManifest:
    return DataManifest.model_validate(
        {
            "format": "data-manifest@2.0",
            "name": "demo",
            "version": "1",
            "license": LICENSE,
            "terms": "Cite the source.",
            "files": [
                {"path": p, "url": "https://x/f", "sha256": f"{i:x}" * 64, "size": 1} for i, p in enumerate(paths, 1)
            ],
        }
    )


def _store(
    files: list[prov.Record], *, key: object = "k", ended: datetime | None = T0 + timedelta(seconds=1)
) -> ProvStore:
    agent = prov.agent(VOCAB, types=["senselab:SoftwareAgent"], attributes={"senselab:modelSource": "hf"})
    execution = prov.activity(
        VOCAB,
        attributes={"senselab:logicalKey": key},
        used=[Usage(f.id, port="audio") for f in files],
        generated=[Generation(OUTPUT, port="transcript")],
        associations=[Association(agent.id, "n1")],
        started=T0,
        ended=ended,
    )
    return ProvStore([execution, agent, prov.entity(OUTPUT, VOCAB), *files])


def _run(root: Path, *paths: str, key: object = "k") -> tuple[Path, ProvStore]:
    run = root / "run"
    run.mkdir(parents=True, exist_ok=True)
    store = _store(_manifest(*paths).file_entities(LabelMap(run)), key=key)
    store.write(run)
    return run, store


def _bytes_under(root: Path) -> list[bytes]:
    return [p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()]


def test_label_map_is_private(tmp_path: Path) -> None:
    """The map is `private/labels@1.json`, 0600 in a 0700 directory, one label per address, and never exported."""
    run, _ = _run(tmp_path, "sub-01/a.wav", "sub-02/b.wav")
    private = run / PRIVATE_DIR
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    assert stat.S_IMODE((private / LABELS_NAME).stat().st_mode) == 0o600
    assert LabelMap(run).paths() == {"file-1": ["sub-01/a.wav"], "file-2": ["sub-02/b.wav"]}
    assert LabelMap(run).assign([("content:" + "2" * 64, "other/c.wav")]) == ["file-2"]
    assert LabelMap(run).paths()["file-2"] == ["sub-02/b.wav", "other/c.wav"]
    dest = tmp_path / "out"
    export(run, dest)
    assert sorted(p.relative_to(dest).parts[0] for p in dest.rglob("*") if p.is_file()) == [PROV_DIR]
    assert all(b"sub-01" not in data for data in _bytes_under(dest))
    assert any(b"sub-01" in data for data in _bytes_under(private))


def test_a_label_map_that_is_not_labels1_is_refused(tmp_path: Path) -> None:
    """A foreign or damaged map is refused with an error that names no path."""
    run, _ = _run(tmp_path, "a.wav")
    (run / PRIVATE_DIR / LABELS_NAME).write_text('{"format": "labels@3.0", "labels": {}}')
    with pytest.raises(ValueError, match="labels@2.0") as caught:
        LabelMap(run).paths()
    assert str(run) not in str(caught.value)


def test_export_conforms_to_bep028(tmp_path: Path) -> None:
    """An export is the run's documents byte for byte, and its union meets the profile."""
    run, store = _run(tmp_path, "sub-01/a.wav", "sub-01/b.wav")
    copies = export(run, tmp_path / "out")
    sources = sorted((run / PROV_DIR).iterdir())
    assert [p.name for p in copies] == [layout_name(p.name) for p in sources]
    assert [p.read_bytes() for p in copies] == [p.read_bytes() for p in sources]
    assert all(BEP028_NAME.fullmatch(p.name) and p.parent.name == "prov" for p in copies)
    assert check_profile(_doc(run)) == []
    assert [r.id for r in ProvStore.read(run).records()] == [r.id for r in store.records()]
    files = [r for r in store.records() if "senselab:File" in r.types()]
    assert sorted(str(prov.literal_value(r.node()["senselab:fileLabel"])) for r in files) == ["file-1", "file-2"]


def _doc(run: Path) -> dict[str, Any]:
    doc: dict[str, Any] = jcs_loads(next((run / PROV_DIR).iterdir()).read_bytes())  # type: ignore[assignment]
    return doc


@pytest.mark.parametrize(
    ("edit", "mentions"),
    [
        (lambda d: d.update({"@context": "https://x/y"}), "@context"),
        (lambda d: d.update({"format": "provenance@2.0"}), "format"),
        (lambda d: d["@graph"][0].update({"@id": "a/b/c.wav"}), "@id"),
        (lambda d: d["@graph"].pop(1), "holds"),
        (
            lambda d: next(n for n in d["@graph"] if n["@id"].startswith("activity:")).update(
                {"prov:startedAtTime": "garbage"}
            ),
            "activity:",
        ),
        (
            lambda d: next(n for n in d["@graph"] if "senselab:fileLabel" in n).update(
                {"senselab:fileLabel": "sub-01"}
            ),
            "senselab:fileLabel",
        ),
    ],
)
def test_profile_violations_are_named(tmp_path: Path, edit: Any, mentions: str) -> None:  # noqa: ANN401
    """Each way a document leaves the profile is one named violation, and none echoes a value."""
    run, _ = _run(tmp_path, "sub-01/a.wav")
    doc = _doc(run)
    assert check_profile(doc) == []
    edit(doc)
    problems = check_profile(doc)
    assert problems and any(mentions in p for p in problems), problems
    assert not any("a/b/c.wav" in p or "sub-01" in p for p in problems)


@pytest.mark.parametrize(
    ("started", "ended", "mentions"),
    [(None, T0, "never starts"), (T0, T0 - timedelta(seconds=1), "before it starts")],
)
def test_activity_times_are_ordered(started: datetime | None, ended: datetime, mentions: str) -> None:
    """An activity that ends without starting, or before it starts, is named."""
    execution = prov.activity(VOCAB, started=started, ended=ended)
    problems = check_profile(ProvStore([execution]).document())
    assert [mentions in p for p in problems] == [True]
    assert check_profile(ProvStore([prov.activity(VOCAB, started=T0, ended=T0)]).document()) == []


def test_one_label_names_one_node(tmp_path: Path) -> None:
    """Two nodes carrying one label are refused."""
    run, _ = _run(tmp_path, "a.wav", "b.wav")
    doc = _doc(run)
    for node in doc["@graph"]:
        if "senselab:fileLabel" in node:
            node["senselab:fileLabel"] = "file-1"
    assert any("another node has" in p for p in check_profile(doc))


@pytest.mark.parametrize(
    "leak",
    [
        "ds/sub-01_task-x.wav",
        "/data/ds/sub-01_task-x.wav",
        "sub-01_task-x.wav",
        "sub-01_task-x.json",
        "./ds//sub-01_task-x.wav",
        "ds\\sub-01_task-x.wav",
        "read from sub-01_task-x.wav",
        {"ds/sub-01_task-x.wav": 1},
        {"audio": ["ds", {"sub-01_task-x.wav": 1}]},
    ],
)
def test_export_refuses_a_recorded_path_and_copies_nothing(tmp_path: Path, leak: object) -> None:
    """A document naming a recorded path, its basename or stem, however spelt, in a value or a key is not exported."""
    run, _ = _run(tmp_path, "ds/sub-01_task-x.wav", key=leak)
    with pytest.raises(ProfileViolation, match="label map"):
        export(run, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("path", "leak"),
    [
        ("ds/clip.wav", "Clip.wav"),
        ("ds/my clip.wav", "read my clip.wav"),
        ("ds/clip,1.wav", "clip,1.wav"),
        ("ds/cafe\u0301.wav", "caf\u00e9.wav"),
    ],
)
def test_export_refuses_a_name_in_another_case_unicode_form_or_with_separators(
    tmp_path: Path, path: str, leak: str
) -> None:
    """A basename is matched after NFC and casefolding, and whole, so a space or comma in it does not split it."""
    run, _ = _run(tmp_path, path, key=leak)
    with pytest.raises(ProfileViolation, match="label map"):
        export(run, tmp_path / "out")


@pytest.mark.parametrize(("path", "key", "port"), [("audio/clip.wav", "k", "audio"), ("k.wav", "k", "audio")])
def test_a_directory_segment_or_a_short_name_alone_is_exported(tmp_path: Path, path: str, key: str, port: str) -> None:
    """A port or key equal to a directory segment or a stem of a recorded path is not that path."""
    run, _ = _run(tmp_path, path, key={port: key})
    assert len(export(run, tmp_path / "out")) == 1


def test_export_refuses_a_bids_subject_that_was_never_recorded(tmp_path: Path) -> None:
    """A `sub-` label is a participant identifier wherever it appears, recorded or not."""
    run, _ = _run(tmp_path, "ds/a.wav", key="/data/sub-07/ses-1/x.wav")
    with pytest.raises(ProfileViolation):
        export(run, tmp_path / "out")


@pytest.mark.parametrize(
    "source", ["/home/user/models/m", "~/m", "./m", "../m", "C:\\m", "a\\b", "org/model", "hf:../m"]
)
def test_a_local_path_is_not_a_model_source(source: str) -> None:
    """`senselab:modelSource` is a source kind and an identity; a local path is refused at the store boundary."""
    with pytest.raises(ValueError, match="never a local path"):
        ProvStore([prov.agent(VOCAB, types=["senselab:SoftwareAgent"], attributes={"senselab:modelSource": source})])
    ProvStore(
        [
            prov.agent(
                VOCAB, types=["senselab:SoftwareAgent"], attributes={"senselab:modelSource": "org/model@" + "a" * 40}
            )
        ]
    )


def test_a_path_is_not_a_file_label(tmp_path: Path) -> None:
    """A path as `senselab:fileLabel` is refused by `add`, and by a read of a document that holds one."""
    entity = prov.entity(
        "content:" + "1" * 64, VOCAB, types=["senselab:File"], attributes={"senselab:fileLabel": "sub-01/a.wav"}
    )
    with pytest.raises(ValueError, match="file-<n>"):
        ProvStore([entity])
    run, _ = _run(tmp_path, "a.wav")
    path = next((run / PROV_DIR).iterdir())
    doc = _doc(run)
    next(n for n in doc["@graph"] if "senselab:fileLabel" in n)["senselab:fileLabel"] = "sub-01/a.wav"
    path.write_bytes(jcs_dumps(doc))
    with pytest.raises(MalformedDocument, match="fileLabel") as caught:
        ProvStore.read(run)
    assert "sub-01" not in str(caught.value)


def test_export_follows_the_bep028_draft(tmp_path: Path) -> None:
    """The draft's `prov/prov-<label>_<suffix>.<ext>` naming and required top-level keys, from its rule and example."""
    draft_names = ["prov-conversion_all.jsonld", "prov-dcm2niix_base.json", "prov-dcm2niix_soft.json"]
    assert all(DRAFT_NAME.fullmatch(n) for n in draft_names)
    run, _ = _run(tmp_path, "a.wav")
    copies = export(run, tmp_path / "out")
    assert all(DRAFT_NAME.fullmatch(p.name) and p.suffix == ".jsonld" for p in copies)
    doc: dict[str, Any] = jcs_loads(copies[0].read_bytes())  # type: ignore[assignment]
    assert {"@context", "@graph", "format"} <= set(doc)
    assert set(DRAFT_REQUIRED) - {"BIDSProvVersion", "Records"} <= set(doc)


def test_file_entity_label_is_file_n() -> None:
    """A label is `file-<n>`; a path is not one."""
    assert prov.literal_value(file_entity("content:" + "1" * 64, "file-3").node()["senselab:fileLabel"]) == "file-3"
    with pytest.raises(ValueError, match="file-<n>"):
        file_entity("content:" + "1" * 64, "sub-01/a.wav")


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(root=COMPONENT, session=COMPONENT, dataset=COMPONENT, name=COMPONENT, out=COMPONENT)
def test_document_holds_no_path(root: str, session: str, dataset: str, name: str, out: str) -> None:
    """Random path components never occur in a written document, an export, or an error of the provenance APIs."""
    parts = {root, session, dataset, name, out}
    with tempfile.TemporaryDirectory() as scratch:
        base = Path(scratch) / root
        run, _ = _run(base / session, f"{dataset}/{name}.wav")
        dest = Path(scratch) / "export" / out
        export(run, dest)
        for data in _bytes_under(run / PROV_DIR) + _bytes_under(dest):
            assert not any(p.encode() in data for p in parts)
        assert any(dataset.encode() in data for data in _bytes_under(run / PRIVATE_DIR))
        failures = _failures(base, run)
    for text in failures:
        assert not any(p in text for p in parts), text


def _failures(base: Path, run: Path) -> list[str]:
    """Return the text of each error a provenance or export API raises over paths holding the run's components."""
    broken = base / "broken"
    (broken / PROV_DIR).parent.mkdir(parents=True)
    (broken / PROV_DIR).write_text("not a directory")
    (broken / PRIVATE_DIR).write_text("not a directory")
    unreadable = base / "unreadable"
    (unreadable / PROV_DIR).mkdir(parents=True)
    shutil.copy(next((run / PROV_DIR).iterdir()), unreadable / PROV_DIR)
    (unreadable / PROV_DIR).chmod(0)
    texts = []
    try:
        for action in (
            lambda: ProvStore([prov.entity(OUTPUT, VOCAB)]).write(broken),
            lambda: LabelMap(broken).assign([("content:" + "1" * 64, "x.wav")]),
            lambda: LabelMap(broken).paths(),
        ):
            with pytest.raises(RunIOError) as caught:
                action()
            err = caught.value
            assert err.filename is None and err.__cause__ is None and err.__context__ is None
            texts += [str(err), repr(err), err.strerror or ""]
        for read in (ProvStore.read, lambda r: export(r, base / "never")):
            with pytest.raises(MalformedDocument) as refused:
                read(unreadable)
            assert refused.value.__context__ is None
            texts += [str(refused.value), repr(refused.value)]
    finally:
        (unreadable / PROV_DIR).chmod(0o700)
    assert os.fspath(run) not in "".join(texts)
    return texts


def test_label_map_is_jcs(tmp_path: Path) -> None:
    """The map on disk is canonical JSON: re-serialising what it parses to gives its bytes back."""
    run, _ = _run(tmp_path, "sub-01/a.wav", "sub-02/b.wav")
    data = (run / PRIVATE_DIR / LABELS_NAME).read_bytes()
    assert jcs_dumps(jcs_loads(data)) == data
    (run / PRIVATE_DIR / LABELS_NAME).write_bytes(json.dumps(jcs_loads(data), indent=2).encode())
    with pytest.raises(ValueError, match="labels@2.0"):
        LabelMap(run).paths()


def test_a_retired_label_map_is_refused_not_corrupt(tmp_path: Path) -> None:
    """A `labels@1.0` map, indented as it was written, is refused as retired, whatever else is wrong with it."""
    run, _ = _run(tmp_path, "a.wav")
    body = json.dumps({"format": "labels@1.0", "labels": {}}, indent=2)
    (run / PRIVATE_DIR / LABELS_NAME).write_text(body)
    with pytest.raises(LabelMapRetired, match="labels@1.0") as caught:
        LabelMap(run).paths()
    assert caught.value.found == "labels@1.0"


def test_a_deeply_nested_label_map_is_refused_typed(tmp_path: Path) -> None:
    """A retired map nested past the parser's depth is refused as retired, not as an untyped recursion failure."""
    run, _ = _run(tmp_path, "a.wav")
    depth = 100_000
    body = '{"format": "labels@1.0", "labels": ' + "[" * depth + "]" * depth + "}"
    (run / PRIVATE_DIR / LABELS_NAME).write_text(body)
    with pytest.raises(ValueError, match="labels@2.0"):
        LabelMap(run).paths()


def test_a_label_map_nested_past_the_bound_is_refused_whatever_the_parser_can_parse(tmp_path: Path) -> None:
    """The bound is the codec's `MAX_DEPTH`, not the parser's recursion limit: 65 levels parse fine and are refused."""
    run, _ = _run(tmp_path, "a.wav")
    depth = MAX_DEPTH + 1
    body = '{"format": "labels@1.0", "labels": ' + "[" * depth + "]" * depth + "}"
    json.loads(body)
    (run / PRIVATE_DIR / LABELS_NAME).write_text(body)
    with pytest.raises(ValueError, match="labels@2.0"):
        LabelMap(run).paths()


def test_the_label_map_lease_outlasts_its_stale_bound_while_it_beats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before the map is written the storage clock passes the stale bound; the lease beats, is never broken, writes."""
    lease = JobLease(stale_after_s=30, renew_fraction=0.01, attribute_delay_s=0)
    clock = install(monkeypatch)
    on_create(monkeypatch, f".{LABELS_NAME}.", lambda: clock.outlast(tmp_path / "private", lease.stale_after_s))
    assert LabelMap(tmp_path, lease=lease).assign([("content:" + "1" * 64, "a.wav")]) == ["file-1"]
    assert clock.attempts == 4 and clock.breaks == 0
    assert LabelMap(tmp_path).paths() == {"file-1": ["a.wav"]}


def test_a_label_map_holder_displaced_just_before_its_commit_does_not_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The map is committed only after a fresh `mkdir` proves the lease; a holder displaced before it writes nothing."""
    displaced = displace_before_proof(monkeypatch, f"{LABELS_NAME}.lease")
    with pytest.raises(RunIOError):
        LabelMap(tmp_path).assign([("content:" + "1" * 64, "a.wav")])
    assert displaced and not (tmp_path / "private" / LABELS_NAME).exists()
