"""Export redaction and the BEP028 profile: no document, error or log line holds a path; the label map is private."""

import copy
import logging
import os
import stat
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from senselab_core.kernel import prov
from senselab_core.kernel.prov import Association, Generation, Usage
from senselab_core.runtime.bep028 import (
    LABELS_NAME,
    PRIVATE_DIR,
    LabelMap,
    ProfileViolation,
    check_profile,
    export,
    file_entity,
)
from senselab_core.runtime.fetch import DataManifest
from senselab_core.runtime.provenance import PROV_DIR, ProvStore, RunIOError
from senselab_core.runtime.store import STORE_DIR, ResultStore
from senselab_core.runtime.vocabulary import core_vocabulary
from senselab_worker.canonical import jcs_loads

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
    (run / PRIVATE_DIR / LABELS_NAME).write_text('{"format": "labels@2.0", "labels": {}}')
    with pytest.raises(ValueError, match="labels@1.0") as caught:
        LabelMap(run).paths()
    assert str(run) not in str(caught.value)


def test_export_conforms_to_bep028(tmp_path: Path) -> None:
    """An export is the run's documents byte for byte, and its union meets the profile."""
    run, store = _run(tmp_path, "sub-01/a.wav", "sub-01/b.wav")
    copies = export(run, tmp_path / "out")
    assert [p.read_bytes() for p in copies] == [(run / PROV_DIR / p.name).read_bytes() for p in copies]
    assert check_profile(ProvStore.read(tmp_path / "out").document()) == []
    assert [r.id for r in ProvStore.read(tmp_path / "out").records()] == [r.id for r in store.records()]
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
        (lambda d: d["@graph"].append(copy.deepcopy(d["@graph"][0])), "only node"),
        (lambda d: d["@graph"][0].update({"@id": "a/b/c.wav"}), "address grammar"),
        (lambda d: d["@graph"].append({"@id": "agent:" + "9" * 64, "@type": ["senselab:File"]}), "prov:Entity"),
        (lambda d: d["@graph"].pop(1), "not a node of the document"),
        (
            lambda d: next(n for n in d["@graph"] if n["@id"].startswith("activity:")).pop("prov:startedAtTime"),
            "never starts",
        ),
        (
            lambda d: next(n for n in d["@graph"] if n["@id"].startswith("activity:")).update(
                {"prov:endedAtTime": {"@value": "2026-11-01T11:00:00Z", "@type": "xsd:dateTime"}}
            ),
            "before it starts",
        ),
        (lambda d: d["@graph"][-1].update({"senselab:fileLabel": "sub-01"}), "not file-<n>"),
    ],
)
def test_profile_violations_are_named(tmp_path: Path, edit: Any, mentions: str) -> None:  # noqa: ANN401
    """Each way a document leaves the profile is one named violation, and none echoes a value."""
    run, _ = _run(tmp_path, "sub-01/a.wav")
    doc = _doc(run)
    assert check_profile(doc) == []
    edit(doc)
    problems = check_profile(doc)
    assert any(mentions in p for p in problems), problems
    assert not any("a/b/c.wav" in p or "sub-01" in p for p in problems)


def test_one_label_names_one_node(tmp_path: Path) -> None:
    """Two nodes carrying one label are refused."""
    run, _ = _run(tmp_path, "a.wav", "b.wav")
    doc = _doc(run)
    for node in doc["@graph"]:
        if "senselab:fileLabel" in node:
            node["senselab:fileLabel"] = "file-1"
    assert any("another node has" in p for p in check_profile(doc))


def test_export_refuses_a_recorded_path_and_copies_nothing(tmp_path: Path) -> None:
    """A document whose string holds a path of the label map is not exported."""
    run, _ = _run(tmp_path, "sub-01/a.wav", key={"audio": "read from sub-01/a.wav"})
    with pytest.raises(ProfileViolation, match="label map"):
        export(run, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_file_entity_label_is_file_n() -> None:
    """A label is `file-<n>`; a path is not one."""
    assert prov.literal_value(file_entity("content:" + "1" * 64, "file-3").node()["senselab:fileLabel"]) == "file-3"
    with pytest.raises(ValueError, match="file-<n>"):
        file_entity("content:" + "1" * 64, "sub-01/a.wav")


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(root=COMPONENT, session=COMPONENT, dataset=COMPONENT, name=COMPONENT, out=COMPONENT)
def test_document_holds_no_path(root: str, session: str, dataset: str, name: str, out: str) -> None:
    """Random path components never occur in a written document, an export, an error or a log line."""
    parts = {root, session, dataset, name, out}
    handler = _Capture()
    logging.getLogger("senselab").addHandler(handler)
    try:
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch) / root
            run, _ = _run(base / session, f"{dataset}/{name}.wav")
            dest = Path(scratch) / "export" / out
            export(run, dest)
            for data in _bytes_under(run / PROV_DIR) + _bytes_under(dest):
                assert not any(p.encode() in data for p in parts)
            assert any(dataset.encode() in data for data in _bytes_under(run / PRIVATE_DIR))
            failures = _failures(base, run)
            quarantine = ResultStore(base / "store-root")
            entry = base / "store-root" / STORE_DIR / "ab" / ("ab" * 32) / ("0" * 64)
            entry.mkdir(parents=True)
            (entry / "result.json").write_text("{")
            quarantine._quarantine(entry, digest="ab" * 32, fingerprint="0" * 64)
    finally:
        logging.getLogger("senselab").removeHandler(handler)
    assert handler.messages
    for text in [*failures, *handler.messages]:
        assert not any(p in text for p in parts), text


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _failures(base: Path, run: Path) -> list[str]:
    """Return the text of each error a writer or the label map raises over paths holding the run's components."""
    broken = base / "broken"
    (broken / PROV_DIR).parent.mkdir(parents=True)
    (broken / PROV_DIR).write_text("not a directory")
    (broken / PRIVATE_DIR).write_text("not a directory")
    texts = []
    for action in (
        lambda: ProvStore([prov.entity(OUTPUT, VOCAB)]).write(broken),
        lambda: LabelMap(broken).assign([("content:" + "1" * 64, "x.wav")]),
        lambda: LabelMap(broken).paths(),
    ):
        with pytest.raises(RunIOError) as caught:
            action()
        err = caught.value
        assert err.filename is None and err.__cause__ is None and err.__suppress_context__
        texts += [str(err), repr(err), err.strerror or ""]
    assert os.fspath(run) not in "".join(texts)
    return texts
