"""The free-speech review page's extractor and renderer.

Every transcript in this file is invented. No fixture here carries corpus speech.
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[5] / "scripts" / "free_speech_review_page.py"
_SPEC = importlib.util.spec_from_file_location("free_speech_review_page", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
page = importlib.util.module_from_spec(_SPEC)
sys.modules["free_speech_review_page"] = page
_SPEC.loader.exec_module(page)


def _entity(
    entity_id: str, prov_type: str, attributes: dict[str, Any], extent: list[float] | None = None
) -> dict[str, Any]:
    """One entity record.

    Args:
        entity_id: The entity's id.
        prov_type: Its PROV type.
        attributes: Its attributes.
        extent: Its extent, or None.

    Returns:
        The JSONL record.
    """
    return {"record": "entity", "id": entity_id, "prov_type": prov_type, "extent": extent, "attributes": attributes}


def _word(index: int, text: str, *, bracketed: bool = False) -> dict[str, Any]:
    """One consensus word entity.

    Args:
        index: Its position in the consensus stream.
        text: Its surface.
        bracketed: Whether it is a transcription-convention token.

    Returns:
        The JSONL record.
    """
    return _entity(
        f"word-{index}",
        "word",
        {"text": text, "bracketed": bracketed, "index": index, "timings": {"asr": [index, index + 1]}},
        [index, index + 1],
    )


def _store(tmp_path: Path, stem: str, records: list[dict[str, Any]]) -> Path:
    """Write a run directory holding one store.

    Args:
        tmp_path: The temporary root.
        stem: The BIDS stem, without a timestamp.
        records: The store's records.

    Returns:
        The run directory.
    """
    participant, session = stem.split("_")[0], stem.split("_")[1]
    run_root = tmp_path / participant / session / f"{stem}_20260101-000000"
    (run_root / "run").mkdir(parents=True)
    (run_root / "run" / "store.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return run_root


_FAMILY_STEM = "sub-aaa_ses-bbb_task-free-speech-1"


def _verdict(release: str, ground: str | None = None, family: str = "free-speech") -> dict[str, Any]:
    """The file-level verdict entity.

    Args:
        release: The release value.
        ground: The release ground, or None.
        family: The declared family.

    Returns:
        The JSONL record.
    """
    return _entity(
        "verdict-file",
        "verdict",
        {
            "node": "VERDICT",
            "outcome": "pass",
            "triage": "pass",
            "release": release,
            "release_ground": ground,
            "declared_family": family,
            "why": "the fold completed",
        },
    )


def test_free_response_families_are_the_graphs_own() -> None:
    """The family list comes from EXPECTATIONS, and excludes the neighbouring item-list tasks."""
    families = page.free_response_families()
    assert "free-speech" in families
    assert "productive-vocabulary" in families
    assert "animal-fluency" not in families
    assert "random-item-generation" not in families
    assert "rainbow-passage" not in families
    assert len(families) == 10


def test_read_store_light_drops_measurement_payloads_but_keeps_the_scan(tmp_path: Path) -> None:
    """The heavy derivative measurements are skipped; the pii_scan measurement survives."""
    run_root = _store(
        tmp_path,
        _FAMILY_STEM,
        [
            _entity("measurement-big", "measurement", {"name": "gammatone", "values": [0.0] * 32}),
            _entity("measurement-scan", "measurement", {"name": "pii_scan", "scanned": True}),
            _word(0, "hello"),
        ],
    )
    view = page.read_store_light(run_root / "run" / "store.jsonl")
    assert [entity.id for entity in view.live("measurement")] == ["measurement-scan"]
    assert page.scan_state(view) is True


def test_scan_state_reads_a_declined_scan(tmp_path: Path) -> None:
    """A scan the graph declined reads as False, not as an absence."""
    run_root = _store(
        tmp_path,
        _FAMILY_STEM,
        [_entity("measurement-scan", "measurement", {"name": "pii_scan", "scanned": False, "why": "declined"})],
    )
    assert page.scan_state(page.read_store_light(run_root / "run" / "store.jsonl")) is False


def test_scan_state_is_none_without_a_measurement(tmp_path: Path) -> None:
    """No pii_scan measurement is unknown, which is not the same as a scan that found nothing."""
    run_root = _store(tmp_path, _FAMILY_STEM, [_word(0, "hello")])
    assert page.scan_state(page.read_store_light(run_root / "run" / "store.jsonl")) is None


def test_marked_words_follows_the_live_label_assertions(tmp_path: Path) -> None:
    """A retired assertion does not mark a word; the live one does."""
    records = [
        _word(0, "call"),
        _word(1, "Tuesday"),
        _entity("assertion-live", "assertion", {"verb": "label", "label": "pii", "category": "DATE_TIME"}),
        _entity("assertion-dead", "assertion", {"verb": "label", "label": "pii", "category": "PERSON"}),
        {"record": "relation", "relation": "wasDerivedFrom", "source": "assertion-live", "target": "word-1"},
        {"record": "relation", "relation": "wasDerivedFrom", "source": "assertion-dead", "target": "word-0"},
        {"record": "relation", "relation": "wasInvalidatedBy", "source": "assertion-dead", "target": "act-replay"},
    ]
    view = page.read_store_light(_store(tmp_path, _FAMILY_STEM, records) / "run" / "store.jsonl")
    assert page.marked_words(view) == {"word-1": ["DATE_TIME"]}


def _span(role: str, start: float, end: float, family: str = "speech") -> dict[str, Any]:
    """A branch span entity.

    Args:
        role: The span's role.
        start: Its start.
        end: Its end.
        family: The branch family that minted it.

    Returns:
        The JSONL record.
    """
    return _entity(f"span-{role}", "span", {"role": role, "family": family}, [start, end])


def _pii(
    entity_id: str, category: str, source: str, start: float, end: float, *, in_stimulus: bool = False
) -> dict[str, Any]:
    """A live pii finding.

    Args:
        entity_id: The entity's id.
        category: Its category.
        source: The detector that produced it.
        start: Its extent start.
        end: Its extent end.
        in_stimulus: Whether the stimulus accounts for it.

    Returns:
        The JSONL record.
    """
    return _entity(
        entity_id,
        "pii",
        {"category": category, "source": source, "haystack": "consensus", "in_stimulus": in_stimulus},
        [start, end],
    )


def _label(entity_id: str, category: str, word_id: str) -> list[dict[str, Any]]:
    """A label assertion and the edge tying it to its word.

    Args:
        entity_id: The assertion's id.
        category: The category it places.
        word_id: The word it marks.

    Returns:
        The two JSONL records.
    """
    return [
        _entity(entity_id, "assertion", {"verb": "label", "label": "pii", "category": category}),
        {"record": "relation", "relation": "wasDerivedFrom", "source": entity_id, "target": word_id},
    ]


def test_recording_record_reads_the_live_generation(tmp_path: Path) -> None:
    """A store carrying a retired verdict reports the surviving one."""
    records = [
        _word(0, "I"),
        _word(1, "[UH]", bracketed=True),
        _word(2, "moved"),
        _entity(
            "verdict-old", "verdict", {"node": "VERDICT", "release": "releasable", "declared_family": "free-speech"}
        ),
        {"record": "relation", "relation": "wasInvalidatedBy", "source": "verdict-old", "target": "act-replay"},
        _verdict("withheld"),
    ]
    row = page.recording_record(_store(tmp_path, _FAMILY_STEM, records), page.free_response_families())
    assert row is not None
    assert row["rel"] == "withheld"
    assert row["p"] == "sub-aaa"
    assert row["fam"] == "free-speech"
    assert row["stem"] == _FAMILY_STEM
    assert row["nw"] == 3
    assert row["nl"] == 2
    assert row["w"][1] == ["[UH]", 1, -1]
    assert row["f"] == []


def test_recording_record_skips_a_family_outside_the_pattern(tmp_path: Path) -> None:
    """An item-list task under the same tree is not a free response."""
    run_root = _store(
        tmp_path, "sub-aaa_ses-bbb_task-animal-fluency", [_verdict("releasable", family="animal-fluency")]
    )
    assert page.recording_record(run_root, page.free_response_families()) is None


def test_recording_record_builds_one_mark_with_its_detector(tmp_path: Path) -> None:
    """A marked word becomes a mark carrying the detector of the finding that covers it."""
    records = [
        _word(0, "Tuesday"),
        _pii("pii-1", "DATE_TIME", "presidio", 0, 1),
        *_label("assertion-1", "DATE_TIME", "word-0"),
        _verdict("withheld"),
    ]
    row = page.recording_record(_store(tmp_path, _FAMILY_STEM, records), page.free_response_families())
    assert row is not None
    assert row["pii"] == [{"c": "DATE_TIME", "s": "presidio", "h": "consensus", "stim": False}]
    assert row["w"][0] == ["Tuesday", 0, 0]
    mark = row["f"][0]
    assert mark["c"] == ["DATE_TIME"]
    assert mark["d"] == ["presidio"]
    assert mark["dn"] == 1
    assert mark["nt"] == 1
    assert mark["brk"] == 0
    assert mark["tx"] == -1


def test_a_mark_prefers_the_tightest_finding_that_contains_it(tmp_path: Path) -> None:
    """Two findings of one category over nested ranges: the narrower one is attributed first."""
    records = [
        _word(0, "alpha"),
        _pii("pii-wide", "PERSON", "gliner/name", 0, 9),
        _pii("pii-tight", "PERSON", "presidio", 0, 1),
        *_label("assertion-1", "PERSON", "word-0"),
        _verdict("withheld"),
    ]
    row = page.recording_record(_store(tmp_path, _FAMILY_STEM, records), page.free_response_families())
    assert row is not None
    assert row["f"][0]["d"] == ["presidio", "gliner/name"]
    assert row["f"][0]["dn"] == 2


def test_a_mark_records_a_bracketed_token_it_covers(tmp_path: Path) -> None:
    """The bracket flag is what makes the convention-token defect countable."""
    records = [
        _word(0, "[UH]", bracketed=True),
        _pii("pii-1", "PERSON", "gliner/name", 0, 1),
        *_label("assertion-1", "PERSON", "word-0"),
        _verdict("withheld"),
    ]
    row = page.recording_record(_store(tmp_path, _FAMILY_STEM, records), page.free_response_families())
    assert row is not None
    assert row["f"][0]["brk"] == 1


def test_task_extent_comes_from_the_branch_span_not_the_duration(tmp_path: Path) -> None:
    """The extent is the SPEECH task_extent span; a mark inside it reads as inside."""
    records = [
        _word(0, "inside"),
        _word(1, "outside"),
        _span("task_extent", 0.0, 1.0),
        _entity("measurement-dur", "measurement", {"name": "response_duration_s", "value": 9.0}),
        _pii("pii-1", "PERSON", "presidio", 0, 1),
        _pii("pii-2", "LOCATION", "presidio", 1, 2),
        *_label("assertion-1", "PERSON", "word-0"),
        *_label("assertion-2", "LOCATION", "word-1"),
        _verdict("withheld"),
    ]
    row = page.recording_record(_store(tmp_path, _FAMILY_STEM, records), page.free_response_families())
    assert row is not None
    assert [mark["tx"] for mark in row["f"]] == [1, 0]


def test_task_extent_absent_reads_as_unknown_not_outside(tmp_path: Path) -> None:
    """No task_extent span is the branch's record that it found no task."""
    records = [
        _word(0, "word"),
        _pii("pii-1", "PERSON", "presidio", 0, 1),
        *_label("assertion-1", "PERSON", "word-0"),
        _verdict("withheld"),
    ]
    row = page.recording_record(_store(tmp_path, _FAMILY_STEM, records), page.free_response_families())
    assert row is not None
    assert row["f"][0]["tx"] == -1


def test_finding_key_is_stable_and_position_independent() -> None:
    """The review key is content-addressed, so re-extraction and re-filtering keep a judgment."""
    first = page.finding_key("sub-a_ses-b_task-free-speech-1", ["PERSON"], 3, 5)
    again = page.finding_key("sub-a_ses-b_task-free-speech-1", ["PERSON"], 3, 5)
    other = page.finding_key("sub-a_ses-b_task-free-speech-1", ["PERSON"], 4, 5)
    elsewhere = page.finding_key("sub-c_ses-b_task-free-speech-1", ["PERSON"], 3, 5)
    assert first == again
    assert first != other
    assert first != elsewhere


def test_extract_writes_one_line_per_recording(tmp_path: Path) -> None:
    """The sweep keeps the free-response recordings and reports its counts."""
    _store(tmp_path, _FAMILY_STEM, [_word(0, "one"), _verdict("releasable")])
    _store(
        tmp_path,
        "sub-ccc_ses-ddd_task-cinderella-story",
        [_word(0, "two"), _verdict("nothing_to_redact", "x", "cinderella-story")],
    )
    _store(tmp_path, "sub-eee_ses-fff_task-animal-fluency", [_verdict("releasable", family="animal-fluency")])
    report = page.extract(tmp_path, tmp_path / "out.jsonl", workers=1)
    assert report["candidates"] == 2
    assert report["participants"] == 2
    assert report["counts"]["recordings"] == 2
    assert report["counts"]["family:cinderella-story"] == 1


def _mark(
    key: str,
    categories: list[str],
    detectors: list[str],
    start: int,
    end: int,
    **rest: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """A mark record in page shape.

    Args:
        key: Its review key.
        categories: Its categories.
        detectors: Its attributed detectors.
        start: Its first word index.
        end: One past its last.
        **rest: Overrides for the facet fields.

    Returns:
        The mark.
    """
    mark = {
        "k": key,
        "c": categories,
        "d": detectors,
        "dn": len(detectors),
        "i": [start, end],
        "nt": end - start,
        "nc": 4,
        "brk": 0,
        "stim": 0,
        "tx": -1,
    }
    mark.update(rest)
    return mark


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        ([["plain", 0, -1]], "plain"),
        ([["[UH]", 1, -1]], '<span class="bracket">[UH]</span>'),
        ([["a<b", 0, -1]], "a&lt;b"),
    ],
)
def test_paragraph_escapes_and_marks_brackets(words: list[list[Any]], expected: str) -> None:
    """A bracketed token renders as its own class; every surface is escaped."""
    assert page.paragraph(words, []) == expected


def test_paragraph_renders_one_mark_per_run() -> None:
    """Two adjacent words under one mark are one element, labelled once."""
    marks = [_mark("k1", ["PERSON"], ["gliner/name"], 0, 2)]
    rendered = page.paragraph([["Ada", 0, 0], ["Byron", 0, 0], ["spoke", 0, -1]], marks)
    assert rendered.count("<mark") == 1
    assert rendered.count('class="cat"') == 1
    assert "Ada Byron" in rendered
    assert rendered.endswith("spoke")


def test_paragraph_carries_the_facets_onto_the_mark() -> None:
    """Every facet the reviewer can filter on is on the element."""
    marks = [_mark("k1", ["PERSON"], ["gliner/name", "presidio"], 0, 1, brk=1, tx=0, stim=1)]
    rendered = page.paragraph([["[UH]", 1, 0]], marks)
    assert 'data-k="k1"' in rendered
    assert 'data-c="PERSON"' in rendered
    assert 'data-d="gliner/name presidio"' in rendered
    assert 'data-brk="1"' in rendered
    assert 'data-tx="0"' in rendered
    assert 'data-stim="1"' in rendered
    assert 'data-nt="1"' in rendered


def test_paragraph_labels_a_mark_carrying_two_categories() -> None:
    """Two categories on one mark render as one joined label."""
    assert ">PERSON+ORG<" in page.paragraph([["Acme", 0, 0]], [_mark("k1", ["PERSON", "ORG"], [], 0, 1)])


def test_paragraph_with_an_unattributed_mark_says_so() -> None:
    """A mark no finding could be joined to is labelled, not silently blank."""
    assert 'data-d="unattributed"' in page.paragraph([["word", 0, 0]], [_mark("k1", ["PERSON"], [], 0, 1)])


def test_the_review_vocabulary_separates_the_two_failure_modes() -> None:
    """The vocabulary keeps 'right category, nobody identified' apart from 'wrong category'."""
    names = [name for name, _, _ in page.VERDICTS]
    assert names == ["identifying", "not-identifying", "not-the-category", "unsure"]
    keys = [key for _, key, _ in page.VERDICTS]
    assert keys == ["1", "2", "3", "4"]


def _row(participant: str, **rest: Any) -> dict[str, Any]:  # noqa: ANN401 -- mixed-type page fields
    """One extract row in page shape.

    Args:
        participant: The participant id.
        **rest: Overrides for any field.

    Returns:
        The row.
    """
    row: dict[str, Any] = {
        "p": participant,
        "ses": "ses-b",
        "task": "free-speech-1",
        "stem": f"{participant}_ses-b_task-free-speech-1",
        "fam": "free-speech",
        "rel": "releasable",
        "rg": None,
        "tri": "pass",
        "why": "",
        "rwhy": "",
        "scan": True,
        "w": [["word", 0, -1]],
        "f": [],
        "pii": [],
        "nw": 1,
        "nl": 1,
        "ch": 4,
    }
    row.update(rest)
    return row


def test_render_is_self_contained_and_groups_by_participant(tmp_path: Path) -> None:
    """The page references nothing external and carries one section per participant."""
    data = tmp_path / "rows.jsonl"
    data.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                _row(
                    "sub-aaa",
                    rel="withheld",
                    why="w",
                    rwhy="r",
                    w=[["one", 0, 0], ["[UH]", 1, -1]],
                    f=[_mark("k1", ["PERSON"], ["gliner/name"], 0, 1)],
                    pii=[{"c": "PERSON", "s": "gliner/name", "h": "consensus", "stim": False}],
                    nw=2,
                    ch=7,
                ),
                _row(
                    "sub-ccc",
                    task="cinderella-story",
                    fam="cinderella-story",
                    rel="nothing_to_redact",
                    rg="the scan ran over the transcript and found nothing to redact",
                    w=[["two", 0, -1]],
                    ch=3,
                ),
            )
        )
        + "\n"
    )
    corpus = page.load(data)
    document = page.render(corpus, "Review")
    assert corpus.recordings == 2
    assert corpus.marks == 1
    assert 'id="sub-aaa"' in document and 'id="sub-ccc"' in document
    assert "http://" not in document and "https://" not in document
    assert "<img" not in document and "<link" not in document
    assert "src=" not in document
    assert "PERSON" in document
    assert "the scan ran over the transcript and found nothing to redact" in document


def test_render_offers_a_facet_for_every_category_and_detector(tmp_path: Path) -> None:
    """The facet lists are built from what the corpus actually holds."""
    corpus = page.Corpus()
    corpus.add(_row("sub-a", f=[_mark("k1", ["PERSON"], ["gliner/name"], 0, 1)]))
    corpus.add(_row("sub-b", f=[_mark("k2", ["DATE_TIME"], ["presidio"], 0, 1)]))
    corpus.add(_row("sub-c", f=[_mark("k3", ["MISC"], [], 0, 1)]))
    document = page.render(corpus, "Review")
    assert corpus.detectors["unattributed"] == 1
    for value in ('class="cat-f" value="PERSON"', 'class="cat-f" value="DATE_TIME"'):
        assert value in document
    for value in ('class="det-f" value="gliner/name"', 'class="det-f" value="presidio"'):
        assert value in document
    assert 'class="det-f" value="unattributed"' in document
    for control in ('id="brk"', 'id="tx"', 'id="minnt"', 'id="maxnt"', 'id="minnf"', 'id="rev"'):
        assert control in document


def test_render_carries_the_review_layer(tmp_path: Path) -> None:
    """The page ships the verdict buttons, the per-recording note and the export controls."""
    corpus = page.Corpus()
    corpus.add(_row("sub-a", f=[_mark("k1", ["PERSON"], ["presidio"], 0, 1)], w=[["word", 0, 0]]))
    document = page.render(corpus, "Review")
    for name, key, _ in page.VERDICTS:
        assert f'data-v="{name}"' in document
        assert f"<kbd>{key}</kbd>" in document
    assert 'class="rnote"' in document
    assert 'id="export"' in document and 'id="import"' in document and 'id="file"' in document
    assert "localStorage" in document
    assert 'data-stem="sub-a_ses-b_task-free-speech-1"' in document


def test_write_pages_shards_by_participant_with_an_index(tmp_path: Path) -> None:
    """Sharding splits participants into files and writes an index over them, all mode 600."""
    corpus = page.Corpus()
    for number in range(5):
        corpus.add(_row(f"sub-{number}"))
    written = page.write_pages(corpus, tmp_path / "pages", "Review", shard_size=2)
    shards = list(written["shards"])
    assert [item["participants"] for item in shards] == [2, 2, 1]
    index = tmp_path / "pages" / "index.html"
    assert index.exists()
    assert index.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "pages" / "shard-001.html").stat().st_mode & 0o777 == 0o600


def test_write_pages_single_file_is_private(tmp_path: Path) -> None:
    """The unsharded page is written mode 600."""
    corpus = page.Corpus()
    corpus.add(_row("sub-a"))
    out = tmp_path / "page.html"
    page.write_pages(corpus, out, "Review", shard_size=0)
    assert out.stat().st_mode & 0o777 == 0o600
