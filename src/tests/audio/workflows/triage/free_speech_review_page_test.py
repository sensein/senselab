"""The free-speech review page's extractor and renderer.

Every transcript in this file is invented. No fixture here carries corpus speech.
"""

import importlib.util
import json
import re
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
            _entity("measurement-scan", "measurement", {"name": "pii_scan", "scanned_by": ["gliner"]}),
            _word(0, "hello"),
        ],
    )
    view = page.read_store_light(run_root / "run" / "store.jsonl")
    assert [entity.id for entity in view.live("measurement")] == ["measurement-scan"]
    assert page.scan_state(page.pii_scan_of(view)) is True


def test_scan_state_reads_a_declined_scan(tmp_path: Path) -> None:
    """A record in which no detector ran reads as False, not as an absence or as a scan."""
    run_root = _store(
        tmp_path,
        _FAMILY_STEM,
        [_entity("measurement-scan", "measurement", {"name": "pii_scan", "scanned_by": [], "residue_content": False})],
    )
    view = page.read_store_light(run_root / "run" / "store.jsonl")
    assert page.scan_state(page.pii_scan_of(view)) is False


def test_scan_state_is_none_without_a_measurement(tmp_path: Path) -> None:
    """No pii_scan measurement is unknown, which is not the same as a scan that found nothing."""
    run_root = _store(tmp_path, _FAMILY_STEM, [_word(0, "hello")])
    view = page.read_store_light(run_root / "run" / "store.jsonl")
    assert page.scan_state(page.pii_scan_of(view)) is None
    assert page.residue_of(page.pii_scan_of(view), []) is None


def test_residue_names_the_words_the_detectors_read(tmp_path: Path) -> None:
    """The residue positions come from the scan's word ids, in transcript order."""
    scan = {
        "name": "pii_scan",
        "scanned_by": ["gliner", "presidio"],
        "residue_method": "stimulus_alignment",
        "residue_words_n": 2,
        "residue_content": True,
        "residue_word_ids": ["word-2", "word-0"],
    }
    run_root = _store(
        tmp_path,
        _FAMILY_STEM,
        [_entity("measurement-scan", "measurement", scan), _word(0, "Alice"), _word(1, "the"), _word(2, "Smith")],
    )
    view = page.read_store_light(run_root / "run" / "store.jsonl")
    words = sorted(view.live("word"), key=lambda entity: int(entity.attributes.get("index", 0)))
    residue = page.residue_of(page.pii_scan_of(view), words)
    assert residue == {"n": 2, "m": "stimulus_alignment", "c": True, "i": [0, 2]}


def test_the_residue_and_a_declined_scan_are_in_the_popup_not_on_the_card() -> None:
    """Owner, 2026-09-27: no explanation on the text; the popup carries the scan and the residue."""
    row = {
        "p": "p",
        "ses": "s",
        "task": "free-speech-1",
        "stem": "p_s_task-free-speech-1",
        "fam": "free-speech",
        "rel": "release_without_redaction",
        "rg": None,
        "why": "",
        "rwhy": "",
        "scan": False,
        "res": {"n": 1, "m": "stimulus_alignment", "c": False, "i": [1]},
        "w": [["Hello", 0, -1], ["the", 0, -1]],
        "f": [],
        "nl": 2,
        "nw": 2,
    }
    card = page.recording_html(row)
    assert "scan" not in card.split('<p class="text">')[0] and "residue" not in card
    account = page.card_account(row)
    assert account["scan"] is False
    assert account["res"] == {"n": 1, "m": "stimulus_alignment", "c": False, "said": "the"}
    assert "the scan was declined" in page._SCRIPT and "function words only" in page._SCRIPT


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
        {
            "category": category,
            "source": source,
            "haystack": "consensus",
            "in_stimulus": in_stimulus,
            "word_ids": [f"word-{i}" for i in range(int(start), max(int(start) + 1, int(end)))],
        },
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
            "verdict-old",
            "verdict",
            {"node": "VERDICT", "release": "release_with_redaction", "declared_family": "free-speech"},
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
        tmp_path, "sub-aaa_ses-bbb_task-animal-fluency", [_verdict("release_with_redaction", family="animal-fluency")]
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
    """Two findings of one category naming the mark's word: the one naming fewer words comes first."""
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
    _store(tmp_path, _FAMILY_STEM, [_word(0, "one"), _verdict("release_with_redaction")])
    _store(
        tmp_path,
        "sub-ccc_ses-ddd_task-cinderella-story",
        [_word(0, "two"), _verdict("release_without_redaction", "x", "cinderella-story")],
    )
    _store(
        tmp_path, "sub-eee_ses-fff_task-animal-fluency", [_verdict("release_with_redaction", family="animal-fluency")]
    )
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


def test_paragraph_renders_a_second_category_as_its_own_nested_mark() -> None:
    """Owner, 2026-09-28: a mark carries one category; another category on its words is its own mark."""
    overlay = {"k": "k2", "c": "ORG", "s": "detected", "hr": 0, "kd": "", "dx": "", "d": []}
    rendered = page.paragraph([["Acme", 0, 0]], [_mark("k1", ["PERSON"], [], 0, 1, o=[overlay])])
    assert rendered.count("<mark") == 2
    assert re.findall(r'data-c="([^"]*)"', rendered) == ["PERSON", "ORG"]
    assert re.findall(r'<span class="cat">([^<]*)</span>', rendered) == ["PERSON", "ORG"]
    assert "+" not in rendered


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
        "rel": "release_with_redaction",
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
                    rel="release_without_redaction",
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


def test_the_category_facet_matches_each_marks_single_category() -> None:
    """Every mark carries one category, so the facet matches ``data-c`` whole; a nested mark matches its own."""
    assert "if(!cats.has(m.dataset.c))return false;" in page._SCRIPT
    assert "split('+')" not in page._SCRIPT
    assert "e.stopPropagation();open(m);" in page._SCRIPT, "a click on a nested mark opens that mark only"


def test_a_recording_note_refreshes_the_progress_line() -> None:
    """The note handler has to re-tally, or the count of notes never moves."""
    assert "save();tally();});" in page._SCRIPT


def test_the_store_reads_and_writes_are_guarded() -> None:
    """Storage throws outright in some contexts, so the page must render without it."""
    assert page._SCRIPT.count("try{") >= 2
    assert "catch(e){store={findings:{},recordings:{},release:{},flags:{}};}" in page._SCRIPT


def test_in_stimulus_keeps_none_apart_from_false() -> None:
    """A task with no stimulus text yields None, which is not the same claim as 'not in it'."""
    assert page.tristate(True) == 1
    assert page.tristate(False) == 0
    assert page.tristate(None) == -1


def test_a_finding_on_a_task_with_no_stimulus_reports_the_unchecked_state(tmp_path: Path) -> None:
    """The third artefact family is only visible if None survives the extract."""
    records = [
        _word(0, "Sonnenschein"),
        _entity(
            "pii-1",
            "pii",
            {"category": "NAME", "source": "rules/ner", "haystack": "consensus", "in_stimulus": None},
            [0, 1],
        ),
        *_label("assertion-1", "NAME", "word-0"),
        _verdict("withheld", family="cinderella-story"),
    ]
    run_root = _store(tmp_path, "sub-aaa_ses-bbb_task-cinderella-story", records)
    row = page.recording_record(run_root, page.free_response_families())
    assert row is not None
    assert row["pii"][0]["stim"] == -1


def _determined_store(tmp_path: Path) -> Path:
    """A store carrying a passing gate, a declared-but-unevaluated gate and a failing REDACT.

    Args:
        tmp_path: The temporary root.

    Returns:
        The run directory.
    """
    verdict = _entity(
        "verdict-file",
        "verdict",
        {
            "node": "VERDICT",
            "outcome": "pass",
            "triage": "pass",
            "release": "withheld",
            "release_ground": None,
            "declared_family": "cinderella-story",
            "why": "folded the node verdicts",
            "llm_redaction": {
                "status": "nothing_to_read",
                "iterations": 0,
                "model_id": "",
                "flagged": [],
                "failure": "none",
            },
            "ran": {"REDACT": "completed", "SPEECH": "completed", "VOICE": "skipped"},
            "critical_absences": {},
            "reasons": [
                {"node": "SPEECH", "outcome": "pass", "why": "spoke"},
                {"node": "REDACT", "outcome": "fail", "why": "verification found pii on the redacted transcript"},
            ],
            "gates": {
                "applied": [
                    {
                        "gate": "response_min_s",
                        "reading": "response_duration_s",
                        "op": "at_least",
                        "bound": 0.5,
                        "value": 242.085,
                        "passed": True,
                        "group": "free_response",
                        "keyed_under": "FREE_RESPONSE",
                        "layer": "by_group",
                    }
                ],
                "flagging": [],
                "bounds": {"response_min_s": 0.5, "verbatim_overlap_max": 0.5, "echo_overlap_max": 0.5},
                "layers": {"response_min_s": "by_group"},
                "group": "free_response",
            },
        },
    )
    records = [
        _word(0, "word"),
        _entity("verdict-redact", "verdict", {"node": "REDACT", "outcome": "fail", "why": "pii survived"}),
        _entity(
            "measurement-ex",
            "measurement",
            {"name": "redaction_exemptions", "expected_speech_declared": False, "n": 0, "n_findings": 3},
        ),
        verdict,
    ]
    return _store(tmp_path, "sub-aaa_ses-bbb_task-cinderella-story", records)


def test_determination_records_who_decided_and_what_never_ran(tmp_path: Path) -> None:
    """The account distinguishes a failed gate, an unevaluated gate and a component that skipped."""
    row = page.recording_record(_determined_store(tmp_path), page.free_response_families())
    assert row is not None
    determination = row["d"]
    assert determination["redact"]["outcome"] == "fail"
    assert determination["redact"]["why"] == "pii survived"
    assert determination["llm"]["status"] == "nothing_to_read"
    assert determination["ran"]["VOICE"] == "skipped"
    assert [gate["gate"] for gate in determination["gates"]["applied"]] == ["response_min_s"]
    assert set(determination["gates"]["bounds"]) == {"response_min_s", "verbatim_overlap_max", "echo_overlap_max"}
    assert determination["exempt"] == {"declared": False, "n": 0, "n_findings": 3, "recorded": True}
    assert [node["node"] for node in determination["nodes"]] == ["SPEECH", "REDACT"]


def test_the_measurement_kept_is_the_exemption_one_too(tmp_path: Path) -> None:
    """read_store_light must not drop redaction_exemptions along with the derivative arrays."""
    run_root = _store(
        tmp_path,
        _FAMILY_STEM,
        [
            _entity("measurement-big", "measurement", {"name": "gammatone", "values": [0.0] * 8}),
            _entity("measurement-ex", "measurement", {"name": "redaction_exemptions", "n": 0}),
            _entity("measurement-scan", "measurement", {"name": "pii_scan", "scanned_by": ["gliner"]}),
        ],
    )
    view = page.read_store_light(run_root / "run" / "store.jsonl")
    assert sorted(str(e.attributes["name"]) for e in view.live("measurement")) == [
        "pii_scan",
        "redaction_exemptions",
    ]


def test_the_pool_deduplicates_what_repeats() -> None:
    """Nine node names over 11,701 recordings must cost nine entries, not a hundred thousand."""
    pool = page.ValuePool()
    assert pool.add("REDACT") == 0
    assert pool.add("REDACT") == 0
    assert pool.add({"a": 1}) == 1
    assert pool.add({"a": 1}) == 1
    assert pool.values == ["REDACT", {"a": 1}]


def test_pooled_determination_keeps_the_per_recording_gate_reading() -> None:
    """The gate specification pools; its measured value does not."""
    pool = page.ValuePool()
    determination = {
        "redact": {"outcome": "fail", "why": "pii survived"},
        "nodes": [{"node": "REDACT", "outcome": "fail", "why": "pii survived"}],
        "llm": {"status": "nothing_to_read"},
        "gates": {
            "applied": [
                {
                    "gate": "response_min_s",
                    "reading": "response_duration_s",
                    "op": "at_least",
                    "bound": 0.5,
                    "value": 242.085,
                    "passed": True,
                }
            ],
            "flagging": [],
            "bounds": {"response_min_s": 0.5, "echo_overlap_max": 0.5},
            "layers": {},
            "group": "free_response",
        },
        "ran": {"REDACT": "completed"},
        "absences": [],
        "exempt": {"declared": False, "n": 0, "n_findings": 1, "recorded": True},
        "findings": [{"c": "NAME", "s": "rules/ner", "h": "consensus", "stim": -1}],
    }
    record = page.pooled_determination(determination, pool)
    assert record["g"] == [[record["g"][0][0], 242.085, 1]]
    assert page.gate_state(True) == 1
    assert page.gate_state(False) == 0
    assert page.gate_state("UNDETERMINED") == -1
    spec = pool.values[record["g"][0][0]]
    assert spec["gate"] == "response_min_s"
    assert spec["kind"] == "applied"
    assert "value" not in spec
    assert record["f"][0][3] == -1


def test_the_page_keeps_the_three_non_readings_apart() -> None:
    """Switched off, nothing to review, and could-not-load are three different silences."""
    assert "Switched off." in page._SCRIPT
    assert "There was no text to read." in page._SCRIPT
    assert "It tried and could not load." in page._SCRIPT
    assert "It reached no conclusion about this recording." in page._SCRIPT


def test_the_page_names_an_unevaluated_gate_as_such() -> None:
    """A gate with a bound and no reading did not pass and did not fail."""
    assert "never evaluated" in page._SCRIPT
    assert "did not pass and did not fail" in page._SCRIPT


def test_the_page_names_a_task_with_no_stimulus_text() -> None:
    """The third artefact family needs saying, not inferring from a blank."""
    assert "no stimulus text" in page._SCRIPT
    assert "no stimulus to check" in page._SCRIPT


def test_render_carries_the_determination_payload() -> None:
    """The page ships the pooled account and a trigger on every card."""
    corpus = page.Corpus()
    corpus.add(
        _row(
            "sub-a",
            rel="withheld",
            d={
                "redact": {"outcome": "fail", "why": "pii survived"},
                "nodes": [{"node": "REDACT", "outcome": "fail", "why": "pii survived"}],
                "llm": {"status": "nothing_to_read"},
                "gates": {"applied": [], "flagging": [], "bounds": {}, "layers": {}, "group": ""},
                "ran": {"REDACT": "completed"},
                "absences": [],
                "exempt": {"declared": False, "n": 0, "n_findings": 0, "recorded": True},
            },
        )
    )
    document = page.render(corpus, "Review")
    assert 'id="whydata"' in document
    assert 'class="whybtn" data-stem="sub-a_ses-b_task-free-speech-1"' in document
    assert "what determined this status" in document
    assert '"pool"' in document and '"rows"' in document


def test_an_unanswerable_gate_is_not_a_passing_one() -> None:
    """``passed`` is the string UNDETERMINED, which is truthy; a bool coercion would pass it."""
    pool = page.ValuePool()
    determination = {
        "redact": {"outcome": "", "why": ""},
        "nodes": [],
        "llm": {},
        "gates": {
            "applied": [
                {
                    "gate": "dominant_speaker_share_min",
                    "reading": "extent_dominant_speaker_share",
                    "op": "at_least",
                    "bound": 0.9,
                    "value": None,
                    "passed": "UNDETERMINED",
                }
            ],
            "flagging": [],
            "bounds": {"dominant_speaker_share_min": 0.9},
            "layers": {},
            "group": "free_response",
        },
        "ran": {},
        "absences": [],
        "exempt": {},
        "findings": [],
    }
    record = page.pooled_determination(determination, pool)
    assert record["g"][0][2] == -1


def test_the_page_separates_every_reviewer_state() -> None:
    """Five states plus an unrecorded one, each with its own sentence."""
    assert "disabled in the " in page._SCRIPT
    assert "tried and could not load" in page._SCRIPT
    assert "It ran and flagged nothing" in page._SCRIPT
    assert "It flagged " in page._SCRIPT
    assert "No annotation was recorded." in page._SCRIPT
    assert page.LLM_STATES == ("disabled", "nothing_to_read", "absent", "clean", "flagged")
    assert page.LLM_RAN == ("absent", "clean", "flagged")


def test_nothing_to_read_means_an_empty_transcript_and_the_old_sentence_is_gone() -> None:
    """The reviewer no longer comes through the detectors, so its old sentence is false twice over."""
    assert "could not have run" not in page._SCRIPT
    assert "REDACT withholds before the reviewer is reached" not in page._SCRIPT
    assert "the detectors found nothing, so there was no redacted text to read back" not in page._SCRIPT
    assert "not_run" not in page.LLM_STATES
    assert "transcript carries no words, so there was nothing to read back" in page._SCRIPT


def test_a_clean_reading_beside_a_failure_is_named_a_disagreement() -> None:
    """detector_outcome is the only record of which decision a reading was taken beside."""
    assert "llm.detector_outcome" in page._SCRIPT
    assert "is a disagreement, not a " in page._SCRIPT


def test_no_reader_may_infer_a_review_happened() -> None:
    """Every non-reading says outright that nothing was concluded."""
    assert "no review was attempted" in page._SCRIPT
    assert "This is not a verdict about the recording." in page._SCRIPT


def test_the_page_says_a_ground_is_empty_by_construction() -> None:
    """A ground is null exactly when REDACT decided; a blank must not read as a missing reason."""
    assert "empty by construction" in page._SCRIPT
    assert "REDACT decided it" in page._SCRIPT


def test_the_flag_key_does_not_collide_with_the_finding_keys() -> None:
    """A review flag and a finding verdict must never be one keystroke apart."""
    finding = {key for _, key, _ in page.VERDICTS}
    assert page.FLAG_KEY not in finding


def test_a_card_carries_the_review_flag() -> None:
    """Every recording gets one flag toggle, independent of its release decision."""
    corpus = page.Corpus()
    corpus.add(_row("sub-a"))
    document = page.render(corpus, "Review")
    assert document.count('class="flagtoggle"') == 1
    assert f"<kbd>{page.FLAG_KEY}</kbd>" in document
    assert 'data-stem="sub-a_ses-b_task-free-speech-1"' in document


def test_the_flag_markup_does_not_repeat_the_stem() -> None:
    """One toggle on 11,701 cards: a repeated stem and title cost megabytes for nothing."""
    assert "data-stem" not in page._FLAG_TOGGLE
    assert "title=" not in page._FLAG_TOGGLE


def test_the_flag_is_a_separate_collection_from_the_decision_and_the_verdicts() -> None:
    """Three records in the export, distinguishable, none shadowing another."""
    assert "flags:store.flags" in page._SCRIPT
    assert "release:store.release" in page._SCRIPT
    assert "findings:store.findings" in page._SCRIPT
    assert "version:4" in page._SCRIPT
    assert "triage:" not in page._SCRIPT.split("function payload()")[1].split("}")[0]


def test_the_flag_persists_and_degrades_like_the_finding_verdicts() -> None:
    """Same namespace, same guarded access, the same collections restored on load."""
    assert "flags:parsed.flags||{}" in page._SCRIPT
    assert "catch(e){store={findings:{},recordings:{},release:{},flags:{}};}" in page._SCRIPT


def test_the_flag_toggles_and_never_touches_the_release_decision() -> None:
    """Pressing it again clears it; it is independent of which release the reviewer chose."""
    body = page._SCRIPT.split("function toggleFlag(card){")[1].split("\n}")[0]
    assert "if(store.flags[stem])delete store.flags[stem];" in body
    assert "store.release" not in body


def test_the_flag_reaches_the_filters_and_the_progress_line() -> None:
    """The reader can sweep what they flagged, or everything they did not."""
    corpus = page.Corpus()
    corpus.add(_row("sub-a"))
    document = page.render(corpus, "Review")
    assert 'id="flag"' in document
    for value in ("flagged", "unflagged"):
        assert f'<option value="{value}">' in document
    assert "flagged for review" in page._SCRIPT


def test_the_release_decision_is_the_axis_own_vocabulary() -> None:
    """A reviewer's decision joins to a verdict without a mapping table between them.

    ``specs/20260924-which-artefact-is-releasable/design.md`` §6.
    """
    assert [value for value, _, _, _ in page.RELEASE_DECISIONS] == [
        "withheld",
        "release_with_redaction",
        "release_without_redaction",
    ]
    assert set(value for value, _, _, _ in page.RELEASE_DECISIONS) <= set(page.RELEASE_ORDER)


def test_the_three_release_choices_are_mutually_exclusive() -> None:
    """One value per recording: choosing one replaces the other two, and choosing it again clears it."""
    assert "else store.release[stem]={v:value,t:new Date().toISOString()};" in page._SCRIPT
    assert "for(const b of card.querySelectorAll('.dec'))b.classList.toggle('on',b.dataset.v===value);" in page._SCRIPT


def test_the_release_decision_keys_collide_with_nothing_else_on_the_page() -> None:
    """Four key families now share one page; a keystroke must mean exactly one thing."""
    decision = {key for _, key, _, _ in page.RELEASE_DECISIONS}
    finding = {key for _, key, _ in page.VERDICTS}
    movement = {"j", "k", "J", "K"}
    assert page.FLAG_KEY not in decision
    assert not decision & finding
    assert not decision & movement


def test_a_card_carries_the_release_decision_controls() -> None:
    """A human reading the recording says which release it warrants, on the card itself."""
    corpus = page.Corpus()
    corpus.add(_row("sub-a"))
    document = page.render(corpus, "Review")
    assert document.count('class="decgroup"') == 1
    for value, key, slug, label in page.RELEASE_DECISIONS:
        assert f'class="dec d-{slug}" data-v="{value}"' in document
        assert f"{label}<kbd>{key}</kbd>" in document


def test_the_release_decision_markup_does_not_repeat_the_stem() -> None:
    """Same argument as the row controls: the enclosing card already names the row."""
    assert "data-stem" not in page._DECISION_GROUP


def test_the_release_decision_is_its_own_collection() -> None:
    """A flag says come back to this; a release decision says which artefact may be handed on."""
    assert "release:store.release" in page._SCRIPT
    assert "flags:store.flags" in page._SCRIPT
    assert "version:4" in page._SCRIPT


def test_the_release_decision_persists_and_degrades_like_the_others() -> None:
    """The same localStorage namespace, the same guarded access, four collections restored."""
    assert "release:parsed.release||{}" in page._SCRIPT
    assert "catch(e){store={findings:{},recordings:{},release:{},flags:{}};}" in page._SCRIPT


def test_re_pressing_a_release_decision_clears_it() -> None:
    """Un-deciding is how a reviewer withdraws one, matching every other control on the page."""
    assert "if(held===value)delete store.release[stem];" in page._SCRIPT


def test_the_release_decision_reaches_the_filters_and_the_progress_line() -> None:
    """A reviewer must be able to sweep what they have not yet decided."""
    corpus = page.Corpus()
    corpus.add(_row("sub-a"))
    document = page.render(corpus, "Review")
    assert 'id="dec"' in document
    for value in ("decided", "undecided", "withheld", "release_without_redaction", "release_with_redaction"):
        assert f'<option value="{value}">' in document
    assert "rows given a release decision" in page._SCRIPT


def test_the_release_decision_does_not_overwrite_the_graphs_own() -> None:
    """The card shows both: what the graph concluded, and what the reviewer says it warrants."""
    corpus = page.Corpus()
    corpus.add(_row("sub-a"))
    document = page.render(corpus, "Review")
    assert 'data-rel="' in document
    assert "card.dataset.dec" in page._SCRIPT
    assert "rels.has(r.dataset.rel)" in page._SCRIPT


def test_the_release_decision_keys_are_discoverable() -> None:
    """A control nobody can find is a control nobody uses."""
    corpus = page.Corpus()
    corpus.add(_row("sub-a"))
    document = page.render(corpus, "Review")
    for _, key, _, _ in page.RELEASE_DECISIONS:
        assert f"<kbd>{key}</kbd>" in document


def test_reset_clears_the_release_decision_filter_too() -> None:
    """A reset that leaves one facet set shows a narrowed page while claiming to show everything."""
    reset = page._SCRIPT.split("getElementById('all')")[1].split("});")[0]
    for control in ("firedSel", "brkSel", "txSel", "revSel", "flagSel", "decSel", "llmSel"):
        assert f"{control}.value='any'" in reset, control


def test_the_reviewer_controls_are_not_searchable_text() -> None:
    """A control labelled in the page's own vocabulary would match every card in the search box.

    A release decision reads ``without redaction``, which is exactly what a reviewer would type to
    find one.
    """
    assert "haystack.set(r,(r.textContent" not in page._SCRIPT
    assert "function haystackOf(card)" in page._SCRIPT


def test_every_progress_term_counts_the_cards_on_this_page() -> None:
    """One ``localStorage`` namespace spans every shard; ``cards`` is this shard only.

    Counting the store's keys against this page's cards can report more decisions than rows.
    """
    for collection in ("flags", "release", "recordings"):
        assert f"Object.keys(store.{collection}).length" not in page._SCRIPT
    assert "rows given a release decision" in page._SCRIPT


def test_the_decided_row_stripe_is_legible_in_both_themes() -> None:
    """A 3px stripe at the light theme's value is all but invisible on the dark card."""
    dark = page._STYLE.split("@media (prefers-color-scheme:dark)")[1]
    for value in ("release_without_redaction", "release_with_redaction", "withheld"):
        assert f'.rec[data-dec="{value}"]' in dark, value


def test_the_row_keys_are_inert_while_typing() -> None:
    """An f typed into a note must not flag the row."""
    assert "e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT'" in page._SCRIPT


def test_the_reviewer_facet_offers_every_state_and_a_did_it_run_shortcut() -> None:
    """The page must be ready to show real verdicts when a GPU pass lands."""
    corpus = page.Corpus()
    corpus.add(_row("sub-a"))
    document = page.render(corpus, "Review")
    assert 'id="llm"' in document
    for state in page.LLM_STATES:
        assert f'<option value="{state}">' in document
    assert '<option value="ran">' in document
    assert "st==='absent'||st==='clean'||st==='flagged'" in page._SCRIPT


def test_a_card_carries_the_reviewer_state_for_the_facet() -> None:
    """The facet filters recordings, so the state rides on the card."""
    corpus = page.Corpus()
    corpus.add(_row("sub-a", d={"llm": {"status": "clean"}}))
    corpus.add(_row("sub-b", d={"llm": {}}))
    document = page.render(corpus, "Review")
    assert 'data-llm="clean"' in document
    assert 'data-llm="unrecorded"' in document


def test_the_movement_keys_do_not_collide_with_either_mark_set() -> None:
    """Three key sets share one document; none may overlap another."""
    movement = {"j", "k", "J", "K"}
    finding = {key for _, key, _ in page.VERDICTS}
    assert page.FLAG_KEY not in movement
    assert not movement & finding
    assert page.FLAG_KEY not in finding


def test_the_page_binds_movement_and_leaves_the_arrows_alone() -> None:
    """The arrows are the page's scroll; taking them would cost more than it buys."""
    assert "e.key==='j'" in page._SCRIPT
    assert "e.key==='k'" in page._SCRIPT
    assert "e.key==='J'" in page._SCRIPT
    assert "e.key==='K'" in page._SCRIPT
    assert "ArrowDown" not in page._SCRIPT
    assert "ArrowUp" not in page._SCRIPT


def test_movement_walks_only_the_visible_cards() -> None:
    """At 11,701 cards, next means the next card the filters left standing."""
    assert "function visibleCards()" in page._SCRIPT
    assert "!c.classList.contains('hidden')" in page._SCRIPT
    assert "const live=visibleCards();" in page._SCRIPT


def test_the_navigated_card_becomes_the_mark_target() -> None:
    """Moving and judging without a mouse means movement sets what the mark keys act on."""
    assert "markActive(live[next],true)" in page._SCRIPT
    assert "const card=activeCard||" in page._SCRIPT


def test_the_pointer_and_the_keyboard_do_not_fight_over_the_target() -> None:
    """A scroll under a still mouse fires mouseenter; it must not steal a keyboard target."""
    assert "let pointerOwns=true;" in page._SCRIPT
    assert "if(pointerOwns)markActive(card,false);" in page._SCRIPT
    assert "document.addEventListener('mousemove'" in page._SCRIPT
    assert "pointerOwns=false;" in page._SCRIPT


def test_a_filter_that_hides_the_target_drops_it() -> None:
    """The active card must never be one the reader cannot see."""
    assert "activeCard.classList.remove('active');activeCard=null;here(null);" in page._SCRIPT


def test_the_keys_are_discoverable_in_the_page() -> None:
    """A reader should not need the spec to find them."""
    corpus = page.Corpus()
    corpus.add(_row("sub-a"))
    document = page.render(corpus, "Review")
    assert 'id="keys"' in document
    assert "<dt>j / k</dt>" in document
    assert "<dt>J / K</dt>" in document
    assert "next / previous sample" in document
    assert f"<dt>{page.FLAG_KEY}</dt>" in document
    assert "<dt>w / d / o</dt>" in document


def test_the_outline_extends_the_rail_rather_than_duplicating_it() -> None:
    """One participant list, carrying progress; a second would duplicate the filter logic."""
    corpus = page.Corpus()
    corpus.add(_row("sub-a"))
    corpus.add(_row("sub-b"))
    document = page.render(corpus, "Review")
    assert document.count('id="jump"') == 1
    assert 'id="outline"' in document
    assert 'id="outsum"' in document
    assert document.count('class="meter"') == 2
    assert document.count('class="mr"') == 2
    assert document.count('class="mj"') == 2


def test_the_outline_reports_progress_and_what_the_filters_left() -> None:
    """It doubles as progress, so it earns the width it takes."""
    assert "function outline()" in page._SCRIPT
    assert "rows decided, " in page._SCRIPT
    assert "findings judged, " in page._SCRIPT
    assert "' shown'" in page._SCRIPT
    assert ".jn').textContent=shown" in page._SCRIPT


def test_the_outline_shows_where_the_reader_is() -> None:
    """The entry for the participant under the cursor is marked and scrolled into the rail."""
    assert "function here(section)" in page._SCRIPT
    assert "now.classList.add('here')" in page._SCRIPT
    assert "block:'nearest'" in page._SCRIPT


def test_the_layout_is_intrinsic_before_it_is_broken_by_a_query() -> None:
    """Relative units and wrapping first; the queries only collapse the two-column shell."""
    assert "grid-template-columns:minmax(190px,230px) minmax(0,1fr)" in page._STYLE
    assert "max-width:min(" in page._STYLE or "width:min(" in page._STYLE
    assert page._STYLE.count("@media (max-width") == 2


def test_nothing_may_scroll_the_body_sideways() -> None:
    """A wide table or a long stem must scroll inside its own box, not the page."""
    assert "html{overflow-x:hidden}" in page._STYLE
    assert "main{padding:18px 26px 140px;min-width:0;max-width:100%}" in page._STYLE
    assert "main *{overflow-wrap:anywhere}" in page._STYLE
    assert "#why .tw{overflow-x:auto;max-width:100%}" in page._STYLE


def test_the_overlays_fit_a_narrow_viewport() -> None:
    """The scrim panel is the element most likely to break on a small screen."""
    assert "width:min(330px,calc(100vw - 28px))" in page._STYLE
    assert "width:min(760px,94vw)" in page._STYLE
    assert "max-width:calc(100vw - 28px)" in page._STYLE


def test_the_why_tables_scroll_inside_their_own_box() -> None:
    """Five columns of gate detail do not fit a phone; the table scrolls, the page does not."""
    assert page._SCRIPT.count('<div class="tw">') == 4
    assert page._SCRIPT.count("</table></div>") == 4


def test_the_rail_collapses_on_a_narrow_screen() -> None:
    """A 1,514-entry outline must not be the first screen on a phone."""
    corpus = page.Corpus()
    corpus.add(_row("sub-a"))
    document = page.render(corpus, "Review")
    assert 'id="railtoggle"' in document
    assert 'id="railbody"' in document
    assert "#rail.open #railbody{display:block}" in page._STYLE
    assert "railToggle.setAttribute('aria-expanded'" in page._SCRIPT


def test_the_row_carries_the_tasks_own_declared_cast() -> None:
    """cinderella-story declares its cast; picture-description cannot and declares none."""
    from senselab.audio.workflows.triage.nodes.branches import expected_names

    assert len(expected_names("cinderella-story")) > 0
    assert expected_names("picture-description") == ()
    assert expected_names("free-speech") == ()


def test_the_stimulus_tally_reports_the_answer_not_the_declaration(tmp_path: Path) -> None:
    """A task can declare no prompt text and still have its findings checked, against its cast."""
    records = [
        _word(0, "alpha"),
        _word(1, "beta"),
        _word(2, "gamma"),
        _entity(
            "pii-1",
            "pii",
            {"category": "NAME", "source": "rules/ner", "haystack": "consensus", "in_stimulus": True},
            [0, 1],
        ),
        _entity(
            "pii-2",
            "pii",
            {"category": "PERSON", "source": "presidio", "haystack": "consensus", "in_stimulus": False},
            [1, 2],
        ),
        _entity(
            "pii-3",
            "pii",
            {"category": "MISC", "source": "rules/ner", "haystack": "consensus", "in_stimulus": None},
            [2, 3],
        ),
        *_label("assertion-1", "NAME", "word-0"),
        *_label("assertion-2", "PERSON", "word-1"),
        *_label("assertion-3", "MISC", "word-2"),
        _entity(
            "measurement-ex",
            "measurement",
            {"name": "redaction_exemptions", "expected_speech_declared": False, "n": 1, "n_findings": 3},
        ),
        _verdict("withheld", family="cinderella-story"),
    ]
    run_root = _store(tmp_path, "sub-aaa_ses-bbb_task-cinderella-story", records)
    row = page.recording_record(run_root, page.free_response_families())
    assert row is not None
    assert row["d"]["stim"] == [1, 1, 1]
    assert row["d"]["exempt"]["declared"] is False
    assert row["names"] > 0


def test_the_panel_no_longer_infers_unchecked_from_an_undeclared_prompt() -> None:
    """The old sentence asserted nothing could be checked whenever no prompt text was declared."""
    assert "so no finding could be checked against it and none was exempted" not in page._SCRIPT
    assert "no stimulus text and no declared cast" in page._SCRIPT
    assert "declared cast of " in page._SCRIPT
    assert "matched what the task " in page._SCRIPT


def test_the_pooled_record_carries_the_tally_and_the_cast() -> None:
    """Both vary per recording and per family, so the panel can state them."""
    pool = page.ValuePool()
    record = page.pooled_determination(
        {
            "redact": {"outcome": "", "why": ""},
            "nodes": [],
            "llm": {},
            "gates": {"applied": [], "flagging": [], "bounds": {}, "layers": {}, "group": ""},
            "ran": {},
            "absences": [],
            "exempt": {},
            "findings": [],
            "stim": [4, 2, 1],
            "names": 20,
        },
        pool,
    )
    assert record["s"] == [4, 2, 1]
    assert record["nn"] == 20


def _census_row(participant: str, **rest: Any) -> str:  # noqa: ANN401 -- mixed-type page fields
    """One extract line.

    Args:
        participant: The participant id.
        **rest: Overrides.

    Returns:
        The JSON line.
    """
    return json.dumps(_row(participant, **rest))


def test_census_counts_what_a_rebuild_is_judged_by(tmp_path: Path) -> None:
    """Findings, marks, release, the stimulus tri-state and the bracketed marks."""
    data = tmp_path / "rows.jsonl"
    data.write_text(
        "\n".join(
            [
                _census_row(
                    "sub-a",
                    rel="withheld",
                    pii=[
                        {"c": "NAME", "s": "rules/ner", "h": "consensus", "stim": 1},
                        {"c": "PERSON", "s": "presidio", "h": "consensus", "stim": -1},
                    ],
                    f=[_mark("k1", ["PERSON"], ["presidio"], 0, 1, brk=1)],
                    d={"llm": {"status": "disabled"}},
                ),
                _census_row("sub-b", rel="release_with_redaction", d={"llm": {"status": "clean"}}),
            ]
        )
        + "\n"
    )
    counted = page.census(data)
    totals = counted["totals"]
    assert totals["recordings"] == 2
    assert totals["findings"] == 2
    assert totals["marks"] == 1
    assert totals["in_stimulus"] == 1
    assert totals["unchecked"] == 1
    assert totals["bracketed_marks"] == 1
    assert totals["bracketed:PERSON"] == 1
    assert totals["release:withheld"] == 1
    assert totals["llm:disabled"] == 1
    assert totals["llm:clean"] == 1
    assert counted["families"]["free-speech"]["recordings"] == 2


def test_compare_reports_each_count_in_both_with_its_delta(tmp_path: Path) -> None:
    """The rebuild has to be measurable against the extract the current page was built from."""
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    before.write_text(_census_row("sub-a", pii=[{"c": "NAME", "s": "rules/ner", "h": "consensus", "stim": -1}]) + "\n")
    after.write_text(_census_row("sub-a", pii=[{"c": "NAME", "s": "rules/ner", "h": "consensus", "stim": 1}]) + "\n")
    moved = page.compare(before, after)
    assert moved["totals"]["unchecked"] == [1, 0, -1]
    assert moved["totals"]["in_stimulus"] == [0, 1, 1]
    assert moved["totals"]["recordings"] == [1, 1, 0]
    assert moved["families"]["free-speech"]["unchecked"] == [1, 0, -1]


def test_an_extract_carries_its_schema_version(tmp_path: Path) -> None:
    """The header is what lets the page know which graph the rows came from."""
    _store(tmp_path, _FAMILY_STEM, [_word(0, "one"), _verdict("release_with_redaction")])
    out = tmp_path / "out.jsonl"
    report = page.extract(tmp_path, out, workers=1)
    assert report["version"] == page.EXTRACT_VERSION
    first = json.loads(out.read_text().splitlines()[0])
    assert first == {"schema": page.EXTRACT_SCHEMA, "version": page.EXTRACT_VERSION}
    corpus = page.load(out)
    assert corpus.version == page.EXTRACT_VERSION
    assert corpus.recordings == 1


def test_an_extract_without_a_header_reads_as_the_older_version(tmp_path: Path) -> None:
    """The extract the current page was built from predates the header."""
    data = tmp_path / "old.jsonl"
    data.write_text(json.dumps(_row("sub-a")) + "\n")
    corpus = page.load(data)
    assert corpus.version == 2
    assert corpus.recordings == 1


def test_the_page_refuses_to_describe_a_reviewer_it_cannot_vouch_for(tmp_path: Path) -> None:
    """Version 2 rows under version 3 wording would make the not_run sentence a false claim."""
    data = tmp_path / "old.jsonl"
    data.write_text(json.dumps(_row("sub-a")) + "\n")
    document = page.render(page.load(data), "Review")
    assert "written before the release axis and the reviewer ladder changed" in document
    fresh = page.Corpus()
    fresh.version = page.EXTRACT_VERSION
    fresh.add(_row("sub-a"))
    assert "written before the reviewer ladder changed" not in page.render(fresh, "Review")


def test_the_status_bar_stays_a_corner_pill_on_a_phone() -> None:
    """A fixed full-width band covers a line of transcript at every scroll position."""
    assert "#status{left:auto;right:8px;bottom:8px" in page._STYLE
    assert "#status{left:8px;right:8px;text-align:center}" not in page._STYLE
    assert "#status{pointer-events:none}" in page._STYLE


def test_the_page_states_corpus_wide_that_no_review_happened() -> None:
    """r3 is CPU-only with the reviewer off; disabled must not read as a review that found nothing."""
    corpus = page.Corpus()
    corpus.version = page.EXTRACT_VERSION
    for number in range(3):
        corpus.add(_row(f"sub-{number}", d={"llm": {"status": "disabled"}}))
    document = page.render(corpus, "Review")
    assert "did not run on any of these 3 recordings" in document
    assert "disabled 3" in document
    assert "no reviewer verdict exists to read" in document
    assert "redaction.llm_check.enabled" in document


def test_the_banner_goes_away_once_a_reviewer_actually_ran() -> None:
    """One real reading is enough to make the corpus-wide claim false."""
    corpus = page.Corpus()
    corpus.version = page.EXTRACT_VERSION
    corpus.add(_row("sub-a", d={"llm": {"status": "disabled"}}))
    corpus.add(_row("sub-b", d={"llm": {"status": "clean"}}))
    assert "did not run on any of these" not in page.render(corpus, "Review")


def test_a_marks_stimulus_state_is_tri_state_like_its_findings(tmp_path: Path) -> None:
    """A boolean here collapses "nothing to check against" into "checked and did not match"."""
    records = [
        _word(0, "alpha"),
        _entity(
            "pii-1",
            "pii",
            {"category": "NAME", "source": "rules/ner", "haystack": "consensus", "in_stimulus": None},
            [0, 1],
        ),
        *_label("assertion-1", "NAME", "word-0"),
        _verdict("withheld", family="cinderella-story"),
    ]
    run_root = _store(tmp_path, "sub-aaa_ses-bbb_task-cinderella-story", records)
    row = page.recording_record(run_root, page.free_response_families())
    assert row is not None
    assert row["f"][0]["stim"] == -1


def test_a_mark_matched_by_the_stimulus_reads_as_matched(tmp_path: Path) -> None:
    """One matching finding is enough for the mark, whatever the others said."""
    records = [
        _word(0, "alpha"),
        _entity(
            "pii-1",
            "pii",
            {
                "category": "NAME",
                "source": "rules/ner",
                "haystack": "consensus",
                "in_stimulus": False,
                "word_ids": ["word-0"],
            },
            [0, 1],
        ),
        _entity(
            "pii-2",
            "pii",
            {
                "category": "NAME",
                "source": "presidio",
                "haystack": "consensus",
                "in_stimulus": True,
                "word_ids": ["word-0"],
            },
            [0, 1],
        ),
        *_label("assertion-1", "NAME", "word-0"),
        _verdict("release_with_redaction", family="cinderella-story"),
    ]
    run_root = _store(tmp_path, "sub-aaa_ses-bbb_task-cinderella-story", records)
    row = page.recording_record(run_root, page.free_response_families())
    assert row is not None
    assert row["f"][0]["stim"] == 1


@pytest.mark.parametrize(
    ("states", "expected"),
    [([True], 1), ([False], 0), ([None], -1), ([None, False], 0), ([False, True], 1), ([], -1)],
)
def test_the_mark_stimulus_fold(states: list[Any], expected: int) -> None:
    """Matched wins over checked, and checked wins over uncheckable."""
    findings = [
        page.Entity(id=f"pii-{n}", prov_type="pii", extent=(0.0, 1.0), attributes={"in_stimulus": state})
        for n, state in enumerate(states)
    ]
    assert page._mark_stimulus(findings) == expected


def test_the_panel_names_all_three_stimulus_states_on_a_mark() -> None:
    """The finding panel must not leave "not asked" looking like "asked and no"."""
    assert "checked, not in the stimulus" in page._SCRIPT
    assert "no stimulus to check against" in page._SCRIPT


def test_the_page_says_when_a_flag_lands_on_a_transcript_no_detector_read() -> None:
    """The 13,810 the scan declined: a flag there is a reading about the gate, not the detectors."""
    assert "declined" in page._SCRIPT
    assert "It flagged a transcript no detector read." in page._SCRIPT
    assert "the only check on that gate" in page._SCRIPT


def test_the_page_reports_the_reviewers_other_two_readings() -> None:
    """Over-redaction and a second speaker are separate judgments and are rendered separately."""
    assert "would stop removing" in page._SCRIPT
    assert "It would remove " in page._SCRIPT, "the proposal's other direction is a judgment too"
    assert "more than one person speaking" in page._SCRIPT
    assert "A reading of the transcript, not of the audio." in page._SCRIPT


def test_the_page_script_is_valid_javascript() -> None:
    r"""The guard 112 assertions on this page did not have: does the script actually parse?

    ``_SCRIPT`` is a plain triple-quoted Python string, so a ``\\'`` written for JavaScript
    collapses to a bare apostrophe, closes the string it sits in and takes the rest of the file
    with it. Two of those shipped, and every keydown handler on the page stopped binding: the
    owner found it by pressing a key, because substring assertions cannot see a parse error.

    Skipped where node is not installed, so the suite still runs on a host without it.
    """
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the javascript cannot be parsed here")
    assert node is not None
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "script.js"
        script.write_text(page._SCRIPT, encoding="utf-8")
        result = subprocess.run([node, "--check", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"the page script does not parse:\n{result.stderr}"


def test_a_flagged_reading_names_the_ground_that_fired() -> None:
    """A flagged reading with no categories used to render as "It flagged 0" and say nothing.

    ``_is_flag`` is true on any of three grounds -- the redaction judged incomplete, the words
    judged to carry pii, or a proposal asking for a removal -- but the category list is populated
    only from the proposal's removal entries. On the running corpus 118 of the first 128 flagged
    readings carried no category, so the page has to name the grounds rather than count them.
    """
    assert "the words themselves carry something identifying" in page._SCRIPT
    assert "the applied redaction did not remove what identifies the speaker" in page._SCRIPT
    assert "it proposes " in page._SCRIPT
    assert "It flagged 0" not in page._SCRIPT


def test_a_flagged_reading_with_no_stated_ground_is_called_a_gap() -> None:
    """Flagged with nothing behind it is a hole in the record, and must not read as a clean pass."""
    assert "none of the three readings says why" in page._SCRIPT
    assert "That is a gap in the record, not a clean result." in page._SCRIPT


def test_the_speaker_reading_is_rendered_outside_the_pii_ladder() -> None:
    """Whether a second voice is in the words is not a question about pii, and a clean reading has one.

    It used to be rendered only inside the ``flagged`` branch, so a reading that found no pii and
    did notice a second speaker said nothing about it -- the one case the note exists for. The
    ladder's own last branch is the marker: the note must come after it.
    """
    ladder_ends = page._SCRIPT.index("No annotation was recorded.")
    speakers = page._SCRIPT.index("more than one person speaking")
    assert speakers > ladder_ends, "the speaker reading must not sit inside a pii-status branch"


def _ledger(masks: list[dict[str, Any]], proposals: list[dict[str, Any]], counts: dict[str, int]) -> dict[str, Any]:
    """The fold's ``pii_ledger`` measurement.

    Args:
        masks: The ledger's masks, each carrying its words and their states.
        proposals: The reviewer's placed ``redact`` entries.
        counts: The per-state counts.

    Returns:
        The JSONL record.
    """
    return _entity(
        "measurement-ledger",
        "measurement",
        {"name": "pii_ledger", "masks": masks, "proposals": proposals, "counts": counts},
    )


def test_the_marks_are_the_ledgers_spans_each_with_its_state(tmp_path: Path) -> None:
    """r6's free-speech-1 card: the date REDACT masked stays masked, and the conditions are shown too."""
    records = [
        _word(0, "this"),
        _word(1, "morning"),
        _word(2, "essential"),
        _word(3, "tremors"),
        _word(4, "today"),
        _pii("pii-1", "DATE_TIME", "presidio", 0, 2),
        *_label("assertion-1", "DATE_TIME", "word-1"),
        *_label("assertion-2", "DATE_TIME", "word-4"),
        _ledger(
            masks=[
                {
                    "category": "DATE_TIME",
                    "words": [
                        {"id": "word-0", "text": "this", "state": "unmasked_by_trim", "named": True, "content": False},
                        {"id": "word-1", "text": "morning", "state": "masked", "named": True, "content": True},
                    ],
                }
            ],
            proposals=[{"category": "CONDITION", "word_ids": ["word-2", "word-3"], "human_review": True}],
            counts={"masked_n": 1, "unmasked_by_trim_n": 1, "proposed_by_reviewer_n": 2},
        ),
        _verdict("withheld"),
    ]
    row = page.recording_record(_store(tmp_path, _FAMILY_STEM, records), page.free_response_families())
    assert row is not None
    states = [(mark["c"], mark["s"], mark["i"]) for mark in row["f"]]
    assert states == [
        (["DATE_TIME"], "unmasked_by_trim", [0, 1]),
        (["DATE_TIME"], "masked", [1, 2]),
        (["CONDITION"], "proposed_by_reviewer", [2, 4]),
        (["DATE_TIME"], "detected", [4, 5]),
    ]
    assert row["f"][1]["nm"] == 1
    assert row["f"][2]["hr"] == 1
    card = page.recording_html(row)
    text = card[card.index('<p class="text">') : card.index("</p>", card.index('<p class="text">'))]
    assert 'class="pii u-red"' in text and 'data-s="proposed_by_reviewer"' in text
    assert 'class="pii u-orange"' in text and 'class="pii u-green"' in text
    visible = re.sub(r"<[^>]+>", " ", text)
    for inline in ("reviewer would unmask", "human review", "proposes", "masked", "padding", "trim"):
        assert inline not in visible, inline
    orange = text[text.index('class="pii u-orange"') :]
    assert orange[: orange.index("</mark>")].count('class="cat"') == 0, "a padding span carries no label"
    assert '<span class="cat">CONDITION</span>' in text


def test_a_reviewer_condition_over_a_detector_mask_is_two_single_category_marks(tmp_path: Path) -> None:
    """r7: a PERSON mask with a reviewer CONDITION proposal on its words is two marks, not "PERSON+CONDITION"."""
    records = [
        _word(0, "Parkinson"),
        _word(1, "disease"),
        _pii("pii-1", "PERSON", "gliner", 0, 1),
        *_label("assertion-1", "PERSON", "word-0"),
        _ledger(
            masks=[
                {
                    "category": "PERSON",
                    "words": [
                        {"id": "word-0", "text": "Parkinson", "state": "masked", "named": False, "content": True}
                    ],
                }
            ],
            proposals=[
                {
                    "category": "CONDITION",
                    "word_ids": ["word-0", "word-1"],
                    "human_review": True,
                    "condition_kind": "cohort",
                    "cohort_diagnosis": "parkinsons_disease",
                }
            ],
            counts={"masked_n": 1, "proposed_by_reviewer_n": 2},
        ),
        _verdict("withheld"),
    ]
    row = page.recording_record(_store(tmp_path, _FAMILY_STEM, records), page.free_response_families())
    assert row is not None
    assert [(mark["c"], mark["s"], [o["c"] for o in mark["o"]]) for mark in row["f"]] == [
        (["PERSON"], "masked", ["CONDITION"]),
        (["CONDITION"], "proposed_by_reviewer", []),
    ]
    card = page.recording_html(row)
    text = card[card.index('<p class="text">') : card.index("</p>", card.index('<p class="text">'))]
    assert re.findall(r'data-c="([^"]*)"', text) == ["PERSON", "CONDITION", "CONDITION"]
    assert all("+" not in value for value in re.findall(r'data-c="([^"]*)"', text))
    assert all("+" not in value for value in re.findall(r'<span class="cat">([^<]*)</span>', text))
    inner = text[text.index('data-c="PERSON"') :]
    assert 'data-s="proposed_by_reviewer"' in inner[: inner.index("</mark>")], "the proposal nests inside the mask"
    assert 'data-kd="cohort"' in text and 'data-dx="parkinsons_disease"' in text


def test_a_word_two_detectors_mark_differently_is_two_single_category_marks(tmp_path: Path) -> None:
    """A word only detectors marked, as PERSON and as ORG, renders as a PERSON mark with an ORG mark inside."""
    records = [
        _word(0, "Acme"),
        *_label("assertion-1", "PERSON", "word-0"),
        *_label("assertion-2", "ORG", "word-0"),
        _verdict("release_without_redaction"),
    ]
    row = page.recording_record(_store(tmp_path, _FAMILY_STEM, records), page.free_response_families())
    assert row is not None
    rendered = page.recording_html(row)
    assert "PERSON+ORG" not in rendered
    assert re.findall(r'<mark class="pii[^"]*"[^>]*data-c="([^"]*)"', rendered) == ["PERSON", "ORG"]


def test_a_store_without_a_ledger_says_so_and_keeps_the_detectors_marks(tmp_path: Path) -> None:
    """A store folded before the ledger existed shows what the detectors marked, labelled as such."""
    records = [_word(0, "Tuesday"), *_label("assertion-1", "DATE_TIME", "word-0"), _verdict("withheld")]
    row = page.recording_record(_store(tmp_path, _FAMILY_STEM, records), page.free_response_families())
    assert row is not None
    assert [mark["s"] for mark in row["f"]] == ["detected"]
    assert "no PII ledger" not in page.recording_html(row)
    assert page.card_account(row)["led"] is None and "no PII ledger" in page._SCRIPT


def test_the_legend_is_not_a_reviewable_mark() -> None:
    """The page script binds every ``mark.pii``; the legend's swatches must not be among them."""
    assert "{legend}" in page._DOCUMENT
    assert 'class="pii' not in page.LEGEND_HTML
    assert page.LEGEND_HTML.count('class="swatch') == 3 == len(page.COLOUR_LEGEND)
    assert page.LEGEND_HTML.count('class="cat"') == 2, "the orange swatch carries no label"


def test_a_detector_mark_on_a_task_word_is_not_drawn(tmp_path: Path) -> None:
    """A finding placed back onto the task's own words is not a span: only residue words are drawn."""
    scan = {
        "name": "pii_scan",
        "scanned_by": ["gliner"],
        "residue_method": "stimulus_alignment",
        "residue_words_n": 1,
        "residue_content": True,
        "residue_word_ids": ["word-2"],
    }
    records = [
        _entity("measurement-scan", "measurement", scan),
        _word(0, "the"),
        _word(1, "caterpillar"),
        _word(2, "Maria"),
        *_label("assertion-1", "PERSON", "word-1"),
        *_label("assertion-2", "PERSON", "word-2"),
        _verdict("release_without_redaction"),
    ]
    row = page.recording_record(_store(tmp_path, _FAMILY_STEM, records), page.free_response_families())
    assert row is not None
    assert [(mark["s"], mark["i"]) for mark in row["f"]] == [("detected", [2, 3])]
    assert 'class="pii u-green"' in page.recording_html(row)


def test_a_card_with_no_consensus_words_shows_one_recogniser_s_stream(tmp_path: Path) -> None:
    """Owner, 2026-09-27: the corpus-fill case, only asr_qwen left words; the card shows that stream."""
    records = [
        _entity(
            "measurement-cw",
            "measurement",
            {"name": "asr_crisperwhisper", "role": "asr_hypothesis", "source": "asr_crisperwhisper", "words": []},
        ),
        _entity(
            "measurement-qwen",
            "measurement",
            {
                "name": "asr_qwen",
                "role": "asr_hypothesis",
                "source": "asr_qwen",
                "words": [{"text": "When", "start": 1.4, "end": 1.6}, {"text": "the", "start": 1.6, "end": 1.7}],
            },
        ),
        _verdict("not_assessed"),
    ]
    row = page.recording_record(_store(tmp_path, _FAMILY_STEM, records), page.free_response_families())
    assert row is not None
    assert row["ss"] == {"src": "asr_qwen", "n": 1}
    assert [entry[0] for entry in row["w"]] == ["When", "the"] and row["f"] == []
    card = page.recording_html(row)
    assert '<span class="ss">asr_qwen only</span>' in card
    assert '<p class="text">When the</p>' in card
    assert page.card_account(row)["ss"] == {"src": "asr_qwen", "n": 1}


def test_a_card_carries_no_explanation_of_its_status() -> None:
    """The release ground and the deciding reason are in the popup's data, not written on the card."""
    row = {
        "p": "p",
        "ses": "s",
        "task": "free-speech-1",
        "stem": "p_s_task-free-speech-1",
        "fam": "free-speech",
        "rel": "withheld",
        "rg": "the redaction reviewer proposed hiding more",
        "why": "folded",
        "rwhy": "",
        "scan": True,
        "res": None,
        "hk": "cohort",
        "w": [["Hello", 0, -1]],
        "f": [],
        "nl": 1,
        "nw": 1,
    }
    card = page.recording_html(row)
    assert "proposed hiding more" not in card and "folded" not in card
    assert 'data-hk="cohort"' in card
    assert page.card_account(row)["rg"] == "the redaction reviewer proposed hiding more"


def test_the_condition_filter_separates_the_study_s_conditions() -> None:
    """The rail offers the cohort conditions apart from every other, as a filter rather than as text."""
    assert '<select id="hkf">' in page._DOCUMENT
    assert "r.dataset.hk===hk" in page._SCRIPT
    assert [value for value, _ in page.CONDITION_SECTIONS] == ["cohort", "other", "none"]


def test_both_themes_define_every_colour_and_the_toggle_is_wired() -> None:
    """Owner, 2026-09-27: a dark/light toggle; the system decides until the reader does, and it persists."""
    style = page._STYLE
    light = style[style.index(":root{") : style.index("}", style.index(":root{"))]
    dark = style[style.index(':root[data-theme="dark"]{') :]
    dark = dark[: dark.index("}")]
    for name in ("--bg", "--fg", "--card", "--line", "--ured", "--ugreen", "--uorange", "--catbg", "--catfg"):
        assert f"{name}:" in light and f"{name}:" in dark, name
    assert ':root:not([data-theme="light"])' in style[style.index("@media (prefers-color-scheme:dark)") :]
    assert 'id="themetoggle"' in page._DOCUMENT and "{theme_boot}" in page._DOCUMENT
    assert "themeBtn.addEventListener('click'" in page._SCRIPT
    assert "localStorage.setItem(THEMEKEY,next)" in page._SCRIPT and page.THEME_KEY in page._THEME_BOOT
    assert "try{" in page._THEME_BOOT and "catch(e){}" in page._THEME_BOOT


def test_the_theme_boot_script_is_valid_javascript() -> None:
    """The head script runs before the page paints; a parse error there would leave the theme unset."""
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    assert node is not None
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(page._THEME_BOOT)
    result = subprocess.run([node, "--check", handle.name], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_a_mark_shows_one_category_on_its_label_and_on_data_c() -> None:
    """Owner, 2026-09-27/28: a span never reads "PERSON+DATE_TIME+MISC", neither visibly nor on ``data-c``."""
    mark = {
        "k": "k",
        "c": ["DATE_TIME", "PERSON"],
        "s": "masked",
        "d": ["rules"],
        "brk": 0,
        "tx": -1,
        "nt": 1,
        "stim": -1,
    }
    mark["c"] = ["DATE_TIME"]
    rendered = page._mark(mark, [["Alan's", 0, 0]])
    assert '<span class="cat">DATE_TIME</span>' in rendered
    assert 'data-c="DATE_TIME"' in rendered
    assert "+" not in rendered


def test_an_agreeing_redact_entry_draws_no_red_on_words_the_release_shows() -> None:
    """r8's open-response card: an entry agreeing with DATE_TIME masks leaves "or so" unmarked, as released."""
    ledger = {
        "masks": [
            {"category": "DATE_TIME", "words": [{"id": "w-week", "state": "masked", "content": True}]},
            {"category": "DATE_TIME", "words": [{"id": "w-ago", "state": "masked", "content": True}]},
        ],
        "proposals": [
            {
                "category": "DATE_TIME",
                "agreement": "masked",
                "word_ids": ["w-a", "w-week", "w-or", "w-so", "w-ago", "w-that"],
            },
            {"category": "LOCATION", "agreement": "new", "word_ids": ["w-wisconsin"]},
        ],
    }
    states = page.word_states(ledger)
    assert {word_id for word_id, state in states.items() if state["s"] == page.PROPOSED_BY_REVIEWER} == {"w-wisconsin"}
    assert states["w-week"]["s"] == page.MASKED and states["w-week"]["pr"] == 1
    assert not {"w-a", "w-or", "w-so", "w-that"} & set(states)


def test_a_proposal_marks_only_its_content_words() -> None:
    """r9's free-speech-2 card: a condition quote's function words get no mark; a trimmed "to" stays orange."""
    ledger = {
        "masks": [
            {
                "category": "DATE_TIME",
                "words": [{"id": "w-to", "state": "unmasked_by_trim", "content": False, "named": True}],
            }
        ],
        "proposals": [
            {
                "category": "CONDITION",
                "agreement": "new",
                "human_review": True,
                "word_ids": ["w-syn", "w-joint", "w-cyst", "w-between", "w-cone", "w-one", "w-and", "w-ctwo"],
                "texts": ["synovial", "joint", "cyst", "between", "Cone", "one", "and", "Ctwo"],
            },
            {
                "category": "CONDITION",
                "agreement": "new",
                "human_review": True,
                "word_ids": ["w-pron", "w-change", "w-in", "w-my", "w-voice"],
                "texts": ["pronounced", "change", "in", "my", "voice"],
            },
        ],
    }
    states = page.word_states(ledger)
    proposed = {word_id for word_id, state in states.items() if state["s"] == page.PROPOSED_BY_REVIEWER}
    assert not {"w-and", "w-in", "w-my"} & proposed
    assert {"w-syn", "w-joint", "w-cyst", "w-voice"} <= proposed
    assert ("w-between" in proposed) == page.is_content_word("between"), "follows the one definition"
    assert states["w-to"]["s"] == page.UNMASKED_BY_TRIM
    assert page.MARK_COLOURS[states["w-to"]["s"]] == "orange"


def test_the_ledgers_content_ids_decide_which_proposal_words_mark() -> None:
    """Where the fold recorded ``content_ids``, those and only those words are proposed."""
    proposal = {"word_ids": ["a", "b", "c"], "texts": ["in", "my", "voice"], "content_ids": ["c"]}
    assert page.proposal_content_ids(proposal) == ["c"]
    assert page.proposal_content_ids({"word_ids": ["a", "b", "c"], "texts": ["in", "my", "voice"]}) == ["c"]


def test_every_checkbox_facet_has_all_and_none(tmp_path: Path) -> None:
    """Each checkbox facet group in the rail carries an all and a none control wired to its class."""
    corpus = page.Corpus()
    corpus.add(_row("sub-a", f=[_mark("k1", ["PERSON"], ["presidio"], 0, 1)]))
    document = page.render(corpus, "Review")
    classes = {"fam-f", "rel-f", "lang-f", "cat-f", "det-f"}
    for cls in classes:
        assert f"','{cls}']" in document, cls
    for button in ("allrel", "norel", "allfam", "nofam", "alllang", "nolang", "allcat", "nocat", "alldet", "nodet"):
        assert f'<button id="{button}" type="button">' in document


def test_a_trim_released_word_is_dashed_and_a_mask_or_proposal_is_solid() -> None:
    """Orange (released by the trim) is dashed in both themes; red and green stay solid."""
    rules = {m.group(1): m.group(2) for m in re.finditer(r"mark\.pii\.(u-\w+)[^{]*\{([^}]*)\}", page._STYLE)}
    assert "dashed" in rules["u-orange"]
    assert "dashed" not in rules["u-red"] and "dashed" not in rules["u-green"]
    assert "mark.swatch.u-orange" in page._STYLE, "the legend swatch shares the dashed rule"


def test_a_language_code_falls_under_english_spanish_or_itself() -> None:
    """Owner, 2026-10-01: es-419 is Spanish; regional English is English; an empty code is unknown."""
    assert page.language_name("en") == "English"
    assert page.language_name("en-US") == "English"
    assert page.language_name("es") == "Spanish"
    assert page.language_name("es-419") == "Spanish"
    assert page.language_name("") == "unknown"
    assert page.language_name("fr") == "fr"


def test_the_language_facet_filters_cards_and_the_card_shows_the_code(tmp_path: Path) -> None:
    """Each language present is a checkbox; a card carries its facet name and keeps the code as a title."""
    corpus = page.Corpus()
    corpus.add(_row("sub-a", lang="en"))
    corpus.add(_row("sub-b", lang="es-419"))
    document = page.render(corpus, "Review")
    assert '<input type="checkbox" class="lang-f" value="English" checked>' in document
    assert '<input type="checkbox" class="lang-f" value="Spanish" checked>' in document
    assert 'data-lang="Spanish"' in document and 'title="es-419">Spanish</span>' in document
    assert "langs.has(r.dataset.lang)" in document


def test_the_language_comes_from_the_recording_sidecar(tmp_path: Path) -> None:
    """The run's run.json names the recording; its acoustic-task sidecar carries the language."""
    audio = tmp_path / "bids" / "sub-a" / "ses-b" / "audio"
    audio.mkdir(parents=True)
    wav = audio / "sub-a_ses-b_task-free-speech-1.wav"
    wav.write_bytes(b"")
    (audio / "sub-a_ses-b_task-free-speech-1_acoustictask-metadata.json").write_text(json.dumps({"language": "es-419"}))
    run_root = tmp_path / "out" / "run-a"
    (run_root / "run").mkdir(parents=True)
    (run_root / "run" / "run.json").write_text(json.dumps({"source": str(wav)}))
    assert page.recording_language(run_root) == "es-419"
    assert page.recording_language(tmp_path / "missing") == ""
