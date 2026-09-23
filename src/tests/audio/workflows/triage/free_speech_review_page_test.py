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
    assert row["nw"] == 3
    assert row["nl"] == 2
    assert row["w"][1] == ["[UH]", 1]


def test_recording_record_skips_a_family_outside_the_pattern(tmp_path: Path) -> None:
    """An item-list task under the same tree is not a free response."""
    run_root = _store(
        tmp_path, "sub-aaa_ses-bbb_task-animal-fluency", [_verdict("releasable", family="animal-fluency")]
    )
    assert page.recording_record(run_root, page.free_response_families()) is None


def test_recording_record_carries_the_findings(tmp_path: Path) -> None:
    """Each live pii entity becomes one finding row, category and detector only."""
    records = [
        _word(0, "Tuesday"),
        _entity("pii-1", "pii", {"category": "DATE_TIME", "source": "presidio", "haystack": "consensus"}, [0, 1]),
        _entity("assertion-1", "assertion", {"verb": "label", "label": "pii", "category": "DATE_TIME"}),
        {"record": "relation", "relation": "wasDerivedFrom", "source": "assertion-1", "target": "word-0"},
        _verdict("withheld"),
    ]
    row = page.recording_record(_store(tmp_path, _FAMILY_STEM, records), page.free_response_families())
    assert row is not None
    assert row["pii"] == [{"c": "DATE_TIME", "s": "presidio", "h": "consensus", "stim": False}]
    assert row["w"][0] == ["Tuesday", 0, ["DATE_TIME"]]


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


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        ([["plain", 0]], "plain"),
        ([["[UH]", 1]], '<span class="bracket">[UH]</span>'),
        ([["a<b", 0]], "a&lt;b"),
    ],
)
def test_paragraph_escapes_and_marks_brackets(words: list[list[Any]], expected: str) -> None:
    """A bracketed token renders as its own class; every surface is escaped."""
    assert page.paragraph(words) == expected


def test_paragraph_merges_an_adjacent_run_of_one_category() -> None:
    """Two adjacent words under one category are one mark, labelled once."""
    rendered = page.paragraph([["Ada", 0, ["PERSON"]], ["Byron", 0, ["PERSON"]], ["spoke", 0]])
    assert rendered.count("<mark") == 1
    assert rendered.count('class="cat"') == 1
    assert "Ada Byron" in rendered
    assert rendered.endswith("spoke")


def test_paragraph_separates_neighbouring_categories() -> None:
    """Adjacent words under different categories stay separate marks."""
    rendered = page.paragraph([["Ada", 0, ["PERSON"]], ["Tuesday", 0, ["DATE_TIME"]]])
    assert rendered.count("<mark") == 2


def test_paragraph_labels_a_word_carrying_two_categories() -> None:
    """Two categories on one word render as one joined label."""
    assert ">PERSON+ORG<" in page.paragraph([["Acme", 0, ["PERSON", "ORG"]]])


def test_render_is_self_contained_and_groups_by_participant(tmp_path: Path) -> None:
    """The page references nothing external and carries one section per participant."""
    data = tmp_path / "rows.jsonl"
    data.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "p": "sub-aaa",
                    "ses": "ses-b",
                    "task": "free-speech-1",
                    "fam": "free-speech",
                    "rel": "withheld",
                    "rg": None,
                    "tri": "pass",
                    "why": "w",
                    "rwhy": "r",
                    "scan": True,
                    "w": [["one", 0], ["[UH]", 1]],
                    "pii": [{"c": "PERSON", "s": "gliner", "h": "consensus", "stim": False}],
                    "nw": 2,
                    "nl": 1,
                    "ch": 7,
                },
                {
                    "p": "sub-ccc",
                    "ses": "ses-d",
                    "task": "cinderella-story",
                    "fam": "cinderella-story",
                    "rel": "nothing_to_redact",
                    "rg": "the scan ran over the transcript and found nothing to redact",
                    "tri": "pass",
                    "why": "",
                    "rwhy": "",
                    "scan": True,
                    "w": [["two", 0]],
                    "pii": [],
                    "nw": 1,
                    "nl": 1,
                    "ch": 3,
                },
            )
        )
        + "\n"
    )
    corpus = page.load(data)
    document = page.render(corpus, "Review")
    assert corpus.recordings == 2
    assert 'id="sub-aaa"' in document and 'id="sub-ccc"' in document
    assert "http://" not in document and "https://" not in document
    assert "<img" not in document and "<link" not in document
    assert "src=" not in document
    assert "PERSON" in document
    assert "the scan ran over the transcript and found nothing to redact" in document


def test_write_pages_shards_by_participant_with_an_index(tmp_path: Path) -> None:
    """Sharding splits participants into files and writes an index over them, all mode 600."""
    corpus = page.Corpus()
    for number in range(5):
        corpus.add(
            {
                "p": f"sub-{number}",
                "ses": "ses-a",
                "task": "free-speech-1",
                "fam": "free-speech",
                "rel": "releasable",
                "rg": None,
                "tri": "pass",
                "why": "",
                "rwhy": "",
                "scan": True,
                "w": [["word", 0]],
                "pii": [],
                "nw": 1,
                "nl": 1,
                "ch": 4,
            }
        )
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
    corpus.add(
        {
            "p": "sub-a",
            "ses": "ses-a",
            "task": "free-speech-1",
            "fam": "free-speech",
            "rel": "releasable",
            "rg": None,
            "tri": "pass",
            "why": "",
            "rwhy": "",
            "scan": True,
            "w": [["word", 0]],
            "pii": [],
            "nw": 1,
            "nl": 1,
            "ch": 4,
        }
    )
    out = tmp_path / "page.html"
    page.write_pages(corpus, out, "Review", shard_size=0)
    assert out.stat().st_mode & 0o777 == 0o600
