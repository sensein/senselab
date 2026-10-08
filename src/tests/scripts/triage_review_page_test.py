"""Tests for ``scripts/triage_review_page.py``: extract a corpus, then render the page from the extract."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]


def _module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


script = _module("triage_review_page", ROOT / "scripts" / "triage_review_page.py")
fixtures = _module("review_page_fixtures", ROOT / "src/tests/audio/workflows/triage/review_page_test.py")


def test_extract_then_render_writes_a_page_over_the_corpus(tmp_path: Path) -> None:
    """One recording extracted, rendered into index.html and one side file, with the mark rules carried."""
    corpus = tmp_path / "corpus"
    fixtures._run_root(corpus)
    extract = tmp_path / "review.jsonl"
    assert script.main(["extract", str(corpus), "--out", str(extract), "--workers", "1"]) == 0
    lines = extract.read_text().splitlines()
    assert json.loads(lines[0]) == {"schema": script.EXTRACT_SCHEMA, "version": script.EXTRACT_VERSION}
    record = json.loads(lines[1])
    assert record["verdict"] == "pass" and record["speech"] is None
    assert script.main(["render", str(extract), "--out", str(tmp_path / "page"), "--title", "sample"]) == 0
    page = (tmp_path / "page" / "index.html").read_text()
    assert "<title>sample</title>" in page and "mark.pii" in page
    assert (tmp_path / "page" / "data" / "shard-0000.js").is_file()


def test_a_manifest_reads_only_the_run_roots_it_lists(tmp_path: Path) -> None:
    """``--manifest`` replaces the walk."""
    corpus = tmp_path / "corpus"
    run_root = fixtures._run_root(corpus)
    manifest = tmp_path / "roots.txt"
    manifest.write_text(f"{run_root}\n")
    out = tmp_path / "review.jsonl"
    summary = script.extract(corpus, out, 1, manifest=manifest)
    assert summary["run_roots"] == 1 and summary["recordings"] == 1
    assert len(script.load(out)) == 1


def test_a_speech_task_carries_its_marked_transcript(tmp_path: Path, monkeypatch: object) -> None:
    """The free-speech page's reader supplies the transcript view; the page keeps its HTML and findings."""
    fs = script._free_speech_page()
    row = {
        "w": [["my", 0, -1], ["name", 0, -1], ["is", 0, -1], ["Ada", 0, 0]],
        "f": [{"k": "k0", "c": ["NAME"], "s": "masked", "d": ["gliner"], "brk": 0, "tx": 0, "nt": 0, "stim": 0}],
        "pii": [{"c": "NAME", "s": "gliner", "h": "Ada", "stim": 0}],
        "rg": None,
        "why": "redacted",
        "rwhy": "masked a name",
        "hk": None,
        "np": [],
        "d": {"llm": {"status": "clean"}},
        "lang": "en",
    }
    monkeypatch.setattr(fs, "recording_record", lambda run_root, families: row)  # type: ignore[attr-defined]
    _asr_store(tmp_path, ["my", "name", "is", "Ada"])
    view = script.speech_view(tmp_path)
    assert view is not None
    assert 'class="pii u-' in view["html"] and "Ada" in view["html"]
    assert view["pii"][0]["h"] == "Ada" and view["llm"] == {"status": "clean"}


def _asr_store(run_root: Path, consensus: list[str]) -> None:
    """A store with two recognisers' transcripts and the consensus words over them."""

    def entity(eid: str, prov: str, attributes: dict) -> dict:
        return {"record": "entity", "id": eid, "prov_type": prov, "extent": [0.0, 1.0], "attributes": attributes}

    lines = [
        entity(
            "h0",
            "measurement",
            {
                "role": "asr_hypothesis",
                "source": "whisper",
                "model_id": "openai/whisper-large-v3-turbo",
                "transcript": "my name is Ada",
                "words": [],
            },
        ),
        entity(
            "h1",
            "measurement",
            {
                "role": "asr_hypothesis",
                "source": "canary",
                "model_id": "nvidia/canary-qwen-2.5b",
                "transcript": "my name is Ida",
                "words": [],
            },
        ),
    ]
    for index, text in enumerate(consensus):
        other = "Ida" if text == "Ada" else text
        lines.append(
            entity(
                f"w{index}",
                "word",
                {
                    "index": index,
                    "text": text,
                    "outcome": "variant" if other != text else "agreement",
                    "agreement": 0.5 if other != text else 1.0,
                    "sources": ["whisper", "canary"],
                    "readings": {"whisper": text, "canary": other},
                },
            )
        )
    (run_root / "run").mkdir(parents=True, exist_ok=True)
    (run_root / "run" / "store.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def test_the_consensus_transcript_carries_each_word_s_agreement_and_every_model(
    tmp_path: Path, monkeypatch: object
) -> None:
    """Consensus words wrap their position, PII marks stay around them, and each model's transcript is named."""
    fs = script._free_speech_page()
    row = {
        "w": [["my", 0, -1], ["name", 0, -1], ["is", 0, -1], ["Ada", 0, 0]],
        "f": [{"k": "k0", "c": ["NAME"], "s": "masked", "d": ["gliner"], "brk": 0, "tx": 0, "nt": 0, "stim": 0}],
        "ss": None,
    }
    monkeypatch.setattr(fs, "recording_record", lambda run_root, families: row)  # type: ignore[attr-defined]
    _asr_store(tmp_path, ["my", "name", "is", "Ada"])
    view = script.speech_view(tmp_path)
    assert view is not None
    assert view["shown"] == {"kind": "consensus", "source": None}
    assert [m["model_id"] for m in view["models"]] == ["openai/whisper-large-v3-turbo", "nvidia/canary-qwen-2.5b"]
    assert view["words"][3] == [0.5, "variant", ["Ada", "Ida"]]
    assert view["own"][1] == {"source": "canary", "model_id": "nvidia/canary-qwen-2.5b", "text": "my name is Ida"}
    assert '<span class="w" data-i="3">Ada</span></mark>' in view["html"]
    assert view["html"].count('class="w"') == 4


def test_without_consensus_words_the_one_model_shown_is_named(tmp_path: Path, monkeypatch: object) -> None:
    """The free-speech reader's single-model fallback is labelled with that model, its words unwrapped."""
    fs = script._free_speech_page()
    row = {"w": [["hello", 0, -1]], "f": [], "ss": {"src": "whisper", "n": 2}}
    monkeypatch.setattr(fs, "recording_record", lambda run_root, families: row)  # type: ignore[attr-defined]
    _asr_store(tmp_path, [])
    view = script.speech_view(tmp_path)
    assert view is not None
    assert view["shown"] == {"kind": "model", "source": "whisper"}
    assert 'class="w"' not in view["html"] and view["words"] == []
