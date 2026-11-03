"""The 1.x producer's calls into 1.x, run as a child process against a recording stand-in for the 1.x package."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[4]
PRODUCER = ROOT / "benchmarks" / "parity_reference_1x.py"
STAND_IN = {
    "__init__.py": "",
    "_log.py": (
        "import json, os\n"
        "def log(call, **fields):\n"
        "    with open(os.environ['FAKE_1X_LOG'], 'a') as f:\n"
        "        f.write(json.dumps({'call': call, **fields}) + '\\n')\n"
    ),
    "audio/__init__.py": "",
    "audio/data_structures/__init__.py": (
        "class Audio:\n    def __init__(self, filepath):\n        self.filepath = filepath\n"
    ),
    "audio/tasks/__init__.py": "",
    "audio/tasks/preprocessing/__init__.py": (
        "def downmix_audios_to_mono(audios):\n    return audios\n"
        "def resample_audios(audios, rate):\n    return audios\n"
    ),
    "audio/tasks/speaker_embeddings/__init__.py": (
        "from senselab._log import log\n"
        "class _Vector(list):\n"
        "    def flatten(self):\n        return self\n"
        "    def tolist(self):\n        return list(self)\n"
        "def extract_speaker_embeddings_from_audios(audios, model, device=None):\n"
        "    log('embed', audios=[a.filepath for a in audios])\n"
        "    return [_Vector([float(len(audios))]) for _ in audios]\n"
    ),
    "audio/tasks/speech_to_text/__init__.py": (
        "from senselab._log import log\n"
        "class _Line:\n    text, speaker, start, end, avg_logprob = 'a', None, None, None, -0.1\n"
        "def transcribe_audios(audios, model, language=None, device=None, **kwargs):\n"
        "    log('transcribe', n=len(audios), language=language, kwargs=kwargs)\n"
        "    return [_Line() for _ in audios]\n"
    ),
    "utils/__init__.py": "",
    "utils/data_structures/__init__.py": (
        "class DeviceType(str):\n    pass\n"
        "class HFModel:\n    def __init__(self, **kw):\n        self.kw = kw\n"
        "SpeechBrainModel = PyannoteAudioModel = HFModel\n"
        "def Language(language_code):\n    return 'Language:' + language_code\n"
    ),
}


def _produce(tmp_path: Path, document: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the producer on `document`; return (the 1.x calls it made, the records it wrote by item id)."""
    fake = tmp_path / "fake1x"
    for name, body in STAND_IN.items():
        path = fake / "senselab" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, "utf-8")
    items, out, log = tmp_path / "items.json", tmp_path / "records", tmp_path / "calls.jsonl"
    items.write_text(json.dumps(document), "utf-8")
    done = subprocess.run(  # noqa: S603 - the repo's own script under this interpreter
        [sys.executable, str(PRODUCER), "--items", str(items), "--out", str(out)],
        env={**os.environ, "PYTHONPATH": str(fake), "FAKE_1X_LOG": str(log)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    calls = [json.loads(line) for line in log.read_text("utf-8").splitlines()]
    records = {p.stem: json.loads(p.read_text("utf-8")) for p in out.glob("*.json")}
    return calls, records


MODEL = {"id": "org/model", "revision": "0" * 40}
ITEMS = {f"u{k}": {"audio": f"/data/u{k}.flac"} for k in range(3)}


@pytest.mark.logic
def test_speaker_embeddings_embed_each_item_alone(tmp_path: Path) -> None:
    """Every item is one 1.x call over that item alone, so no item is padded to another's length."""
    calls, records = _produce(tmp_path, {"task": "speaker_embeddings", "params": {}, "model": MODEL, "items": ITEMS})
    assert sorted(c["audios"][0] for c in calls) == sorted(i["audio"] for i in ITEMS.values())
    assert all(len(c["audios"]) == 1 for c in calls)
    assert {i: r["vector"] for i, r in records.items()} == {i: [1.0] for i in ITEMS}


@pytest.mark.logic
@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"language": None, "timestamps": "none"}, {"language": None, "kwargs": {"return_timestamps": False}}),
        (
            {"language": "en", "timestamps": "segment"},
            {"language": "Language:en", "kwargs": {"return_timestamps": True}},
        ),
        ({"language": None, "timestamps": "word"}, {"language": None, "kwargs": {"return_timestamps": "word"}}),
    ],
)
def test_speech_to_text_decodes_under_the_benchmark_params(
    tmp_path: Path, params: dict[str, Any], expected: dict[str, Any]
) -> None:
    """The 1.x call is given the benchmark's language and timestamp mode, not 1.x's own defaults."""
    calls, records = _produce(tmp_path, {"task": "speech_to_text", "params": params, "model": MODEL, "items": ITEMS})
    assert [{k: c[k] for k in ("language", "kwargs")} for c in calls] == [expected]
    assert set(records) == set(ITEMS)


@pytest.mark.logic
def test_a_param_the_1x_call_is_not_given_is_refused(tmp_path: Path) -> None:
    """A task param with no 1.x counterpart stops the run instead of being dropped."""
    document = {"task": "speaker_embeddings", "params": {"pooling": "mean"}, "model": MODEL, "items": ITEMS}
    (tmp_path / "items.json").write_text(json.dumps(document), "utf-8")
    done = subprocess.run(  # noqa: S603 - the repo's own script under this interpreter
        [sys.executable, str(PRODUCER), "--items", str(tmp_path / "items.json"), "--out", str(tmp_path / "out")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert done.returncode != 0 and "not given the params ['pooling']" in done.stderr
