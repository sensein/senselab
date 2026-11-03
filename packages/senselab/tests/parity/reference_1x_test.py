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
    calls, records = _produce(tmp_path, {"task": "speaker_embeddings", "model": MODEL, "items": ITEMS})
    assert sorted(c["audios"][0] for c in calls) == sorted(i["audio"] for i in ITEMS.values())
    assert all(len(c["audios"]) == 1 for c in calls)
    assert {i: r["vector"] for i, r in records.items()} == {i: [1.0] for i in ITEMS}
