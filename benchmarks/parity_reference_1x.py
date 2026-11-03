"""Standalone 1.x reference producer: run the 1.x task over the held-out items `parity_reference.py items` wrote.

Runs under the pinned `alpha` checkout in its own locked env; it imports 1.x senselab and nothing from 2.x.
Writes one `<item id>.json` per item under `--out`: `{"task", "lines": [ScriptLine fields]}` for speech_to_text and
diarize, `{"task", "vector": [...]}` for speaker_embeddings, each item embedded alone. `parity_reference.py convert`
reads them.

    python parity_reference_1x.py --items items.json --out records/ [--device cpu|cuda]
"""

import argparse
import json
from pathlib import Path
from typing import Any

RATE = 16000


def _audios(items: dict[str, dict[str, str]]) -> list[Any]:
    from senselab.audio.data_structures import Audio
    from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios

    loaded = [Audio(filepath=spec["audio"]) for spec in items.values()]
    return resample_audios(downmix_audios_to_mono(loaded), RATE)


def _lines(lines: list[Any]) -> list[dict[str, Any]]:
    return [{k: getattr(line, k, None) for k in ("text", "speaker", "start", "end", "avg_logprob")} for line in lines]


def produce(
    task: str, model: dict[str, str], items: dict[str, dict[str, str]], device: str
) -> dict[str, dict[str, Any]]:
    """Return one output record per item id."""
    from senselab.utils.data_structures import DeviceType, HFModel

    where = DeviceType(device)
    audios = _audios(items)
    ids = list(items)
    if task == "speech_to_text":
        from senselab.audio.tasks.speech_to_text import transcribe_audios

        lines = transcribe_audios(audios, HFModel(path_or_uri=model["id"], revision=model["revision"]), device=where)
        return {i: {"task": task, "lines": _lines([line])} for i, line in zip(ids, lines, strict=True)}
    if task == "diarize":
        from senselab.audio.tasks.speaker_diarization import diarize_audios
        from senselab.utils.data_structures import PyannoteAudioModel

        found = diarize_audios(
            audios, PyannoteAudioModel(path_or_uri=model["id"], revision=model["revision"]), device=where
        )
        return {i: {"task": task, "lines": _lines(lines)} for i, lines in zip(ids, found, strict=True)}
    if task == "speaker_embeddings":
        from senselab.audio.tasks.speaker_embeddings import extract_speaker_embeddings_from_audios
        from senselab.utils.data_structures import SpeechBrainModel

        encoder = SpeechBrainModel(path_or_uri=model["id"], revision=model["revision"])
        return {
            i: {
                "task": task,
                "vector": extract_speaker_embeddings_from_audios([a], encoder, device=where)[0].flatten().tolist(),
            }
            for i, a in zip(ids, audios, strict=True)
        }
    raise SystemExit(f"no 1.x reference for task {task!r}")


def main() -> None:
    """Read the items file, run the 1.x task, write the records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    document = json.loads(args.items.read_text("utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    for item_id, record in produce(document["task"], document["model"], document["items"], args.device).items():
        (args.out / f"{item_id}.json").write_text(json.dumps(record), "utf-8")


if __name__ == "__main__":
    main()
