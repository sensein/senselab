"""Standalone 1.x reference producer: run the 1.x task over the held-out items `parity_reference.py items` wrote.

Runs under the pinned `alpha` checkout in its own locked env; it imports 1.x senselab and nothing from 2.x.
Each 1.x call is given the 2.x task and backend params of the items file, the file `parity_reference.py run` also
reads them from. Writes one `<item id>.json` per item under `--out`: `{"task", "lines": [ScriptLine
fields]}` for speech_to_text and diarize, `{"task", "vector": [...]}` for speaker_embeddings, each item embedded
alone. `parity_reference.py convert` reads them.

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


TIMESTAMPS = {"none": False, "segment": True, "word": "word"}
"""1.x `return_timestamps` of each 2.x `timestamps`."""
PASSED = {
    "speech_to_text": {"max_new_tokens", "chunk_length_s", "batch_size"},
    "diarize": {"num_speakers", "min_speakers", "max_speakers", "exclusive"},
    "speaker_embeddings": set(),
}
"""The 2.x task and backend params each task's 1.x call takes under the same name."""
MAPPED = {"speech_to_text": {"language", "timestamps"}}
"""The 2.x params each task's 1.x call takes in another form."""


def call_params(task: str, params: dict[str, Any], backend_params: dict[str, Any]) -> dict[str, Any]:
    """Return the 1.x keyword arguments of the 2.x task `params` and `backend_params`, `language` an ISO code.

    Raises:
        SystemExit: a param the 1.x call does not take, or a `task` with no 1.x reference.
    """
    if task not in PASSED:
        raise SystemExit(f"no 1.x reference for task {task!r}")
    given = {**params, **{k: v for k, v in backend_params.items() if v is not None}}
    dropped = sorted(set(given) - PASSED[task] - MAPPED.get(task, set()))
    if dropped:
        raise SystemExit(f"the 1.x {task} call is not given the params {dropped}")
    kwargs = {k: v for k, v in given.items() if k in PASSED[task]}
    if task == "speech_to_text":
        kwargs |= {"language": given["language"], "return_timestamps": TIMESTAMPS[given["timestamps"]]}
    return kwargs


def produce(
    task: str, kwargs: dict[str, Any], model: dict[str, str], items: dict[str, dict[str, str]], device: str
) -> dict[str, dict[str, Any]]:
    """Return one output record per item id, the 1.x call given `kwargs` from `call_params`."""
    from senselab.utils.data_structures import DeviceType, HFModel

    where = DeviceType(device)
    audios = _audios(items)
    ids = list(items)
    if task == "speech_to_text":
        from senselab.audio.tasks.speech_to_text import transcribe_audios
        from senselab.utils.data_structures import Language

        code = kwargs.pop("language")
        lines = transcribe_audios(
            audios,
            HFModel(path_or_uri=model["id"], revision=model["revision"]),
            language=None if code is None else Language(language_code=code),
            device=where,
            **kwargs,
        )
        return {i: {"task": task, "lines": _lines([line])} for i, line in zip(ids, lines, strict=True)}
    if task == "diarize":
        from senselab.audio.tasks.speaker_diarization import diarize_audios
        from senselab.utils.data_structures import PyannoteAudioModel

        found = diarize_audios(
            audios, PyannoteAudioModel(path_or_uri=model["id"], revision=model["revision"]), device=where, **kwargs
        )
        return {i: {"task": task, "lines": _lines(lines)} for i, lines in zip(ids, found, strict=True)}
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


def main() -> None:
    """Read the items file, run the 1.x task, write the records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    document = json.loads(args.items.read_text("utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    kwargs = call_params(document["task"], document["params"], document["backend_params"])
    found = produce(document["task"], kwargs, document["model"], document["items"], args.device)
    for item_id, record in found.items():
        (args.out / f"{item_id}.json").write_text(json.dumps(record), "utf-8")


if __name__ == "__main__":
    main()
