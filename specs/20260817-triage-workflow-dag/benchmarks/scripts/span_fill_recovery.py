"""Controlled span-fill recovery experiment for YAMNet's periodic frame fill.

Builds a reference set of spans classified natively by YAMNet with no padding at all (extent
>= YAMNET_WINDOW_SECONDS, native top-1 score >= --min-confidence), trims each reference span to a
grid of shorter durations from both the start and the centre, runs each trimmed fragment through
the pipeline's own ``span_yamnet_input`` (the periodic fill) and through a zero-pad control, and
classifies every fragment with YAMNet in as few subprocess-venv calls as possible.

Two reference sources, both restricted to the recordings and subjects the run actually has local
audio for:

- ``store_span``: spans the triage pipeline itself proposed, read from a completed run's
  ``store.jsonl`` (extent, ``measure``, and the pipeline's own native ``span_yamnet`` measurement
  reused as the reference rather than recomputed).
- ``native_window``: YAMNet's own native hop windows (0.96 s, 0.48 s hop, no span-detection step
  at all) taken directly from *other* recordings of the same subjects that were never run through
  triage. Only the matched subjects' recordings had confidently-classified airway/transient
  content in ``store_span`` at all (the analysed recording was a narration task), so this source
  exists to populate that stratum from genuine unpadded YAMNet classifications rather than from
  pipeline-proposed spans. Overlapping windows within one recording are thinned by greedy
  non-overlap selection so references are not the same bout counted many times over.

Usage:
    uv run python span_fill_recovery.py \
        --stores "~/Downloads/triage_10subj_20260908/out/*/*/run/store.jsonl" \
        --bids-root ~/Downloads/b2ai_v31_bids_07_01_v3 \
        --supplement-glob "*Respiration-and-cough*.wav" \
        --out results.json

Reads only; the pipeline's own ``span_yamnet_input`` is imported unmodified. The zero-pad control
is implemented locally (mirrors its centring, fills with silence instead of tiled repeats), and is
not used by the pipeline itself.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from senselab.audio.data_structures import Audio  # noqa: E402
from senselab.audio.tasks.classification.api import classify_audios  # noqa: E402
from senselab.audio.tasks.classification.yamnet import YAMNET_WINDOW_SECONDS, span_yamnet_input  # noqa: E402
from senselab.audio.tasks.preprocessing import resample_audios  # noqa: E402
from senselab.utils.prov_store import ProvStore  # noqa: E402

TARGET_HZ = 16000  # matches resample.target_hz in the triage default config
TRIMS = (0.06, 0.10, 0.15, 0.20, 0.30, 0.40, 0.48, 0.70)
MIN_REFERENCE_DURATION = YAMNET_WINDOW_SECONDS
CLASSIFY_TOP_K = 10

STRATA: dict[str, tuple[str, ...]] = {
    "stationary": ("Hum", "Mains hum", "Buzz", "Buzzer", "Noise", "Static", "Silence", "White noise", "Pink noise"),
    "speech": ("Speech", "Narration, monologue", "Conversation", "Male speech", "Female speech", "Child speech"),
    "transient": (
        # Airway sounds named in the design brief.
        "Breathing",
        "Cough",
        "Gasp",
        "Snort",
        "Wheeze",
        "Sneeze",
        "Sigh",
        "Throat clearing",
        "Sniff",
        "Burp, eructation",
        "Hiccup",
        "Snoring",
        # "anything impulsive" (design brief's own broadening): single-shot, non-airway bursts that
        # a periodic tiling would turn into a pulse train the same way an airway burst would.
        "Explosion",
        "Burst, pop",
        "Whack, thwack",
        "Thump, thud",
        "Clang",
        "Crack",
        "Bang",
        "Slap, smack",
        "Gunshot, gunfire",
    ),
}


def stratum_of(label: str) -> str:
    """Map a native top-1 label to one of the three named strata, or ``"other"``."""
    for stratum, members in STRATA.items():
        if label in members:
            return stratum
    return "other"


def plain_audio(wav_path: Path) -> Audio:
    """Reproduce the triage pipeline's ``plain`` signal: mono, resampled, peak-scaled if clipping."""
    source = Audio(filepath=str(wav_path))
    mono = Audio(waveform=source.waveform.mean(dim=0, keepdim=True), sampling_rate=source.sampling_rate)
    [resampled] = resample_audios([mono], TARGET_HZ)
    peak = float(resampled.waveform.abs().max())
    if peak > 1.0:
        resampled = Audio(waveform=resampled.waveform / peak, sampling_rate=TARGET_HZ)
    return resampled


def zero_pad_input(audio: Audio, extent: tuple[float, float]) -> Audio:
    """The zero-pad control: same centring as ``span_yamnet_input``, silence instead of tiling."""
    start, end = extent
    rate = audio.sampling_rate
    first = int(round(start * rate))
    last = int(round(end * rate))
    span = audio.waveform[..., first:last]
    frame = int(round(YAMNET_WINDOW_SECONDS * rate))
    length = span.shape[-1]
    if length == 0 or length >= frame:
        return Audio(waveform=span.clone(), sampling_rate=rate)
    left = (frame - length) // 2
    right = frame - length - left
    before = torch.zeros((*span.shape[:-1], left), dtype=span.dtype)
    after = torch.zeros((*span.shape[:-1], right), dtype=span.dtype)
    return Audio(waveform=torch.cat((before, span, after), dim=-1), sampling_rate=rate)


def find_wav(bids_root: Path, basename: str) -> Path | None:
    """Locate a recording under the BIDS root by basename (the stored path is a cluster path)."""
    hits = list(bids_root.rglob(basename))
    return hits[0] if hits else None


def top1(raw_scores: dict[str, float]) -> tuple[str, float]:
    """Return the ``(label, score)`` with the highest score."""
    label = max(raw_scores, key=raw_scores.get)
    return label, float(raw_scores[label])


def build_reference_set(store: ProvStore, min_confidence: float) -> list[dict[str, Any]]:
    """Reference spans: extent >= 0.96s, native (unfilled) top-1 score >= min_confidence.

    The reference label/score is taken from the highest-scoring of the span's own native
    ``span_yamnet`` windows (there can be more than one when the span is longer than 0.96s);
    ``window_agreement`` records how many of those native windows share that top-1 label, out of
    how many total, as a homogeneity check on treating the whole span as one label.
    """
    spans = [s for s in store.entities("span") if s.extent and (s.extent[1] - s.extent[0]) >= MIN_REFERENCE_DURATION]
    measurements = [m for m in store.entities("measurement") if m.attributes.get("name") == "span_yamnet"]
    by_span: dict[str, list[Any]] = {}
    for m in measurements:
        by_span.setdefault(m.attributes["span_id"], []).append(m)

    reference = []
    for span in spans:
        windows = by_span.get(span.id, [])
        windows = [w for w in windows if "raw_scores" in w.attributes]
        if not windows:
            continue
        assert all(w.attributes.get("frame_filled") is False for w in windows), (
            f"span {span.id} >= {MIN_REFERENCE_DURATION}s native window unexpectedly filled"
        )
        scored = [(top1(w.attributes["raw_scores"]), w) for w in windows]
        (label, score), best_window = max(scored, key=lambda item: item[0][1])
        if score < min_confidence:
            continue
        agree = sum(1 for (lbl, _), _ in scored if lbl == label)
        reference.append(
            {
                "span_id": span.id,
                "extent": list(span.extent),
                "duration": span.extent[1] - span.extent[0],
                "measure": span.attributes.get("measure"),
                "native_label": label,
                "native_score": score,
                "stratum": stratum_of(label),
                "window_agreement": f"{agree}/{len(scored)}",
                "reference_window_extent": list(best_window.extent),
                "source": "store_span",
            }
        )
    return reference


def build_native_window_reference(
    wav_path: Path, windows: list[dict[str, Any]], min_confidence: float
) -> list[dict[str, Any]]:
    """Reference spans from YAMNet's own native hop windows over a whole, unsegmented recording.

    ``windows`` is one file's windowed ``classify_audios(model="yamnet")`` output. A window is
    eligible when it is a full 0.96s frame (the file's final window can be shorter) and its top-1
    score clears ``min_confidence``. Eligible windows are then thinned by greedy non-overlap
    selection in time order, so two 50%-overlapping windows over the same breath are not both kept
    as independent references.
    """
    candidates = []
    for w in windows:
        duration = float(w["end"]) - float(w["start"])
        if duration < MIN_REFERENCE_DURATION - 1e-6:
            continue
        raw = {k: v for pair in w["label_scores"] for k, v in pair.items()}
        label, score = top1(raw)
        if score < min_confidence:
            continue
        candidates.append((float(w["start"]), float(w["end"]), label, score))

    candidates.sort(key=lambda c: c[0])
    selected = []
    cursor = -1.0
    for start, end, label, score in candidates:
        if start < cursor:
            continue
        selected.append((start, end, label, score))
        cursor = end

    return [
        {
            "span_id": f"native_window:{wav_path.name}:{start:.3f}",
            "extent": [start, end],
            "duration": end - start,
            "measure": "native_window",
            "native_label": label,
            "native_score": score,
            "stratum": stratum_of(label),
            "window_agreement": "1/1",
            "reference_window_extent": [start, end],
            "source": "native_window",
        }
        for start, end, label, score in selected
    ]


def build_fragments(
    reference: list[dict[str, Any]], plain: Audio, subject: str
) -> tuple[list[Audio], list[dict[str, Any]]]:
    """One filled + one zero-padded fragment per (reference span, trim, position)."""
    audios: list[Audio] = []
    meta: list[dict[str, Any]] = []
    for ref in reference:
        start, end = ref["extent"]
        duration = ref["duration"]
        for trim in TRIMS:
            if trim >= duration:
                continue
            for position in ("start", "centre"):
                if position == "start":
                    frag_start = start
                else:
                    frag_start = start + (duration - trim) / 2.0
                frag_end = frag_start + trim
                filled, was_filled = span_yamnet_input(plain, (frag_start, frag_end))
                zeroed = zero_pad_input(plain, (frag_start, frag_end))
                assert was_filled, f"trim {trim}s unexpectedly not filled"
                for arm, audio in (("filled", filled), ("zero", zeroed)):
                    audios.append(audio)
                    meta.append(
                        {
                            **{k: v for k, v in ref.items() if k not in ("extent",)},
                            "subject": subject,
                            "trim": trim,
                            "position": position,
                            "arm": arm,
                            "fragment_extent": [frag_start, frag_end],
                        }
                    )
    return audios, meta


def classify_in_chunks(audios: list[Audio], chunk_size: int) -> list[list[dict[str, Any]]]:
    """One or a few subprocess-venv calls rather than one per fragment."""
    results: list[list[dict[str, Any]]] = []
    for i in range(0, len(audios), chunk_size):
        chunk = audios[i : i + chunk_size]
        results.extend(classify_audios(chunk, model="yamnet", top_k=CLASSIFY_TOP_K))
        print(f"  classified {min(i + chunk_size, len(audios))}/{len(audios)}", file=sys.stderr)
    return results


def main() -> None:
    """Build the reference set, build every fragment, classify, and write the raw results JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stores", default="~/Downloads/triage_10subj_20260908/out/*/*/run/store.jsonl")
    parser.add_argument("--bids-root", default="~/Downloads/b2ai_v31_bids_07_01_v3")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument(
        "--supplement-glob",
        default="*Respiration-and-cough*.wav",
        help="Extra recordings of the same matched subjects to mine for native_window references "
        "(no span-detection step), for strata the matched store_span recordings do not cover. "
        "Empty string disables this source.",
    )
    parser.add_argument("--out", default="span_fill_recovery_results.json")
    args = parser.parse_args()

    bids_root = Path(os.path.expanduser(args.bids_root))
    store_paths = sorted(glob.glob(os.path.expanduser(args.stores)))
    print(f"{len(store_paths)} stores found", file=sys.stderr)

    all_reference: list[dict[str, Any]] = []
    all_fragments: list[Audio] = []
    all_meta: list[dict[str, Any]] = []
    skipped_subjects: list[str] = []
    matched: list[tuple[str, Path]] = []  # (basename, wav_path) of store-matched recordings

    for store_path in store_paths:
        store = ProvStore.read_jsonl(store_path)
        recordings = [e for e in store.entities("stream") if e.attributes.get("name") == "recording"]
        if not recordings:
            continue
        basename = Path(recordings[0].attributes["path"]).name
        wav_path = find_wav(bids_root, basename)
        if wav_path is None:
            skipped_subjects.append(basename)
            continue
        matched.append((basename, wav_path))
        reference = build_reference_set(store, args.min_confidence)
        if not reference:
            continue
        plain = plain_audio(wav_path)
        fragments, meta = build_fragments(reference, plain, subject=basename)
        for r in reference:
            r["subject"] = basename
        all_reference.extend(reference)
        all_fragments.extend(fragments)
        all_meta.extend(meta)
        print(f"{basename}: {len(reference)} store_span references, {len(fragments)} fragments", file=sys.stderr)

    print(f"skipped (no local wav): {skipped_subjects}", file=sys.stderr)

    if args.supplement_glob:
        supplement_files: list[Path] = []
        for _basename, wav_path in matched:
            subject_dir = wav_path.parents[2]  # sub-*/ses-*/audio/file.wav
            supplement_files.extend(sorted(subject_dir.rglob(args.supplement_glob)))
        supplement_files = sorted(set(supplement_files))
        print(f"{len(supplement_files)} supplementary recordings matching {args.supplement_glob!r}", file=sys.stderr)
        if supplement_files:
            windowed = classify_audios(
                [Audio(filepath=str(p)) for p in supplement_files], model="yamnet", top_k=CLASSIFY_TOP_K
            )
            for wav_path, windows in zip(supplement_files, windowed):
                reference = build_native_window_reference(wav_path, windows, args.min_confidence)
                if not reference:
                    continue
                plain = plain_audio(wav_path)
                fragments, meta = build_fragments(reference, plain, subject=wav_path.name)
                for r in reference:
                    r["subject"] = wav_path.name
                all_reference.extend(reference)
                all_fragments.extend(fragments)
                all_meta.extend(meta)
                print(
                    f"{wav_path.name}: {len(reference)} native_window references, {len(fragments)} fragments",
                    file=sys.stderr,
                )

    print(f"total reference spans: {len(all_reference)}", file=sys.stderr)
    print(f"total fragments to classify: {len(all_fragments)}", file=sys.stderr)

    classified = classify_in_chunks(all_fragments, args.chunk_size)

    records = []
    for meta, windows in zip(all_meta, classified):
        if not windows:
            records.append({**meta, "top1_label": None, "top1_score": None, "labels_ranked": []})
            continue
        window = windows[0]
        raw = {k: v for pair in window["label_scores"] for k, v in pair.items()}
        ranked = sorted(raw.items(), key=lambda kv: -kv[1])
        label, score = ranked[0]
        records.append(
            {
                **meta,
                "top1_label": label,
                "top1_score": score,
                "labels_ranked": ranked,
            }
        )

    out_path = Path(os.path.expanduser(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"reference": all_reference, "records": records}, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
