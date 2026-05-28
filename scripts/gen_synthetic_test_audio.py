"""One-time generator for the committed synthetic test fixtures (T010a).

Produces the synthetic test corpus under ``src/tests/data_for_testing/synthetic/``
from **SpeechT5** (MIT) using fixed CMU-Arctic x-vector speaker embeddings and
a fixed seed. The corpus covers several content categories so a single
generation step yields fixtures usable for multiple tests:

  - **Speaker-profile subjects** (3 tiers): a confident target subject, a
    thin subject below the low-confidence floor, and an insufficient subject
    (sub-1s fragments). These exercise FR-005 / FR-014 / FR-016.
  - **Long passages** (Rainbow, North Wind, Grandfather): public-domain
    speech-research standards. Useful as additional enrollment material and
    as ASR/quality-test inputs.
  - **Short word tasks**: clinical-style animal-naming, picture-naming, and
    counting clips. Useful for the "drop sub-1s / short-file" path
    (FR-016) and as varied phonetic content.
  - **DDK approximation** (pa-ta-ka, one clip): labeled as a TTS approximation
    rather than authentic clinical DDK — useful as a corpus marker, not as a
    DDK-rate measurement input.
  - **PII free response**: fictional name/phone/email/address/DOB/medication
    sentences in a separate ``pii-test/`` group. They do **not** belong to
    any speaker-profile subject (so they don't bias aggregates) and reuse
    speaker A's voice for cross-test scenarios (e.g. overlay an intruder on
    a PII clip to exercise the full ASR + diarization + speaker-profile +
    PII stack as a future integration test).

This script is **not** part of CI. Run it **once** (or whenever the fixture
corpus needs to be regenerated), and commit its outputs to the repo. Tests
load the committed clips and compose contamination/overlay/noise scenarios
deterministically at run time (see
``src/tests/audio/workflows/speaker_profile/conftest.py``).

See research.md R13 ("Synthetic test fixtures") for rationale: pin local
model + commit outputs → replicability and license cleanliness without the
runtime cost / non-determinism of TTS in the test path.

Usage::

    uv run python scripts/gen_synthetic_test_audio.py

Expected runtime: a few minutes on CPU (~600 MB SpeechT5 download on first run).
The script prints the observed CMU-Arctic ``speaker_id`` for each picked
x-vector index so you can verify the deliberate-speaker selection at runtime.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

# ──────────────────────────────────────────────────────────────────────────
# Configuration — pinned, deterministic, license-safe.

# SpeechT5 (TTS + HiFi-GAN) is MIT-licensed, which makes its outputs safe to
# commit into an open-source repo. It's one of the few HF TTS lineups with
# permissive licensing (vs. MMS — CC-BY-NC — and Coqui XTTS — non-commercial).
SPEECHT5_TTS_ID = "microsoft/speecht5_tts"
SPEECHT5_HIFIGAN_ID = "microsoft/speecht5_hifigan"
SPEECHT5_REVISION = "main"

CMU_ARCTIC_DATASET_ID = "Matthijs/cmu-arctic-xvectors"

# Deliberate speaker selection (Option 1):
#
#   A — target subject. Index 7306 is the well-known SpeechT5 demo voice
#       (commonly maps to CMU-Arctic ``slt`` — US female). Used everywhere
#       in HF SpeechT5 tutorials, so it's the most-auditioned reference.
#   B — intruder. Index 5306 is selected to land on a *male* CMU-Arctic
#       speaker (likely ``rms``), guaranteeing perceptually obvious
#       contrast with A for SC-002 / SC-003 acceptance tests.
#   C — similar-timbre spare. Index 1000 is a *different female* CMU-Arctic
#       voice (likely ``clb`` or another) used as the cluster-ambiguity
#       stress test (FR-014: balanced-multi-speaker subject must be flagged
#       ``confidence="ambiguous"``).
#
# The *expected* speaker_id strings below are best-guess based on common HF
# SpeechT5 demo configurations. ``_load_speaker_embeddings`` logs the
# *observed* speaker_id from the dataset at runtime so the picked indices
# can be verified and adjusted on first generation.
CMU_ARCTIC_SPEAKERS: dict[str, tuple[str, int]] = {
    "A": ("slt", 7306),   # target — US female
    "B": ("rms", 5306),   # intruder — US male
    "C": ("clb", 1000),   # similar-timbre spare — US female (different person)
}

SAMPLE_RATE_HZ = 16000
SEED = 20260527


# ──────────────────────────────────────────────────────────────────────────
# Text constants — all public domain.

# Harvard / IEEE Recommended Practice for Speech Quality Measurements (IEEE
# Rec. 297) — public domain phonetically-rich sentences.
HARVARD_SENTENCES: tuple[str, ...] = (
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It is easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.",
    "A large size in stockings is hard to sell.",
)

# Long speech-research passages — all public domain.
RAINBOW_PASSAGE = (
    "When the sunlight strikes raindrops in the air, they act as a prism and "
    "form a rainbow. The rainbow is a division of white light into many "
    "beautiful colors. These take the shape of a long round arch, with its "
    "path high above, and its two ends apparently beyond the horizon."
)
NORTH_WIND_FULL = (
    "The North Wind and the Sun were disputing which was the stronger, when "
    "a traveller came along wrapped in a warm cloak. They agreed that the "
    "one who first succeeded in making the traveller take his cloak off "
    "should be considered stronger than the other."
)
GRANDFATHER_PASSAGE = (
    "You wished to know all about my grandfather. Well, he is nearly "
    "ninety-three years old. He dresses himself in an ancient black frock "
    "coat, usually minus several buttons. A long beard clings to his chin, "
    "giving those who observe him a pronounced feeling of the utmost respect."
)

# Short clinical-style word tasks. Animal- and picture-naming words are
# everyday English nouns — no copyright concerns. Counting is universal.
ANIMAL_NAMING: tuple[str, ...] = ("Cat.", "Dog.", "Elephant.", "Mouse.", "Tiger.")
PICTURE_NAMING: tuple[str, ...] = (
    "Window.", "Pumpkin.", "Carrot.", "Hammer.", "Telephone.",
)
COUNTING_SHORT = "One, two, three, four, five, six, seven, eight, nine, ten."

# DDK approximation. SpeechT5 will inject natural prosody between repetitions
# rather than the rhythmic rapid articulation a real clinical DDK elicits, so
# this clip is labeled in the manifest as ``tts_approximation: true`` — useful
# as a "repeated nonsense-syllable" stress test, NOT as a DDK-rate measurement
# substrate. A real human-recorded DDK clip can drop in later under a different
# file_id without conflicting with this entry.
DDK_PATAKA = "Pa-ta-ka, pa-ta-ka, pa-ta-ka, pa-ta-ka, pa-ta-ka, pa-ta-ka."

# PII free-response sentences. ALL identifiers below are obviously fictional
# (placeholder names, sample phone-number digits, ``example.com`` etc.). They
# produce ASR transcripts containing the named PII patterns; whether PII
# detectors trigger on the *audio* depends on how digits/email symbols are
# spoken (TTS will say "five five five" not "555", "at example dot com" not
# "@example.com"), which is itself a useful integration-test signal.


@dataclass(frozen=True)
class PIITranscript:
    """One PII free-response clip plus its labeled category."""

    file_stem: str          # used to build ``pii-test/<file_stem>.flac``
    category: str           # NAME / PHONE / EMAIL / ADDRESS / DOB / MEDICATION / MIXED / NONE
    transcript: str


PII_TRANSCRIPTS: tuple[PIITranscript, ...] = (
    PIITranscript(
        "intro-name", "NAME",
        "Hi, my name is John Smith.",
    ),
    PIITranscript(
        "contact-phone", "PHONE",
        "You can reach me at five five five, one two three, four five six seven.",
    ),
    PIITranscript(
        "email-spoken", "EMAIL",
        "My email is example dot user at example dot com.",
    ),
    PIITranscript(
        "address-spoken", "ADDRESS",
        "I live at twelve thirty-four Maple Street in Springfield.",
    ),
    PIITranscript(
        "dob-spoken", "DOB",
        "I was born on March third, nineteen eighty-five.",
    ),
    PIITranscript(
        "medication", "MEDICATION",
        "I take ibuprofen for my headaches.",
    ),
    PIITranscript(
        "free-mixed", "MIXED",
        "My name is Jane Doe, my date of birth is January twelfth nineteen ninety, "
        "and I currently live at five hundred Oak Avenue.",
    ),
    PIITranscript(
        "free-no-pii", "NONE",
        "I really enjoyed the hike this weekend. The weather was perfect and we "
        "saw a lot of wildlife.",
    ),
)


# ──────────────────────────────────────────────────────────────────────────
# FixtureRecipe — one clip to synthesize.


@dataclass
class FixtureRecipe:
    """One clip to synthesize: which voice, what to say, where to write it.

    Optional ``clinical_task`` / ``pii_category`` / ``tts_approximation``
    annotations are recorded in the manifest so consumers can filter by
    intent (e.g. "all clips labeled PII=NAME", "skip TTS-approximated DDK").
    """

    file_id: str            # used as the manifest key + filename stem
    speaker_key: str        # "A" / "B" / "C"
    transcript: str
    session_id: str | None
    clinical_task: str | None = None
    pii_category: str | None = None
    tts_approximation: bool = False
    notes: dict[str, str] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# Category recipe-builders.


def _speaker_profile_subjects() -> list[FixtureRecipe]:
    """Three confidence-tier subjects: confident, thin, insufficient."""
    recipes: list[FixtureRecipe] = []

    # Confident target subject — first 5 Harvard sentences in session 1.
    for i, sent in enumerate(HARVARD_SENTENCES[:5]):
        recipes.append(
            FixtureRecipe(
                file_id=f"sub-A-confident/ses-1/harvard-{i:02d}.flac",
                speaker_key="A",
                transcript=sent,
                session_id="ses-1",
            )
        )

    # Extra Harvard sentences on the confident subject — variety + more total
    # enrollment material above the ~30 s target.
    for i, sent in enumerate(HARVARD_SENTENCES[5:10], start=5):
        recipes.append(
            FixtureRecipe(
                file_id=f"sub-A-confident/ses-1/harvard-{i:02d}.flac",
                speaker_key="A",
                transcript=sent,
                session_id="ses-1",
            )
        )

    # Thin target subject — two short clips, just under the 20 s floor.
    for i, sent in enumerate(HARVARD_SENTENCES[5:7]):
        recipes.append(
            FixtureRecipe(
                file_id=f"sub-A-thin/ses-1/harvard-{i:02d}.flac",
                speaker_key="A",
                transcript=sent,
                session_id="ses-1",
            )
        )

    # Insufficient target subject — one very short fragment (single word).
    recipes.append(
        FixtureRecipe(
            file_id="sub-A-insufficient/ses-1/single-word.flac",
            speaker_key="A",
            transcript="Rice.",
            session_id="ses-1",
        )
    )
    return recipes


def _long_passages() -> list[FixtureRecipe]:
    """Public-domain speech-research passages on the confident subject."""
    return [
        FixtureRecipe(
            file_id="sub-A-confident/ses-1/rainbow.flac",
            speaker_key="A",
            transcript=RAINBOW_PASSAGE,
            session_id="ses-1",
            clinical_task="rainbow-passage",
        ),
        FixtureRecipe(
            file_id="sub-A-confident/ses-1/north-wind.flac",
            speaker_key="A",
            transcript=NORTH_WIND_FULL,
            session_id="ses-1",
            clinical_task="north-wind-and-the-sun",
        ),
        FixtureRecipe(
            file_id="sub-A-confident/ses-1/grandfather.flac",
            speaker_key="A",
            transcript=GRANDFATHER_PASSAGE,
            session_id="ses-1",
            clinical_task="grandfather-passage",
        ),
    ]


def _short_word_tasks() -> list[FixtureRecipe]:
    """Clinical-style brief responses: animal naming, picture naming, counting."""
    recipes: list[FixtureRecipe] = []
    for i, word in enumerate(ANIMAL_NAMING):
        recipes.append(
            FixtureRecipe(
                file_id=f"sub-A-confident/ses-1/animal-naming-{i:02d}.flac",
                speaker_key="A",
                transcript=word,
                session_id="ses-1",
                clinical_task="animal-naming",
            )
        )
    for i, word in enumerate(PICTURE_NAMING):
        recipes.append(
            FixtureRecipe(
                file_id=f"sub-A-confident/ses-1/picture-naming-{i:02d}.flac",
                speaker_key="A",
                transcript=word,
                session_id="ses-1",
                clinical_task="picture-naming",
            )
        )
    recipes.append(
        FixtureRecipe(
            file_id="sub-A-confident/ses-1/counting.flac",
            speaker_key="A",
            transcript=COUNTING_SHORT,
            session_id="ses-1",
            clinical_task="counting-1-10",
        )
    )
    return recipes


def _ddk_approx() -> list[FixtureRecipe]:
    """A single TTS-approximated DDK clip (labeled as approximation)."""
    return [
        FixtureRecipe(
            file_id="sub-A-confident/ses-1/ddk-pataka-approx.flac",
            speaker_key="A",
            transcript=DDK_PATAKA,
            session_id="ses-1",
            clinical_task="ddk-pataka",
            tts_approximation=True,
            notes={
                "caveat": (
                    "SpeechT5 reads this with natural prosody between syllables "
                    "rather than the rhythmic rapid articulation of clinical DDK; "
                    "useful as a repeated-nonsense-syllable stress test only."
                )
            },
        )
    ]


def _pii_free_response() -> list[FixtureRecipe]:
    """Free-form sentences containing fictional PII patterns.

    Kept in a separate ``pii-test/`` group (not part of any speaker-profile
    subject) so they don't bias the speaker-profile aggregates. Uses
    speaker A's voice so they're available for the future integration test
    that overlays an intruder voice on top.
    """
    return [
        FixtureRecipe(
            file_id=f"pii-test/{t.file_stem}.flac",
            speaker_key="A",
            transcript=t.transcript,
            session_id=None,
            pii_category=t.category,
        )
        for t in PII_TRANSCRIPTS
    ]


def _intruder_and_spare() -> list[FixtureRecipe]:
    """Standalone clips for speakers B and C — used by the test composers."""
    recipes: list[FixtureRecipe] = []
    # Intruder (B) — multiple clips so the composers have variety.
    for i, sent in enumerate(HARVARD_SENTENCES[7:10], start=0):
        recipes.append(
            FixtureRecipe(
                file_id=f"speaker-B/clip-{i:02d}.flac",
                speaker_key="B",
                transcript=sent,
                session_id=None,
            )
        )
    # Similar-timbre spare (C) — one clip.
    recipes.append(
        FixtureRecipe(
            file_id="speaker-C/clip-00.flac",
            speaker_key="C",
            transcript=HARVARD_SENTENCES[6],
            session_id=None,
        )
    )
    return recipes


def _build_recipes() -> list[FixtureRecipe]:
    """Concatenate all category recipe-builders into the full corpus."""
    return [
        *_speaker_profile_subjects(),
        *_long_passages(),
        *_short_word_tasks(),
        *_ddk_approx(),
        *_pii_free_response(),
        *_intruder_and_spare(),
    ]


# ──────────────────────────────────────────────────────────────────────────


def _load_speaker_embeddings(
    speakers: dict[str, tuple[str, int]],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load the picked x-vectors and verify the observed CMU-Arctic speaker_ids.

    Returns ``(embeddings, observed_speaker_ids)`` where:
      - ``embeddings`` is ``{speaker_key -> torch.Tensor of shape (1, dim)}``
      - ``observed_speaker_ids`` is ``{speaker_key -> str}`` reflecting what the
        dataset actually reports for the picked index — useful so the user can
        confirm Option-1 deliberate selection landed on the intended speakers.
    """
    import torch
    from datasets import load_dataset

    ds = load_dataset(CMU_ARCTIC_DATASET_ID, split="validation")
    embeddings: dict[str, torch.Tensor] = {}
    observed: dict[str, str] = {}
    for key, (expected_sid, idx) in speakers.items():
        row = ds[idx]
        vec = torch.tensor(row["xvector"]).unsqueeze(0)
        # ``filename`` in CMU-Arctic xvectors is typically of the form
        # ``cmu_us_<speaker>_arctic-...`` so we extract the speaker id from it.
        sid = row.get("filename", "")
        if isinstance(sid, str) and sid.startswith("cmu_us_"):
            parts = sid.split("_")
            observed_sid = parts[2] if len(parts) >= 3 else sid
        else:
            observed_sid = str(sid) if sid else "unknown"
        embeddings[key] = vec
        observed[key] = observed_sid
        marker = "OK" if observed_sid == expected_sid else "DIFFERENT"
        print(
            f"  speaker {key}: index={idx}  expected={expected_sid!r}  "
            f"observed={observed_sid!r}  [{marker}]"
        )
    return embeddings, observed


def _generate(
    recipes: Sequence[FixtureRecipe],
    out_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Synthesize each recipe and return per-clip manifest entries + observed speakers."""
    import soundfile as sf
    import torch
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    torch.manual_seed(SEED)

    processor = SpeechT5Processor.from_pretrained(SPEECHT5_TTS_ID, revision=SPEECHT5_REVISION)
    tts = SpeechT5ForTextToSpeech.from_pretrained(SPEECHT5_TTS_ID, revision=SPEECHT5_REVISION)
    vocoder = SpeechT5HifiGan.from_pretrained(SPEECHT5_HIFIGAN_ID, revision=SPEECHT5_REVISION)
    tts.eval()
    vocoder.eval()

    speaker_embeddings, observed_speakers = _load_speaker_embeddings(CMU_ARCTIC_SPEAKERS)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, object]] = []
    for r in recipes:
        inputs = processor(text=r.transcript, return_tensors="pt")
        speaker_emb = speaker_embeddings[r.speaker_key]
        with torch.inference_mode():
            speech = tts.generate_speech(
                inputs["input_ids"], speaker_emb, vocoder=vocoder
            )
        waveform = speech.cpu().numpy().astype("float32")
        # SpeechT5/HiFi-GAN output is 16 kHz mono.
        target = out_dir / r.file_id
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(target), waveform, SAMPLE_RATE_HZ, format="FLAC")
        duration_s = float(waveform.shape[0]) / SAMPLE_RATE_HZ
        entry: dict[str, object] = {
            "file_id": r.file_id,
            "speaker_id": r.speaker_key,
            "speaker_id_cmu_arctic": observed_speakers[r.speaker_key],
            "transcript": r.transcript,
            "duration_s": round(duration_s, 3),
            "session_id": r.session_id,
        }
        if r.clinical_task is not None:
            entry["clinical_task"] = r.clinical_task
        if r.pii_category is not None:
            entry["pii_category"] = r.pii_category
        if r.tts_approximation:
            entry["tts_approximation"] = True
        if r.notes:
            entry["notes"] = dict(r.notes)
        manifest_entries.append(entry)
        print(f"  wrote {r.file_id}  ({duration_s:.2f}s, speaker {r.speaker_key})")
    return manifest_entries, observed_speakers


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("src/tests/data_for_testing/synthetic"),
        help="Output directory for committed FLAC clips and manifest.json.",
    )
    args = parser.parse_args(argv)

    recipes = _build_recipes()
    by_category: dict[str, int] = {}
    for r in recipes:
        key = (
            "pii"
            if r.pii_category
            else (r.clinical_task or "speaker-profile-default")
        )
        by_category[key] = by_category.get(key, 0) + 1
    print(f"[gen] synthesizing {len(recipes)} clips with SpeechT5 (seed={SEED}).")
    print(f"[gen] by category: {dict(sorted(by_category.items()))}")
    manifest_entries, observed_speakers = _generate(recipes, args.out_dir)

    manifest_path = args.out_dir / "manifest.json"
    manifest_payload = {
        "schema_version": 1,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "seed": SEED,
        "tts_model": SPEECHT5_TTS_ID,
        "tts_revision": SPEECHT5_REVISION,
        "vocoder_model": SPEECHT5_HIFIGAN_ID,
        "speaker_xvector_dataset": CMU_ARCTIC_DATASET_ID,
        "speakers": {
            key: {
                "xvector_index": idx,
                "expected_cmu_arctic_speaker_id": expected_sid,
                "observed_cmu_arctic_speaker_id": observed_speakers.get(key, "unknown"),
            }
            for key, (expected_sid, idx) in CMU_ARCTIC_SPEAKERS.items()
        },
        "clips": manifest_entries,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    print(f"[gen] wrote manifest: {manifest_path}")

    # Surface any expected/observed mismatch as a soft warning so the user
    # knows to either accept the actual speaker or pick a different index.
    mismatched = [
        f"  {k}: expected={CMU_ARCTIC_SPEAKERS[k][0]!r}  observed={observed_speakers[k]!r}"
        for k in CMU_ARCTIC_SPEAKERS
        if observed_speakers.get(k) != CMU_ARCTIC_SPEAKERS[k][0]
    ]
    if mismatched:
        print("[gen] NOTE — observed speaker_ids differ from expected:")
        for line in mismatched:
            print(line)
        print(
            "[gen] If the actual speakers sound fine, you can update "
            "CMU_ARCTIC_SPEAKERS to match the observed ids; otherwise pick "
            "different xvector indices and regenerate."
        )
    # Keep ``asdict`` import used (silences ruff if recipes never use ``notes``).
    _ = asdict
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
