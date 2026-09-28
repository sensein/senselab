"""The Bridge2AI-Voice task registry: which text a recording's task is given, and from where."""

from __future__ import annotations

import json
from pathlib import Path

from senselab.audio.workflows.triage.b2ai_hints import build_hint
from senselab.audio.workflows.triage.task_registry import (
    CURATED,
    DESCRIPTIONS,
    REGISTRY_DIR,
    SIDECAR,
    description_key,
    registry_task,
    task_text,
)


def test_the_vendored_files_carry_their_provenance() -> None:
    """Vendored verbatim from one b2aiprep commit, named beside them."""
    provenance = (REGISTRY_DIR / "PROVENANCE.yaml").read_text()
    assert "8c43256899fbceb776e5bc536ec3b12c0d044cfa" in provenance
    for name in ("audio_task_descriptions.json", "registry.json", "task_instructions_curated.json"):
        assert (REGISTRY_DIR / name).is_file()
        assert name in provenance


def test_a_description_key_ignores_case_and_parentheses() -> None:
    """``free-speech-(v2)-1`` in the descriptions is ``free-speech-v2-1`` in BIDS."""
    assert description_key("free-speech-(v2)-1") == description_key("Free-Speech-v2-1") == "free-speech-v2-1"


def test_an_alias_two_versions_share_is_resolved_by_the_name() -> None:
    """``story-recall`` is v1's alias and v2's; the registry's own index picks v2, the recording is v1."""
    assert registry_task("story-recall") == ("adult.story-recall.v1", "")
    assert registry_task("story-recall-v2") == ("adult.story-recall.v2", "")


def test_a_per_recording_correction_is_found_by_its_recording_key() -> None:
    """The curated file keys DDK v1's corrections by syllable."""
    assert registry_task("diadochokinesis-pa") == ("adult.diadochokinesis.v1", "pa")
    text = task_text("diadochokinesis-pa", {"instructions": "harvested text"})
    assert text.instructions_source == CURATED and "/PA/" in (text.instructions or "")


def test_a_recall_task_without_a_sidecar_stimulus_takes_the_descriptions_prompts() -> None:
    """story-recall-v2's boy-and-frog story, where the sidecar leaves the stimulus empty."""
    text = task_text("story-recall-v2", {"speech_type": "recall", "stimulus_text": ""})
    assert text.stimulus_source == DESCRIPTIONS and "frog" in (text.stimulus or "")


def test_a_stimulus_empty_by_design_stays_empty() -> None:
    """A breath task's sidecar carries no stimulus, and none is borrowed for it."""
    text = task_text("respiration-and-cough-v2-breath", {"speech_type": "non-lexical", "stimulus_text": ""})
    assert text.stimulus is None and text.stimulus_source == ""


def test_the_sidecar_stimulus_wins_where_it_carries_one() -> None:
    """The recording's own stimulus is the one it was given."""
    text = task_text("story-recall", {"speech_type": "recall", "stimulus_text": "he is ninety-three"})
    assert (text.stimulus, text.stimulus_source) == ("he is ninety-three", SIDECAR)


def test_the_hint_carries_the_instructions_and_names_their_source(tmp_path: Path) -> None:
    """The hint the pipeline and the reviewer read, built from a recording's two sidecars."""
    wav = tmp_path / "sub-a_ses-b_task-story-recall.wav"
    (tmp_path / "sub-a_ses-b_task-story-recall_acoustictask-metadata.json").write_text(
        json.dumps(
            {
                "acoustic_task_name": "story-recall",
                "instructions": "You are given a text.",
                "stimulus_text": "he is nearly ninety-three years old",
                "speech_type": "recall",
                "language": "en",
            }
        )
    )
    hint, record = build_hint(wav)
    assert hint is not None
    assert hint.speech_type == "recall"
    assert hint.instructions and record["instructions_from"] == CURATED
    assert [prompt.text for prompt in hint.expected_speech] == ["he is nearly ninety-three years old"]
    assert hint.metadata["stimulus_from"] == SIDECAR
    assert hint.metadata["registry_task"] == "adult.story-recall.v1"


def test_a_recording_in_another_language_keeps_its_own_instructions() -> None:
    """r6's Spanish sessions: the curated corrections are English and would replace Spanish text."""
    spanish = "A continuación, se le presentará una historia."
    text = task_text("story-recall", {"instructions": spanish, "language": "es", "speech_type": "recall"})
    assert (text.instructions, text.instructions_source) == (spanish, SIDECAR)
    assert text.stimulus is None, "no English prompt is borrowed for a Spanish recording"
