"""Declared capabilities for the diarization backends."""

import dataclasses
from pathlib import Path

import pytest

from senselab.audio.tasks.speaker_diarization.api import ROLE_LABEL_ONLY_PREFIXES, capabilities_for
from senselab.audio.tasks.speaker_diarization.capabilities import DiarizationCapabilities


def test_record_is_frozen() -> None:
    """A capability record is a declaration, not mutable state.

    If a caller could mutate one, two callers would disagree about what a backend
    can do, and the disagreement would depend on import order.
    """
    caps = DiarizationCapabilities(
        populates_text=False,
        speaker_label_kind="identity",
        labels_stable_across_files=False,
        max_speakers=None,
        honors_speaker_hints=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        caps.max_speakers = 4  # type: ignore[misc]


def test_max_speakers_none_is_allowed_and_means_unmeasured() -> None:
    """None is 'nobody has measured this', not 'unlimited'.

    Four of six backends have no published or measured ceiling. Encoding that as
    None keeps it distinguishable from a real limit, so the NeMo probe can fill it
    in later without anyone having guessed in the meantime.
    """
    caps = DiarizationCapabilities(
        populates_text=False,
        speaker_label_kind="identity",
        labels_stable_across_files=False,
        max_speakers=None,
        honors_speaker_hints=False,
    )
    assert caps.max_speakers is None


def test_max_speakers_must_be_at_least_one_when_given() -> None:
    """A ceiling of zero or less describes nothing that can diarize.

    Catches a typo or an off-by-one in a declaration at construction time rather
    than as a confusing empty result much later.
    """
    with pytest.raises(ValueError, match="max_speakers"):
        DiarizationCapabilities(
            populates_text=False,
            speaker_label_kind="identity",
            labels_stable_across_files=False,
            max_speakers=0,
            honors_speaker_hints=False,
        )


def test_speaker_label_kind_rejects_an_unknown_value() -> None:
    """Only 'identity' and 'role' are meaningful.

    The distinction decides whether labels may reach embedding clustering, so a
    third value silently defaulting to one branch would be a correctness bug.
    """
    with pytest.raises(ValueError, match="speaker_label_kind"):
        DiarizationCapabilities(
            populates_text=False,
            speaker_label_kind="cluster",  # type: ignore[arg-type]
            labels_stable_across_files=False,
            max_speakers=None,
            honors_speaker_hints=False,
        )


_ALL_BACKEND_IDS = (
    "pyannote/speaker-diarization-community-1",
    "nvidia/diar_sortformer_4spk-v1",
    "microsoft/VibeVoice-ASR-HF",
    "AlexXu811/whisper-child-adult",
    "OpenMOSS-Team/MOSS-Transcribe-Diarize",
    "BUT-FIT/diarizen-wavlm-large-s80-md",
)


@pytest.mark.parametrize("model_id", _ALL_BACKEND_IDS)
def test_every_dispatchable_backend_declares_capabilities(model_id: str) -> None:
    """A backend reachable from diarize_audios must say what it provides.

    This is the test that stops a seventh backend being added without declaring
    itself, which is how the current situation arose: six backends, no declarations,
    and the only way to learn the differences was to run each one.
    """
    caps = capabilities_for(model_id)
    assert isinstance(caps, DiarizationCapabilities)


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("microsoft/VibeVoice-ASR-HF", True),
        ("OpenMOSS-Team/MOSS-Transcribe-Diarize", True),
        ("BUT-FIT/diarizen-wavlm-large-s80-md", False),
        ("AlexXu811/whisper-child-adult", False),
        ("pyannote/speaker-diarization-community-1", False),
        ("nvidia/diar_sortformer_4spk-v1", False),
    ],
)
def test_populates_text_matches_what_was_measured(model_id: str, expected: bool) -> None:
    """Exactly the two joint ASR+diarization backends fill `text`.

    Measured on an H100: VibeVoice returned 7 segments all carrying text, MOSS 6,
    while DiariZen (10) and child-adult (19) returned none. A consumer reading
    text=None otherwise cannot tell a backend limitation from an empty segment.
    """
    assert capabilities_for(model_id).populates_text is expected


def test_child_adult_is_a_two_speaker_role_classifier() -> None:
    """Count and kind are separate facts about the same backend.

    child-adult can only ever emit CHILD/ADULT, making it a 2-speaker diarizer by
    count. But its labels denote roles, which is what decides they must not reach
    embedding clustering. A 2-speaker identity diarizer would share the ceiling and
    need different handling, so one field cannot carry both.
    """
    caps = capabilities_for("AlexXu811/whisper-child-adult")
    assert caps.max_speakers == 2
    assert caps.speaker_label_kind == "role"


def test_sortformer_declares_the_ceiling_in_its_own_name() -> None:
    """`diar_sortformer_4spk` tops out at four."""
    assert capabilities_for("nvidia/diar_sortformer_4spk-v1").max_speakers == 4


@pytest.mark.parametrize(
    "model_id",
    [
        "pyannote/speaker-diarization-community-1",
        "microsoft/VibeVoice-ASR-HF",
        "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        "BUT-FIT/diarizen-wavlm-large-s80-md",
    ],
)
def test_unmeasured_ceilings_are_none_not_guessed(model_id: str) -> None:
    """Four backends have no measured ceiling, so they declare None.

    None means unmeasured. The NeMo synthetic-speaker probe fills these in with a
    number that carries its measurement; a value copied from a model card would be
    exactly the unfitted literal this repo's conventions warn against.
    """
    assert capabilities_for(model_id).max_speakers is None


def test_only_pyannote_honors_speaker_hints() -> None:
    """Five of six ignore num_speakers, and api.py already warns when they do.

    Declaring it lets a caller avoid passing a hint that will be dropped, rather
    than discovering it in a log line after the run.
    """
    assert capabilities_for("pyannote/speaker-diarization-community-1").honors_speaker_hints is True
    for model_id in _ALL_BACKEND_IDS:
        if model_id.startswith("pyannote/"):
            continue
        assert capabilities_for(model_id).honors_speaker_hints is False


def test_diarizen_labels_are_not_stable_across_files() -> None:
    """VBx clusters per audio, so a label means nothing outside its own file.

    Measured: the same run produced ['1','2'] for one recording and
    ['0','0','1','0'] for another. A consumer joining on label across files would
    silently merge unrelated speakers.
    """
    assert capabilities_for("BUT-FIT/diarizen-wavlm-large-s80-md").labels_stable_across_files is False


def test_role_kind_agrees_with_the_existing_prefix_list() -> None:
    """The new declaration and the old ROLE_LABEL_ONLY_PREFIXES must not diverge.

    Both encode "these labels are roles, keep them out of the identity axis". While
    both exist, a backend appearing in one and not the other is a latent bug — the
    audio_analysis guards read the prefix list, and future code will read the record.
    """
    for model_id in _ALL_BACKEND_IDS:
        in_prefix_list = any(model_id.startswith(p) for p in ROLE_LABEL_ONLY_PREFIXES)
        is_role = capabilities_for(model_id).speaker_label_kind == "role"
        assert in_prefix_list == is_role, f"{model_id}: prefix list says {in_prefix_list}, record says {is_role}"


def test_an_unknown_model_id_falls_back_like_the_dispatch_does() -> None:
    """An unmatched id resolves to Pyannote, mirroring diarize_audios' own fallback.

    Returning None instead would make every caller write the same None-check for a
    case the dispatch itself treats as ordinary.
    """
    assert capabilities_for("some/unknown-diarizer").honors_speaker_hints is True


def test_registry_capabilities_match_the_code() -> None:
    """The YAML and the backend declarations must agree, on every field, for every entry.

    The registry is what a human reads when choosing a model; the code is what runs.
    Two sources of truth are acceptable here only because this test makes drift a
    test failure rather than a surprise — which requires the set of entries under
    test to come from the registry's own task label, not from "has a capabilities
    key already", or a backend added without one would simply never be visited. It
    also requires comparing the whole record, not a chosen subset of fields, or a
    field outside that subset could drift unnoticed.
    """
    import yaml

    registry = yaml.safe_load((Path(__file__).parents[3] / "senselab" / "model_registry.yaml").read_text())

    diarization_entries = [entry for entry in registry if entry.get("task") == "speaker_diarization"]
    # A sanity floor, not a magic total: the count itself is never asserted below,
    # since every entry in this list is required to declare capabilities and is
    # checked in the loop — a hard-coded total would only duplicate that. This just
    # guards against the list being silently empty (e.g. the task label was renamed).
    assert diarization_entries, "expected at least one speaker_diarization entry in the registry"

    for entry in diarization_entries:
        model_id = entry.get("model_id")
        assert "capabilities" in entry, (
            f"{model_id}: every speaker_diarization registry entry must declare capabilities"
        )
        caps = capabilities_for(model_id)
        assert entry["capabilities"] == dataclasses.asdict(caps), (
            f"{model_id}: registry capabilities do not match capabilities_for()"
        )
