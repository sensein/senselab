"""Declared capabilities for the diarization backends."""

import dataclasses
from pathlib import Path

import pytest

from senselab.audio.tasks.speaker_diarization.api import ROLE_LABEL_ONLY_PREFIXES, capabilities_for
from senselab.audio.tasks.speaker_diarization.capabilities import (
    UNMEASURED,
    DiarizationCapabilities,
)


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
        max_speakers_evidence=UNMEASURED,
        honors_speaker_hints=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        caps.max_speakers = 4  # type: ignore[misc]


def test_max_speakers_none_means_no_structural_ceiling_observed() -> None:
    """None means 'no structural ceiling was found', which is not the same as unlimited.

    CONTRACT CHANGED: this test used to be named
    `test_max_speakers_none_is_allowed_and_means_unmeasured` and its docstring claimed
    None always means "nobody has measured this". The seed-17 speaker-ceiling probe
    measured all six backends and left four of them at None anyway -- Pyannote,
    VibeVoice, MOSS and DiariZen never plateaued across k=1..8, so their structural
    ceiling is genuinely unbounded within what was tested, not merely unlooked-at.
    `max_speakers=None` alone can no longer tell those two situations apart; that is
    exactly why `max_speakers_evidence` exists (see the tests below), and this test now
    only asserts the part of the old claim that still holds: None never means unlimited.
    """
    caps = DiarizationCapabilities(
        populates_text=False,
        speaker_label_kind="identity",
        labels_stable_across_files=False,
        max_speakers=None,
        max_speakers_evidence="measured: no saturation, emits up to 8 (probe seed-17)",
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
            max_speakers_evidence="measured: saturates at 0 on 20/20 k=8 sessions (probe seed-17)",
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
            max_speakers_evidence=UNMEASURED,
            honors_speaker_hints=False,
        )


def test_max_speakers_evidence_must_be_unmeasured_or_start_with_measured() -> None:
    """The two-state prefix convention is what makes the field machine-readable.

    A third spelling (a typo, or free prose with neither prefix) would defeat the
    point: code checking `evidence == UNMEASURED` vs. `.startswith("measured:")`
    would silently mis-sort it into neither bucket instead of raising here.
    """
    with pytest.raises(ValueError, match="max_speakers_evidence"):
        DiarizationCapabilities(
            populates_text=False,
            speaker_label_kind="identity",
            labels_stable_across_files=False,
            max_speakers=None,
            max_speakers_evidence="nobody checked",
            honors_speaker_hints=False,
        )


def test_max_speakers_evidence_cannot_claim_unmeasured_for_a_declared_number() -> None:
    """A number with no measurement behind it is exactly the unfitted literal this repo warns about.

    Declaring `max_speakers=4` while `max_speakers_evidence="unmeasured"` would let a
    guess masquerade as a measured value with nothing to catch it; construction must
    refuse instead.
    """
    with pytest.raises(ValueError, match="max_speakers_evidence"):
        DiarizationCapabilities(
            populates_text=False,
            speaker_label_kind="identity",
            labels_stable_across_files=False,
            max_speakers=4,
            max_speakers_evidence=UNMEASURED,
            honors_speaker_hints=False,
        )


def test_unmeasured_is_allowed_when_max_speakers_is_none() -> None:
    """The genuinely-unmeasured state -- no probe has ever run -- is still representable.

    Distinct from `max_speakers=None` paired with a "measured: no saturation..."
    evidence string: both currently read `None`, but only one of them means "nobody
    has looked yet". This is the case none of the six real backends are in anymore,
    but the mechanism must still accept it for a future seventh backend.
    """
    caps = DiarizationCapabilities(
        populates_text=False,
        speaker_label_kind="identity",
        labels_stable_across_files=False,
        max_speakers=None,
        max_speakers_evidence=UNMEASURED,
        honors_speaker_hints=False,
    )
    assert caps.max_speakers_evidence == UNMEASURED


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
    """`capabilities_for` returns a real record, not None, for every known backend id.

    This does NOT prove a seventh backend can't be added without declaring itself:
    `capabilities_for` falls back to Pyannote's record for any unmatched id, so this
    assertion is true for *every* string, known or not. It only guards the shape of
    the return value (a `DiarizationCapabilities`, never `None`) for the six ids this
    repo currently knows about. The actual "can't add a backend silently" guarantee
    is `test_every_dispatch_prefix_has_a_capability_record` below, which inspects the
    dispatch tables themselves rather than probing pre-known ids.
    """
    caps = capabilities_for(model_id)
    assert isinstance(caps, DiarizationCapabilities)


def test_every_dispatch_prefix_has_a_capability_record() -> None:
    """Every prefix `diarize_audios` dispatches on must have a capabilities entry.

    `test_every_dispatchable_backend_declares_capabilities` above cannot catch a
    seventh backend added to `diarize_audios` with no entry in
    `_CAPABILITIES_BY_PREFIX`: such a backend would silently report Pyannote's
    record via the fallback, including `honors_speaker_hints=True`, which is wrong
    for every backend but Pyannote. This test reads the dispatch tables directly
    (any module-level name ending `_PREFIXES`, excluding `ROLE_LABEL_ONLY_PREFIXES`,
    which is a derived cross-reference rather than a dispatch table) so a new prefix
    table with no matching capabilities entry fails here instead of passing silently
    through the fallback.
    """
    import senselab.audio.tasks.speaker_diarization.api as api

    mapped = {p for prefixes, _ in api._CAPABILITIES_BY_PREFIX for p in prefixes}
    for name, value in vars(api).items():
        if name.endswith("_PREFIXES") and name != "ROLE_LABEL_ONLY_PREFIXES":
            for prefix in value:
                assert prefix in mapped, f"{name} dispatches but declares no capabilities"


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

    The `2` used to be a structural claim from the model's architecture alone; the
    seed-17 speaker-ceiling probe has since confirmed it directly (20/20 k=8 sessions
    counted exactly 2), which `max_speakers_evidence` now records.
    """
    caps = capabilities_for("AlexXu811/whisper-child-adult")
    assert caps.max_speakers == 2
    assert caps.speaker_label_kind == "role"
    assert caps.max_speakers_evidence == "measured: saturates at 2 on 20/20 k=8 sessions (probe seed-17)"


def test_sortformer_declares_the_ceiling_in_its_own_name() -> None:
    """`diar_sortformer_4spk` tops out at four.

    The checkpoint name was the original source of the `4`; the seed-17 probe has
    since confirmed it structurally (20/20 k=8 sessions predicted exactly 4), which
    `max_speakers_evidence` now records rather than leaving the name as the only trace
    of where the number came from.
    """
    caps = capabilities_for("nvidia/diar_sortformer_4spk-v1")
    assert caps.max_speakers == 4
    assert caps.max_speakers_evidence == "measured: saturates at 4 on 20/20 k=8 sessions (probe seed-17)"


@pytest.mark.parametrize(
    ("model_id", "highest_observed"),
    [
        ("pyannote/speaker-diarization-community-1", 8),
        ("microsoft/VibeVoice-ASR-HF", 16),
        ("OpenMOSS-Team/MOSS-Transcribe-Diarize", 12),
        ("BUT-FIT/diarizen-wavlm-large-s80-md", 8),
    ],
)
def test_unmeasured_ceilings_are_none_not_guessed(model_id: str, highest_observed: int) -> None:
    """Four backends still declare `max_speakers=None` -- but not because nobody measured them.

    CONTRACT CHANGED: this test's name and docstring predate the seed-17 speaker-ceiling
    probe and originally meant "None means unmeasured; the probe will fill these in with
    a number". The probe ran, and for these four the answer was not a number: none of
    them plateaued at a fixed output across k=1..8, so `None` here now means "measured,
    and no structural ceiling was found" -- a different, and stronger, claim than "nobody
    looked". `max_speakers=None` alone cannot tell the two apart (see
    `capabilities.py`'s `UNMEASURED` docstring); `max_speakers_evidence` is the field
    that can, and this test now asserts that too rather than treating `None` as
    self-explanatory the way the original version did.
    """
    caps = capabilities_for(model_id)
    assert caps.max_speakers is None
    assert caps.max_speakers_evidence == f"measured: no saturation, emits up to {highest_observed} (probe seed-17)"


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

    import senselab.audio.tasks.speaker_diarization.api as api

    registry = yaml.safe_load((Path(__file__).parents[3] / "senselab" / "model_registry.yaml").read_text())

    diarization_entries = [entry for entry in registry if entry.get("task") == "speaker_diarization"]
    # Derived, not hard-coded: one registry entry per dispatch-prefix table, plus one
    # for Pyannote, which has no prefix entry of its own and is reached only via
    # `capabilities_for`'s fallback (and `diarize_audios`'s `isinstance` check). A
    # literal `6` here would need updating by hand every time a backend is added —
    # exactly the kind of drift this suite exists to catch instead of require.
    expected_count = len(api._CAPABILITIES_BY_PREFIX) + 1
    assert len(diarization_entries) == expected_count, (
        f"expected {expected_count} speaker_diarization registry entries "
        f"({len(api._CAPABILITIES_BY_PREFIX)} dispatch prefixes + 1 Pyannote fallback), "
        f"got {len(diarization_entries)}"
    )

    for entry in diarization_entries:
        model_id = entry.get("model_id")
        assert "capabilities" in entry, (
            f"{model_id}: every speaker_diarization registry entry must declare capabilities"
        )
        caps = capabilities_for(model_id)
        assert entry["capabilities"] == dataclasses.asdict(caps), (
            f"{model_id}: registry capabilities do not match capabilities_for()"
        )
