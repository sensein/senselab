"""PII detection run directly on an Audio object."""

from unittest.mock import patch

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.pii_detection import detect_pii_in_audios
from senselab.utils.data_structures import ScriptLine


def _stub_hf_model_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid Hub validation when the default ``HFModel`` gets constructed.

    ``detect_pii_in_audios`` falls back to ``HFModel(path_or_uri="openai/whisper-tiny")``
    whenever the caller omits ``asr_model``. The real validator does a network round trip
    (and ``_resolve_commit_sha`` a second one) on every construction, which would make
    these tests flaky/slow for no reason -- they only exercise the transcribe-then-detect
    composition, not HF Hub connectivity.
    """
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "f" * 40)


def test_detect_pii_in_audios_transcribes_then_detects(
    mono_audio_sample: Audio, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Audio entry point is a two-step composition, not a new engine.

    It must pass the transcript through unchanged -- a transcription bug and a
    detection bug should stay distinguishable.
    """
    _stub_hf_model_validation(monkeypatch)
    with patch(
        "senselab.audio.tasks.pii_detection.api.transcribe_audios",
        return_value=[ScriptLine(text="my name is Jane Doe")],
    ) as mock_asr:
        reports = detect_pii_in_audios([mono_audio_sample], detectors=[])
    mock_asr.assert_called_once()
    assert len(reports) == 1
    assert "pii_disabled" in reports[0].failures


def test_one_report_per_audio(mono_audio_sample: Audio, monkeypatch: pytest.MonkeyPatch) -> None:
    """One ``PiiReport`` per input audio, in order."""
    _stub_hf_model_validation(monkeypatch)
    with patch(
        "senselab.audio.tasks.pii_detection.api.transcribe_audios",
        return_value=[ScriptLine(text="a"), ScriptLine(text="b")],
    ):
        reports = detect_pii_in_audios([mono_audio_sample, mono_audio_sample], detectors=[])
    assert len(reports) == 2


def test_empty_transcript_reports_no_detector_ran_not_a_clean_bill_of_health(
    mono_audio_sample: Audio, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An audio that transcribes to nothing must not read as "scanned, found nothing".

    ``detect_pii`` already treats an empty flattened transcript as "never reached a
    detector" (``detector_used=None``, ``detection_confidence=None``) rather than a
    confident ``0.0``. This locks in that ``detect_pii_in_audios`` doesn't get in the
    way of that honesty property -- e.g. by joining an all-empty transcript into
    something that no longer looks empty by the time it reaches ``detect_pii``.
    """
    _stub_hf_model_validation(monkeypatch)
    with patch(
        "senselab.audio.tasks.pii_detection.api.transcribe_audios",
        # A diarization-only line: speaker label, no text -- flattens to "".
        return_value=[ScriptLine(speaker="spk1")],
    ):
        reports = detect_pii_in_audios([mono_audio_sample])
    assert len(reports) == 1
    assert reports[0].detector_used is None
    assert reports[0].detection_confidence is None
