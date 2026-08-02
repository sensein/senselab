"""Per-stage contract tests (T051 step 5).

Model-free: each test monkeypatches the ``tasks/`` call the stage dispatches to, so
this runs in CI without downloads or a GPU. What's pinned is the part that fails
*silently* if it drifts — the fragment keys that ``speech_presence.py`` / ``compute.py`` /
``global_summary.py`` / the adaptive interventions read, and the deliberate
asymmetry between what ``stage_features`` returns and what it writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.audio_analysis import stages as stages_mod
from senselab.audio.workflows.audio_analysis.stage_context import PassPlan, StageContext
from senselab.audio.workflows.audio_analysis.stages import (
    run_pass,
    stage_alignment,
    stage_asr,
    stage_diarization,
    stage_features,
    stage_ppg,
    stage_scene,
)


@pytest.fixture
def audio() -> Audio:
    """One second of quiet 16 kHz mono audio."""
    return Audio(waveform=torch.full((1, 16000), 0.05, dtype=torch.float32), sampling_rate=16000)


@pytest.fixture
def ctx(tmp_path: Path) -> StageContext:
    """A cache- and sidecar-enabled context rooted in tmp_path."""
    return StageContext(
        perturbation="raw",
        audio_signature="a" * 64,
        cache_dir=tmp_path / "cache",
        out_dir=tmp_path / "raw",
        senselab_ver="test",
    )


def _out(ctx: StageContext) -> Path:
    """Narrow `ctx.out_dir` for assertions — the fixture always sets it."""
    assert ctx.out_dir is not None
    return ctx.out_dir


# ── stage_diarization ─────────────────────────────────────────────────


def test_diarization_fragment_shape(audio: Audio, ctx: StageContext, monkeypatch: pytest.MonkeyPatch) -> None:
    """Consumers read pass_summary["diarization"]["by_model"][model_id]."""
    monkeypatch.setattr(stages_mod, "diarize_audios", lambda *a, **k: ["segments"])
    fragment = stage_diarization(audio, ctx, models=["pyannote/speaker-diarization-3.1"])
    assert set(fragment) == {"diarization"}
    outcome = fragment["diarization"]["by_model"]["pyannote/speaker-diarization-3.1"]
    assert outcome["status"] == "ok"
    assert (_out(ctx) / "diarization" / "pyannote_speaker_diarization_3_1.json").exists()


def test_diarization_with_no_models_is_empty(audio: Audio, ctx: StageContext) -> None:
    """Absence means skip — no model calls, still a well-formed fragment."""
    assert stage_diarization(audio, ctx, models=[]) == {"diarization": {"by_model": {}}}


def test_diarization_caches_between_calls(audio: Audio, ctx: StageContext, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second identical call must replay from cache without re-running."""
    calls: list[int] = []

    def _spy(*args: Any, **kwargs: Any) -> list[str]:  # noqa: ANN401 — passthrough spy
        calls.append(1)
        return ["segs"]

    monkeypatch.setattr(stages_mod, "diarize_audios", _spy)
    stage_diarization(audio, ctx, models=["pyannote/speaker-diarization-3.1"])
    second = stage_diarization(audio, ctx, models=["pyannote/speaker-diarization-3.1"])
    assert len(calls) == 1, "cache hit must not re-run the model"
    assert second["diarization"]["by_model"]["pyannote/speaker-diarization-3.1"]["cache"] == "hit"


# ── stage_scene ───────────────────────────────────────────────────────


def test_scene_skips_each_classifier_independently(
    audio: Audio, ctx: StageContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ast_model=None / yamnet_model=None each skip only their own classifier."""
    monkeypatch.setattr(stages_mod, "classify_audios", lambda *a, **k: [[{"start": 0.0, "end": 1.0}]])
    only_yamnet = stage_scene(
        audio,
        ctx,
        ast_model=None,
        yamnet_model="yamnet",
        ast_win_length=1.0,
        ast_hop_length=1.0,
        yamnet_win_length=1.0,
        yamnet_hop_length=1.0,
        top_k=5,
    )
    assert "yamnet" in only_yamnet and "ast" not in only_yamnet


def test_scene_records_the_window_grid(audio: Audio, ctx: StageContext, monkeypatch: pytest.MonkeyPatch) -> None:
    """Downstream projection needs each classifier's actual grid."""
    monkeypatch.setattr(stages_mod, "classify_audios", lambda *a, **k: [[{"start": 0.0, "end": 1.0}]])
    fragment = stage_scene(
        audio,
        ctx,
        ast_model="MIT/ast-finetuned-audioset-10-10-0.4593",
        yamnet_model=None,
        ast_win_length=10.24,
        ast_hop_length=10.24,
        yamnet_win_length=0.96,
        yamnet_hop_length=0.48,
        top_k=5,
    )
    assert fragment["ast"]["window"] == {"win_length": 10.24, "hop_length": 10.24}


def test_scene_agreement_only_on_a_shared_grid(
    audio: Audio, ctx: StageContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mismatched grids make a side-by-side comparison meaningless — so skip it."""
    monkeypatch.setattr(stages_mod, "classify_audios", lambda *a, **k: [[{"start": 0.0, "end": 1.0}]])
    kwargs: dict[str, Any] = {
        "ast_model": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "yamnet_model": "yamnet",
        "top_k": 5,
    }
    mismatched = stage_scene(
        audio, ctx, ast_win_length=10.24, ast_hop_length=10.24, yamnet_win_length=0.96, yamnet_hop_length=0.48, **kwargs
    )
    assert "scene_agreement" not in mismatched


# ── stage_features (the silent-failure case) ──────────────────────────


def test_features_returns_live_rows_but_writes_a_placeholder(
    audio: Audio, ctx: StageContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The returned result must be the real dict; only the JSON gets a placeholder.

    speech_presence.py / compute.py / global_summary.py read
    pass_summary["features"]["result"] as {backend: rows}. Returning the sidecar
    shape instead would leave every loudness/quality column None rather than
    raising — the exact failure this pins.
    """
    import json

    rows = {"opensmile": [{"start": 0.0, "end": 0.01, "Loudness_sma3": 0.5}]}
    monkeypatch.setattr(stages_mod, "extract_temporal_features", lambda *a, **k: rows)
    fragment = stage_features(audio, ctx, win_length=1.0, hop_length=0.5)

    assert isinstance(fragment["features"]["result"], dict), "in-memory result must stay live"
    assert fragment["features"]["result"]["opensmile"][0]["Loudness_sma3"] == 0.5

    written = json.loads((_out(ctx) / "features.json").read_text())
    assert written["result"] == "see features/*.parquet", "sidecar must not duplicate the rows"


def test_features_writes_one_parquet_per_backend(
    audio: Audio, ctx: StageContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backends have different columns and grids, so they can't share a schema."""
    pytest.importorskip("pandas")
    rows = {
        "opensmile": [{"start": 0.0, "end": 0.01, "a": 1.0}],
        "parselmouth": [{"start": 0.0, "end": 1.0, "b": 2.0}],
        "torchaudio_squim": [],
    }
    monkeypatch.setattr(stages_mod, "extract_temporal_features", lambda *a, **k: rows)
    stage_features(audio, ctx, win_length=1.0, hop_length=0.5)
    assert (_out(ctx) / "features" / "opensmile.parquet").exists()
    assert (_out(ctx) / "features" / "parselmouth.parquet").exists()
    assert not (_out(ctx) / "features" / "torchaudio_squim.parquet").exists(), "empty backend writes nothing"


# ── stage_asr ─────────────────────────────────────────────────────────


def test_asr_fragment_shape(audio: Audio, ctx: StageContext, monkeypatch: pytest.MonkeyPatch) -> None:
    """Consumers read pass_summary["asr"]["by_model"]."""
    monkeypatch.setattr(stages_mod, "transcribe_audios", lambda *a, **k: ["hello"])
    fragment = stage_asr(audio, ctx, models=["openai/whisper-tiny"])
    assert fragment["asr"]["by_model"]["openai/whisper-tiny"]["result"] == ["hello"]


def test_asr_qwen_timestamp_optout_reaches_the_backend(
    audio: Audio, ctx: StageContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabling Qwen's bundled aligner must actually pass return_timestamps=False."""
    seen: dict[str, Any] = {}

    def _fake(*args: Any, **kwargs: Any) -> list[str]:  # noqa: ANN401 — passthrough spy
        seen.update(kwargs)
        return ["x"]

    monkeypatch.setattr(stages_mod, "transcribe_audios", _fake)
    stage_asr(audio, ctx, models=["Qwen/Qwen3-ASR-1.7B"], qwen_native_timestamps=False)
    assert seen.get("return_timestamps") is False


def test_asr_optout_does_not_affect_other_backends(
    audio: Audio, ctx: StageContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The opt-out is Qwen-specific; Whisper must not receive it."""
    seen: dict[str, Any] = {}

    def _fake(*args: Any, **kwargs: Any) -> list[str]:  # noqa: ANN401 — passthrough spy
        seen.update(kwargs)
        return ["x"]

    monkeypatch.setattr(stages_mod, "transcribe_audios", _fake)
    stage_asr(audio, ctx, models=["openai/whisper-tiny"], qwen_native_timestamps=False)
    assert "return_timestamps" not in seen


# ── stage_alignment ───────────────────────────────────────────────────


def _asr_block(text: str, *, timestamped: bool) -> dict[str, Any]:
    from senselab.utils.data_structures import ScriptLine

    line = (
        ScriptLine(text=text, start=0.0, end=1.0, chunks=[ScriptLine(text=text, start=0.0, end=1.0)])
        if timestamped
        else ScriptLine(text=text)
    )
    return {"status": "ok", "result": [line], "cache_key": "parent_asr_key"}


def test_alignment_skips_natively_timestamped_asr(
    audio: Audio, ctx: StageContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aligning an already-timestamped transcript would be a wasteful no-op."""
    monkeypatch.setattr(stages_mod.QwenASR, "align_with_qwen", staticmethod(lambda *a, **k: ["aligned"]))
    fragment = stage_alignment(audio, ctx, asr_by_model={"whisper": _asr_block("hi", timestamped=True)})
    assert fragment["alignment"]["by_model"] == {}


def test_alignment_runs_for_text_only_asr(audio: Audio, ctx: StageContext, monkeypatch: pytest.MonkeyPatch) -> None:
    """Text-only backends (Granite, Canary) get per-word timestamps here."""
    monkeypatch.setattr(stages_mod.QwenASR, "align_with_qwen", staticmethod(lambda *a, **k: ["aligned"]))
    fragment = stage_alignment(audio, ctx, asr_by_model={"granite": _asr_block("hello world", timestamped=False)})
    assert fragment["alignment"]["by_model"]["granite"]["result"] == ["aligned"]


def test_alignment_records_the_parent_asr_cache_key(
    audio: Audio, ctx: StageContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adaptive loop follows this link back to the transcript that was aligned."""
    monkeypatch.setattr(stages_mod.QwenASR, "align_with_qwen", staticmethod(lambda *a, **k: ["aligned"]))
    fragment = stage_alignment(audio, ctx, asr_by_model={"granite": _asr_block("hello", timestamped=False)})
    prov = fragment["alignment"]["by_model"]["granite"]["provenance"]
    assert prov["parent_asr_cache_key"] == "parent_asr_key"
    assert prov["transcript_sha"]


def test_alignment_takes_asr_explicitly_not_from_a_shared_dict(
    audio: Audio, ctx: StageContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller can align a cached ASR block it never produced.

    This is the point of passing asr_by_model as a parameter: the adaptive loop's
    escalation path aligns transcripts recovered from cache, with no preceding
    stage_asr call in the same process.
    """
    monkeypatch.setattr(stages_mod.QwenASR, "align_with_qwen", staticmethod(lambda *a, **k: ["aligned"]))
    recovered = {"from-cache": _asr_block("recovered text", timestamped=False)}
    fragment = stage_alignment(audio, ctx, asr_by_model=recovered)
    assert "from-cache" in fragment["alignment"]["by_model"]


def test_alignment_skips_empty_transcripts(audio: Audio, ctx: StageContext, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing to align — and an empty transcript must not poison the cache."""
    monkeypatch.setattr(stages_mod.QwenASR, "align_with_qwen", staticmethod(lambda *a, **k: ["aligned"]))
    fragment = stage_alignment(audio, ctx, asr_by_model={"granite": _asr_block("", timestamped=False)})
    assert fragment["alignment"]["by_model"] == {}


def test_alignment_skips_failed_asr(audio: Audio, ctx: StageContext) -> None:
    """A failed ASR has no transcript to align."""
    fragment = stage_alignment(audio, ctx, asr_by_model={"m": {"status": "failed", "error": "x"}})
    assert fragment["alignment"]["by_model"] == {}


# ── stage_ppg ─────────────────────────────────────────────────────────


def test_ppg_uses_the_plural_key(audio: Audio, ctx: StageContext, monkeypatch: pytest.MonkeyPatch) -> None:
    """Consumers accept "ppg" and "ppgs", so a rename degrades silently — pin it."""
    import numpy as np

    monkeypatch.setattr(
        "senselab.audio.tasks.features_extraction.ppg.extract_ppgs_from_audios",
        lambda *a, **k: [np.zeros((40, 10), dtype="float32")],
    )
    fragment = stage_ppg(audio, ctx)
    assert set(fragment) == {"ppgs"}, "the key must be plural"
    assert fragment["ppgs"]["phoneme_labels"], "inventory must ride along for the harvester"


# ── run_pass ──────────────────────────────────────────────────────────


def test_run_pass_emits_the_summary_envelope(audio: Audio, ctx: StageContext) -> None:
    """Label / duration_s / audio_signature are what adaptive/loop.py reads."""
    summary = run_pass(audio, ctx, PassPlan())
    assert summary["label"] == "raw"
    assert summary["duration_s"] == pytest.approx(1.0)
    assert summary["audio_signature"] == ctx.audio_signature


def test_run_pass_with_an_empty_plan_runs_no_models(audio: Audio, ctx: StageContext) -> None:
    """The no-speech triage path yields an empty plan; it must stay cheap.

    If a default ever flipped to "run everything", a silent clip would burn the
    full model suite — so assert no expensive fragment appears.
    """
    summary = run_pass(audio, ctx, PassPlan())
    for expensive in ("diarization", "asr", "alignment", "features", "ppgs", "ast", "yamnet"):
        assert expensive not in summary, f"empty plan should not have run {expensive}"


def test_run_pass_threads_asr_output_into_alignment(
    audio: Audio, ctx: StageContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one load-bearing ordering dependency: alignment consumes stage_asr."""
    from senselab.utils.data_structures import ScriptLine

    monkeypatch.setattr(stages_mod, "transcribe_audios", lambda *a, **k: [ScriptLine(text="hello world")])
    monkeypatch.setattr(stages_mod.QwenASR, "align_with_qwen", staticmethod(lambda *a, **k: ["aligned"]))
    plan = PassPlan(asr_models=("ibm-granite/granite-speech-3.3-8b",), align_asr=True)
    summary = run_pass(audio, ctx, plan)
    assert summary["alignment"]["by_model"]["ibm-granite/granite-speech-3.3-8b"]["result"] == ["aligned"]


def test_run_pass_honors_align_asr_false(audio: Audio, ctx: StageContext, monkeypatch: pytest.MonkeyPatch) -> None:
    """--no-align-asr must skip the stage entirely, not just drop its output."""
    from senselab.utils.data_structures import ScriptLine

    monkeypatch.setattr(stages_mod, "transcribe_audios", lambda *a, **k: [ScriptLine(text="hello")])
    summary = run_pass(audio, ctx, PassPlan(asr_models=("ibm-granite/granite-speech-3.3-8b",), align_asr=False))
    assert "alignment" not in summary
