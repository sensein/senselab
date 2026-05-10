"""Smoke tests for scripts/analyze_audio.py.

These tests exercise the script's pure-Python helpers (argparse, audio
signature stability, cache key composition, auto-align skip-condition
detection, LS-export label collection) WITHOUT loading any senselab models.
They run in the default CI install path; nothing here is guarded by
`@pytest.mark.skipif`.

The expensive end-to-end paths (model loads, subprocess venv provisioning,
real LS import) are validated by the per-phase manual validation tasks
documented in artifacts/.../validation.md, not here.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "analyze_audio.py"


def _load_analyze_audio_module() -> types.ModuleType:
    """Import scripts/analyze_audio.py as a module so its helpers can be tested directly."""
    spec = importlib.util.spec_from_file_location("analyze_audio_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None, f"could not load {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_audio_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def aa() -> types.ModuleType:
    """The analyze_audio module loaded once per test session."""
    return _load_analyze_audio_module()


def test_parse_args_default_invocation(aa: types.ModuleType) -> None:
    """The argparse layer accepts a bare positional path and fills sensible defaults."""
    args = aa.parse_args(["/tmp/dummy.wav"])
    assert str(args.audio).endswith("dummy.wav")
    assert args.device == "auto"
    assert args.no_enhancement is False
    assert args.no_cache is False
    assert args.no_align_asr is False
    assert args.aligner_model == "facebook/mms-1b-all"
    assert args.asr_language is None
    # Default model lists per spec FR-005 and contracts/cli.md
    assert "openai/whisper-large-v3-turbo" in args.asr_models
    assert "ibm-granite/granite-speech-3.3-8b" in args.asr_models
    assert "nvidia/canary-qwen-2.5b" in args.asr_models
    assert "Qwen/Qwen3-ASR-1.7B" in args.asr_models
    # Native temporal precision per scene-classification model (FR-008)
    assert args.ast_win_length == 10.24
    assert args.ast_hop_length == 10.24
    assert args.yamnet_win_length == 0.96
    assert args.yamnet_hop_length == 0.48
    # Alignment is one of the skippable tasks
    assert "alignment" in aa.ALL_TASKS


def test_parse_args_skip_choices(aa: types.ModuleType) -> None:
    """The --skip flag accepts the documented task names, including the new 'alignment'."""
    args = aa.parse_args(["/tmp/dummy.wav", "--skip", "alignment", "asr"])
    assert "alignment" in args.skip
    assert "asr" in args.skip


def test_audio_signature_is_stable(aa: types.ModuleType) -> None:
    """Identical Audio objects produce identical signatures (FR-010)."""
    audio_a = SimpleNamespace(
        waveform=torch.zeros((1, 16000), dtype=torch.float32),
        sampling_rate=16000,
    )
    audio_b = SimpleNamespace(
        waveform=torch.zeros((1, 16000), dtype=torch.float32),
        sampling_rate=16000,
    )
    assert aa.audio_signature(audio_a) == aa.audio_signature(audio_b)


def test_audio_signature_changes_with_content(aa: types.ModuleType) -> None:
    """Different waveforms produce different signatures."""
    audio_a = SimpleNamespace(
        waveform=torch.zeros((1, 16000), dtype=torch.float32),
        sampling_rate=16000,
    )
    audio_b = SimpleNamespace(
        waveform=torch.ones((1, 16000), dtype=torch.float32),
        sampling_rate=16000,
    )
    assert aa.audio_signature(audio_a) != aa.audio_signature(audio_b)


def test_audio_signature_changes_with_sampling_rate(aa: types.ModuleType) -> None:
    """Same PCM bytes but different sampling rate -> different signature."""
    pcm = torch.zeros((1, 16000), dtype=torch.float32)
    audio_16k = SimpleNamespace(waveform=pcm, sampling_rate=16000)
    audio_8k = SimpleNamespace(waveform=pcm, sampling_rate=8000)
    assert aa.audio_signature(audio_16k) != aa.audio_signature(audio_8k)


def test_cache_key_changes_when_any_input_changes(aa: types.ModuleType) -> None:
    """Per FR-010: every component of the cache key matters."""
    base = dict(
        audio_sig="a" * 64,
        task="asr",
        model_id="openai/whisper-tiny",
        params={"device": "auto"},
        wrapper_hash="b" * 64,
        senselab_ver="1.3.1-alpha.27",
    )
    base_key = aa.cache_key(**base)

    # Each tweak must produce a different key
    for field, override in [
        ("audio_sig", "z" * 64),
        ("task", "embeddings"),
        ("model_id", "openai/whisper-base"),
        ("params", {"device": "cuda"}),
        ("wrapper_hash", "c" * 64),
        ("senselab_ver", "1.3.1-alpha.28"),
    ]:
        modified = base | {field: override}
        assert aa.cache_key(**modified) != base_key, f"changing {field} did not invalidate cache key"


def test_align_cache_key_is_independent_from_asr_cache_key(aa: types.ModuleType) -> None:
    """Per FR-024: ASR and alignment cache keys must diverge by construction."""
    asr_k = aa.cache_key(
        audio_sig="a" * 64,
        task="asr",
        model_id="ibm-granite/granite-speech-3.3-8b",
        params={"device": "auto"},
        wrapper_hash="b" * 64,
        senselab_ver="1.3.1-alpha.27",
    )
    align_k = aa.align_cache_key(
        audio_sig="a" * 64,
        transcript_sha="c" * 64,
        language="en",
        aligner_model_id="facebook/mms-1b-all",
        aligner_params={"romanize": False},
        wrapper_hash="b" * 64,
        senselab_ver="1.3.1-alpha.27",
    )
    assert asr_k != align_k

    # Re-running alignment with a different transcript on the same audio invalidates only the alignment key
    align_k2 = aa.align_cache_key(
        audio_sig="a" * 64,
        transcript_sha="d" * 64,
        language="en",
        aligner_model_id="facebook/mms-1b-all",
        aligner_params={"romanize": False},
        wrapper_hash="b" * 64,
        senselab_ver="1.3.1-alpha.27",
    )
    assert align_k != align_k2


def test_transcript_signature_stable_and_unique(aa: types.ModuleType) -> None:
    """sha256(text) is deterministic and content-sensitive."""
    assert aa.transcript_signature("hello world") == aa.transcript_signature("hello world")
    assert aa.transcript_signature("hello world") != aa.transcript_signature("hello, world")


def test_asr_has_timestamps_detects_native_timestamps(aa: types.ModuleType) -> None:
    """ScriptLines with start/end set are recognized as 'has timestamps'."""
    timed = [SimpleNamespace(text="hi", start=0.0, end=1.0, chunks=None)]
    text_only = [SimpleNamespace(text="hi", start=None, end=None, chunks=None)]
    text_with_chunks = [
        SimpleNamespace(text="hi", start=None, end=None, chunks=[SimpleNamespace(text="hi", start=0.1, end=0.5)])
    ]
    assert aa._asr_has_timestamps(timed) is True
    assert aa._asr_has_timestamps(text_only) is False
    assert aa._asr_has_timestamps(text_with_chunks) is True
    assert aa._asr_has_timestamps([]) is False
    assert aa._asr_has_timestamps(None) is False


def test_safe_sanitizes_model_ids_for_filenames(aa: types.ModuleType) -> None:
    """Forward slashes and dots in model ids must become underscore-safe stems."""
    assert "/" not in aa._safe("openai/whisper-large-v3-turbo")
    assert "/" not in aa._safe("speechbrain/spkrec-ecapa-voxceleb")
    # idempotent on already-safe input
    assert aa._safe("plain_name") == "plain_name"


def test_collect_classification_labels_pulls_unique_labels(aa: types.ModuleType) -> None:
    """The LS-config XML builder collects every distinct AudioSet label observed."""
    classify_result = [
        [
            {"start": 0.0, "end": 0.5, "labels": ["Speech", "Music"], "scores": [0.9, 0.1]},
            {"start": 0.5, "end": 1.0, "labels": ["Speech", "Silence"], "scores": [0.8, 0.2]},
        ]
    ]
    labels = aa._collect_classification_labels(classify_result)
    assert labels == {"Speech", "Music", "Silence"}


def test_serialize_handles_tensors_dicts_and_lists(aa: types.ModuleType) -> None:
    """The output JSON serializer preserves tensor metadata + handles nested structures."""
    payload = {
        "embedding": torch.zeros(3, dtype=torch.float32),
        "items": [{"a": 1}, {"a": 2}],
        "ok": True,
    }
    out = aa.serialize(payload)
    assert out["embedding"]["_tensor_shape"] == [3]
    assert "_dtype" in out["embedding"]
    assert out["items"] == [{"a": 1}, {"a": 2}]
    assert out["ok"] is True


# ── Phase 2 (foundational comparator) tests ───────────────────────────


def test_comparator_cli_flags_parse(aa: types.ModuleType) -> None:
    """parse_args accepts the new comparator flags with documented defaults.

    Defaults reflect the 2026-05-09 clarifications: cross-stream grid is now
    0.5 s non-overlapping (the 0.1 / 0.2 grid over-resolved every signal in
    the system), and ``--speech-presence-labels`` is ``nargs="+"`` since
    AudioSet labels themselves contain commas (e.g. ``"Narration, monologue"``).
    """
    args = aa.parse_args(["/tmp/dummy.wav"])
    assert args.cross_stream_win_length == 0.5
    assert args.cross_stream_hop_length == 0.5
    assert args.uncertainty_aggregator == "min"
    assert args.phoneme_disagreement_threshold == 0.50
    assert args.diarization_boundary_shift_ms == 50.0
    assert args.disagreements_top_n == 100
    assert args.asr_reference_model == "openai/whisper-large-v3-turbo"
    assert tuple(args.skip_comparisons) == ()
    assert "comparisons" in aa.ALL_TASKS
    # The default labels include "Narration, monologue" — survives nargs="+".
    assert any("Narration" in lbl for lbl in args.speech_presence_labels)


def test_comparator_cli_flag_validation(aa: types.ModuleType) -> None:
    """Out-of-range comparator flag values are rejected by argparse."""
    with pytest.raises(SystemExit):
        aa.parse_args(["/tmp/dummy.wav", "--cross-stream-win-length", "-1"])
    with pytest.raises(SystemExit):
        aa.parse_args(["/tmp/dummy.wav", "--cross-stream-hop-length", "0.6", "--cross-stream-win-length", "0.2"])
    with pytest.raises(SystemExit):
        aa.parse_args(["/tmp/dummy.wav", "--phoneme-disagreement-threshold", "1.5"])
    with pytest.raises(SystemExit):
        aa.parse_args(["/tmp/dummy.wav", "--disagreements-top-n", "-3"])
    with pytest.raises(SystemExit):
        aa.parse_args(["/tmp/dummy.wav", "--diarization-boundary-shift-ms", "-1"])


def test_classification_to_ls_emits_regions_for_dict_shape(aa: types.ModuleType) -> None:
    """The LS conversion must skip empty entries but emit one region per dict window."""
    result = [
        [
            {"start": 0.0, "end": 0.5, "labels": ["Speech"], "scores": [0.95]},
            {"start": 0.5, "end": 1.0, "labels": ["Music"], "scores": [0.62]},
        ]
    ]
    regions = aa._classification_to_ls(result, prefix="raw__ast", win_length=0.5, hop_length=0.5)
    assert len(regions) == 2
    labels = [r["value"]["labels"][0] for r in regions]
    assert labels == ["Speech", "Music"]


def test_speech_presence_labels_preserves_multi_word_audioset_labels(aa: types.ModuleType) -> None:
    """AudioSet labels themselves contain commas (e.g. 'Narration, monologue').

    nargs="+" + space-separated quoted args means the inner commas survive parsing.
    """
    args = aa.parse_args(
        [
            "/tmp/dummy.wav",
            "--speech-presence-labels",
            "Speech",
            "Narration, monologue",
            "Female speech, woman speaking",
        ]
    )
    labels = aa._speech_presence_labels(args)
    assert "Narration, monologue" in labels
    assert "Female speech, woman speaking" in labels
    assert "Speech" in labels


def test_skip_comparisons_disables_workflow_outputs(aa: types.ModuleType) -> None:
    """``--skip comparisons`` sets ``comparisons`` in ``args.skip`` (T009b / FR-008 / SC-005).

    The script's main() gates the workflow call on that exact membership.
    """
    args = aa.parse_args(["/tmp/dummy.wav", "--skip", "comparisons"])
    assert "comparisons" in args.skip


def test_disagreements_top_n_zero_disables_index_only(aa: types.ModuleType) -> None:
    """--disagreements-top-n 0 keeps the parquets + plot; only the index file is skipped."""
    args = aa.parse_args(["/tmp/dummy.wav", "--disagreements-top-n", "0"])
    assert args.disagreements_top_n == 0
    # comparisons stay enabled
    assert "comparisons" not in args.skip


def test_utterance_grid_defaults_to_1s_window_05s_hop(aa: types.ModuleType) -> None:
    """Utterance has its own grid: 1.0 s window with 0.5 s hop (overlapping).

    Wider than presence/identity (0.5/0.5) so most words land fully inside at least
    one bucket — pairs with the fully-contained rule in harvest_utterance_votes.
    """
    args = aa.parse_args(["/tmp/dummy.wav"])
    assert args.utterance_win_length == 1.0
    assert args.utterance_hop_length == 0.5


def test_utterance_grid_validation(aa: types.ModuleType) -> None:
    """Out-of-range utterance grid values are rejected."""
    with pytest.raises(SystemExit):
        aa.parse_args(["/tmp/dummy.wav", "--utterance-win-length", "-1"])
    with pytest.raises(SystemExit):
        aa.parse_args(["/tmp/dummy.wav", "--utterance-hop-length", "1.5", "--utterance-win-length", "1.0"])
