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
            [{"label": "Speech", "score": 0.9}, {"label": "Music", "score": 0.1}],
            [{"label": "Speech", "score": 0.8}, {"label": "Silence", "score": 0.2}],
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
    """parse_args accepts the new comparator flags with documented defaults."""
    args = aa.parse_args(["/tmp/dummy.wav"])
    assert args.cross_stream_win_length == 0.2
    assert args.cross_stream_hop_length == 0.1
    assert args.uncertainty_aggregator == "min"
    assert args.phoneme_disagreement_threshold == 0.50
    assert args.diarization_boundary_shift_ms == 50.0
    assert args.disagreements_top_n == 100
    assert args.asr_reference_model == "openai/whisper-large-v3-turbo"
    assert tuple(args.skip_comparisons) == ()
    assert "comparisons" in aa.ALL_TASKS


def test_comparator_cli_flag_validation(aa: types.ModuleType) -> None:
    """Out-of-range comparator flag values are rejected by argparse."""
    with pytest.raises(SystemExit):
        aa.parse_args(["/tmp/dummy.wav", "--cross-stream-win-length", "-1"])
    with pytest.raises(SystemExit):
        aa.parse_args(["/tmp/dummy.wav", "--cross-stream-hop-length", "0.5", "--cross-stream-win-length", "0.2"])
    with pytest.raises(SystemExit):
        aa.parse_args(["/tmp/dummy.wav", "--phoneme-disagreement-threshold", "1.5"])
    with pytest.raises(SystemExit):
        aa.parse_args(["/tmp/dummy.wav", "--disagreements-top-n", "-3"])
    with pytest.raises(SystemExit):
        aa.parse_args(["/tmp/dummy.wav", "--diarization-boundary-shift-ms", "-1"])


def test_comparison_grid_iterates_buckets(aa: types.ModuleType) -> None:
    """ComparisonGrid yields buckets that respect win/hop and end-of-audio."""
    grid = aa.ComparisonGrid(win_length=1.0, hop_length=0.5, name="test")
    buckets = list(grid.iter_buckets(2.5))
    # Expected starts: 0.0, 0.5, 1.0, 1.5 (1.5 + 1.0 = 2.5 ≤ 2.5)
    assert [b[0] for b in buckets] == [0.0, 0.5, 1.0, 1.5]
    assert [b[2] for b in buckets] == [0, 1, 2, 3]
    assert all(end - start == 1.0 for start, end, _ in buckets[:-1])


def test_aggregate_uncertainty_min_default(aa: types.ModuleType) -> None:
    """Min-aggregator returns 1 - min(confidences); most-doubtful model wins ranking."""
    assert aa._aggregate_uncertainty([0.9, 0.4], "min") == pytest.approx(0.6)
    # mean: 1 - 0.65 = 0.35
    assert aa._aggregate_uncertainty([0.9, 0.4], "mean") == pytest.approx(0.35)
    # disagreement_weighted with default mismatch_severity=1.0 → (1 - mean)
    assert aa._aggregate_uncertainty([0.9, 0.4], "disagreement_weighted") == pytest.approx(0.35)
    # harmonic_mean handles 0 via clamp
    assert aa._aggregate_uncertainty([0.9, 0.0], "harmonic_mean") == pytest.approx(1.0, abs=1e-3)


def test_aggregate_uncertainty_drops_none_not_zeroes(aa: types.ModuleType) -> None:
    """Models lacking native confidence are dropped, not treated as zero."""
    assert aa._aggregate_uncertainty([0.9, None], "min") == pytest.approx(0.1)
    assert aa._aggregate_uncertainty([None, None], "min") is None


def test_token_overlaps_window_with_chunks(aa: types.ModuleType) -> None:
    """A ScriptLine-shaped result with chunks reports overlap with a window."""
    # Mock a result with chunks at [10.0, 11.0]; query window [9.0, 10.5] → overlaps
    line = SimpleNamespace(
        text="hello",
        start=10.0,
        end=11.0,
        chunks=[SimpleNamespace(text="hi", start=10.0, end=10.5, chunks=None)],
    )
    assert aa._token_overlaps_window([line], 9.0, 10.5) is True
    assert aa._token_overlaps_window([line], 11.0, 12.0) is False
    # Dict-shaped (cache-restored) version
    line_dict = {"text": "hi", "start": 10.0, "end": 11.0, "chunks": [{"text": "hi", "start": 10.0, "end": 10.5}]}
    assert aa._token_overlaps_window([line_dict], 9.0, 10.5) is True


def test_comparator_skip_no_op_preserves_ls_bundle_shape(aa: types.ModuleType) -> None:
    """build_labelstudio_config with no comparator_tracks emits no comparator-track XML."""
    summary: dict[str, Any] = {"passes": {"raw_16k": {}}}
    xml_no_comparator = aa.build_labelstudio_config(summary)
    xml_explicit_empty = aa.build_labelstudio_config(summary, comparator_tracks=[])
    assert xml_no_comparator == xml_explicit_empty
    assert "compare__" not in xml_no_comparator
    # Adding a comparator track does inject the fixed-enum Labels block.
    xml_with = aa.build_labelstudio_config(
        summary,
        comparator_tracks=[{"name": "raw_16k__compare__asr__a_vs_b", "with_textarea": True}],
    )
    assert 'name="raw_16k__compare__asr__a_vs_b"' in xml_with
    assert '<Label value="agree"/>' in xml_with
    assert '<Label value="disagree"/>' in xml_with
    assert '<Label value="incomparable"/>' in xml_with
    assert '<Label value="one_sided"/>' in xml_with
    assert 'name="raw_16k__compare__asr__a_vs_b__text"' in xml_with


def test_comparison_cache_key_changes_with_inputs(aa: types.ModuleType, tmp_path: Path) -> None:
    """comparison_cache_key flips when any input changes (including upstream cache keys)."""
    base: dict[str, Any] = {
        "audio_sig": "A" * 64,
        "comparison_kind": "within_stream",
        "task_or_pair": "asr",
        "upstream_cache_keys": ["k1", "k2"],
        "params": {"grid": "cross_stream", "agg": "min"},
        "wrapper_hash": "W" * 64,
        "senselab_ver": "1.0",
    }
    k0 = aa.comparison_cache_key(**base)
    # Same inputs → same key
    assert aa.comparison_cache_key(**base) == k0
    # Order of upstream keys is normalized
    assert aa.comparison_cache_key(**{**base, "upstream_cache_keys": ["k2", "k1"]}) == k0
    # Any input change → different key
    assert aa.comparison_cache_key(**{**base, "comparison_kind": "cross_stream"}) != k0
    assert aa.comparison_cache_key(**{**base, "task_or_pair": "diar"}) != k0
    new_params = {**base["params"], "agg": "mean"}
    assert aa.comparison_cache_key(**{**base, "params": new_params}) != k0


def test_disagreements_index_top_n_zero_disables(aa: types.ModuleType, tmp_path: Path) -> None:
    """top_n=0 returns an index with empty entries list and zero totals."""
    idx = aa._build_disagreements_index(
        parquet_paths=[],
        top_n=0,
        aggregator="min",
        run_dir=tmp_path,
        config={"top_n": 0},
        incomparable_reasons={},
        missing_confidence_signals=[],
    )
    assert idx["entries"] == []
    assert idx["totals"]["total_rows"] == 0


def test_write_comparison_parquet_no_op_for_empty_rows(aa: types.ModuleType, tmp_path: Path) -> None:
    """write_comparison_parquet skips writing entirely when given no rows."""
    dest = tmp_path / "empty.parquet"
    aa.write_comparison_parquet([], dest, provenance={"k": "v"})
    assert not dest.exists()


# ── Phase 3 (US1 raw_vs_enhanced) tests ───────────────────────────────


def _mk_diar_result(segments: list[tuple[float, float, str]]) -> list[list[Any]]:
    """Build a List[List[ScriptLine-like]] from a list of (start, end, speaker) tuples."""
    inner = [SimpleNamespace(text=None, speaker=spk, start=s, end=e, chunks=None) for s, e, spk in segments]
    return [inner]


def _mk_asr_result(chunks: list[tuple[float, float, str]]) -> list[Any]:
    """Build a List[ScriptLine-like] with one line whose ``chunks`` come from the input."""
    inner_chunks = [SimpleNamespace(text=t, start=s, end=e, chunks=None, speaker=None) for s, e, t in chunks]
    return [
        SimpleNamespace(text=" ".join(t for _, _, t in chunks), start=None, end=None, chunks=inner_chunks, speaker=None)
    ]


def test_raw_vs_enhanced_diarization_diff(aa: types.ModuleType, tmp_path: Path) -> None:
    """Diarization differencer flags speech-presence flips between raw and enhanced."""
    raw = _mk_diar_result([(0.0, 10.0, "spk0")])
    enh = _mk_diar_result([(0.0, 5.0, "spk0"), (6.0, 10.0, "spk0")])
    grid = aa.ComparisonGrid(win_length=0.5, hop_length=0.5, name="test")
    base = {
        "comparison_kind": "raw_vs_enhanced",
        "task": "diarization",
        "stream_pair": None,
        "model_a": "M",
        "model_b": "M",
        "pass_a": "raw_16k",
        "pass_b": "enhanced_16k",
        "confidence_a": None,
        "confidence_b": None,
        "combined_uncertainty": None,
    }
    rows = aa._diff_diarization(raw, enh, grid, duration_s=10.0, boundary_shift_ms=50.0, **base)
    # Bucket [5.0, 5.5] is silent in enhanced; bucket [5.5, 6.0] also silent in enhanced.
    flipped = [r for r in rows if not r["agree"]]
    assert flipped, "expected at least one boundary_shift bucket"
    assert all(r["mismatch_type"] == "boundary_shift" for r in flipped)
    assert any(5.0 <= r["start"] < 6.0 for r in flipped)


def test_raw_vs_enhanced_asr_text_diff(aa: types.ModuleType) -> None:
    """ASR differencer emits per-bucket WER + a_text/b_text on the cross-stream grid."""
    raw = _mk_asr_result([(0.0, 0.5, "hello"), (0.5, 1.0, "world")])
    enh = _mk_asr_result([(0.0, 0.5, "hello"), (0.5, 1.0, "earth")])
    grid = aa.ComparisonGrid(win_length=0.5, hop_length=0.5, name="cross_stream")
    base = {
        "comparison_kind": "raw_vs_enhanced",
        "task": "asr",
        "stream_pair": None,
        "model_a": "whisper",
        "model_b": "whisper",
        "pass_a": "raw_16k",
        "pass_b": "enhanced_16k",
        "confidence_a": None,
        "confidence_b": None,
        "combined_uncertainty": None,
    }
    rows = aa._diff_asr(raw, enh, grid, duration_s=1.0, reference_side="a", **base)
    assert len(rows) == 2
    # Bucket 0 agrees; bucket 1 disagrees with WER > 0.
    assert rows[0]["agree"] is True
    assert rows[0]["wer"] == 0.0
    assert rows[1]["agree"] is False
    assert rows[1]["wer"] > 0.0
    assert rows[1]["mismatch_type"] == "text_edit"
    assert "world" in rows[1]["a_text"]
    assert "earth" in rows[1]["b_text"]
    assert rows[1]["reference_side"] == "a"


# ── Phase 4 (US2 within-stream) tests ─────────────────────────────────


def test_within_stream_asr_pair_emits_wer(aa: types.ModuleType, tmp_path: Path) -> None:
    """compare_within_stream with two ASR models emits a per-pair parquet with WER."""
    # Coarser fixture: word "hello" alone spans bucket [0.0, 0.5]; "world"/"earth"
    # alone spans bucket [0.5, 1.0]. Use a 0.5/0.5 grid so the buckets line up
    # with the chunk boundaries and the disagreement is unambiguous.
    res_a = _mk_asr_result([(0.0, 0.5, "hello"), (0.5, 1.0, "world")])
    res_b = _mk_asr_result([(0.0, 0.5, "hello"), (0.5, 1.0, "earth")])
    summary = {
        "duration_s": 1.0,
        "asr": {
            "by_model": {
                "whisper": {"status": "ok", "result": res_a, "cache_key": "ka"},
                "qwen": {"status": "ok", "result": res_b, "cache_key": "kb"},
            }
        },
    }
    args = aa.parse_args(["/tmp/dummy.wav", "--cross-stream-win-length", "0.5", "--cross-stream-hop-length", "0.5"])
    parquets, tracks, incomparable = aa.compare_within_stream(
        pass_label="raw_16k",
        pass_summary=summary,
        args=args,
        run_dir=tmp_path,
        cache_dir=None,
        audio_sig="A" * 64,
        duration_s=1.0,
        wrapper_hash="W",
        senselab_ver="1.0",
    )
    assert len(parquets) == 1
    assert any("compare__asr__" in t["name"] for t in tracks)

    import pandas as pd

    df = pd.read_parquet(parquets[0])
    assert len(df) == 2
    # Both rows have a_text + b_text; first agrees, second disagrees.
    agree_col = df["agree"].astype(bool).tolist()
    wer_col = df["wer"].tolist()
    assert agree_col == [True, False]
    assert wer_col[0] == 0.0
    assert wer_col[1] > 0.0


def test_within_stream_single_model_no_op(aa: types.ModuleType, tmp_path: Path) -> None:
    """A task with only one successful model produces no within-stream parquet."""
    summary = {
        "duration_s": 1.0,
        "asr": {"by_model": {"whisper": {"status": "ok", "result": _mk_asr_result([(0.0, 1.0, "hi")])}}},
    }
    args = aa.parse_args(["/tmp/dummy.wav"])
    parquets, tracks, _ = aa.compare_within_stream(
        pass_label="raw_16k",
        pass_summary=summary,
        args=args,
        run_dir=tmp_path,
        cache_dir=None,
        audio_sig="A" * 64,
        duration_s=1.0,
        wrapper_hash="W",
        senselab_ver="1.0",
    )
    assert parquets == []
    assert tracks == []


def test_within_stream_classification_superset_of_scene_agreement(aa: types.ModuleType, tmp_path: Path) -> None:
    """AST + YAMNet on a matching grid produce a within-stream scene parquet."""
    # Two windows each on both AST and YAMNet, agreeing on window 0, differing on window 1.
    ast_result = [
        [
            [{"label": "Speech", "score": 0.9}, {"label": "Music", "score": 0.1}],
            [{"label": "Speech", "score": 0.55}, {"label": "Music", "score": 0.45}],
        ]
    ]
    yam_result = [
        [
            [{"label": "Speech", "score": 0.85}, {"label": "Music", "score": 0.15}],
            [{"label": "Music", "score": 0.6}, {"label": "Speech", "score": 0.4}],
        ]
    ]
    # Force matching grids by overriding the AST/YAMNet flags to YAMNet's native.
    args = aa.parse_args(
        [
            "/tmp/dummy.wav",
            "--ast-win-length",
            "0.96",
            "--ast-hop-length",
            "0.96",
            "--yamnet-win-length",
            "0.96",
            "--yamnet-hop-length",
            "0.96",
        ]
    )
    summary = {
        "duration_s": 2.0,
        "ast": {"status": "ok", "result": ast_result, "cache_key": "kast"},
        "yamnet": {"status": "ok", "result": yam_result, "cache_key": "kyam"},
    }
    parquets, tracks, _ = aa.compare_within_stream(
        pass_label="raw_16k",
        pass_summary=summary,
        args=args,
        run_dir=tmp_path,
        cache_dir=None,
        audio_sig="A" * 64,
        duration_s=2.0,
        wrapper_hash="W",
        senselab_ver="1.0",
    )
    scene_parquets = [p for p in parquets if "scene" in str(p)]
    assert scene_parquets, "expected an ast_vs_yamnet parquet"
    import pandas as pd

    df = pd.read_parquet(scene_parquets[0])
    assert len(df) == 2
    assert df.iloc[0]["top1_a"] == "Speech"
    assert df.iloc[0]["top1_b"] == "Speech"
    assert df["agree"].astype(bool).tolist() == [True, False]
    assert df.iloc[1]["mismatch_type"] == "label_flip"


# ── Phase 5 (US3 cross-stream) tests ──────────────────────────────────


def test_cross_stream_asr_speech_in_silent_window(aa: types.ModuleType) -> None:
    """ASR returned a token in [12.0, 12.3] but diarization says silence — disagreement."""
    asr = _mk_asr_result([(12.0, 12.3, "hello")])
    diar = _mk_diar_result([(0.0, 11.0, "spk0"), (15.0, 20.0, "spk0")])
    grid = aa.ComparisonGrid(win_length=0.2, hop_length=0.1, name="cross_stream")
    rows = aa._diff_asr_vs_diarization(
        asr,
        diar,
        grid,
        duration_s=20.0,
        asr_model="whisper",
        diar_model="pyannote",
        pass_label="raw_16k",
    )
    # Find a bucket inside [12.0, 12.3]: e.g. start=12.1
    flag_buckets = [r for r in rows if r["start"] >= 12.0 and r["end"] <= 12.4]
    assert flag_buckets
    assert any(r["asr_says_speech"] is True and r["diar_says_speech"] is False for r in flag_buckets)
    assert any(r["mismatch_type"] == "speech_presence_flip" for r in flag_buckets)


def test_cross_stream_classification_speech_allowlist(aa: types.ModuleType) -> None:
    """AST top-1 'Speech' in window 0, 'Music' in window 1; diar speaks only in window 1."""
    ast_result = [
        [
            [{"label": "Speech", "score": 0.9}, {"label": "Music", "score": 0.1}],
            [{"label": "Music", "score": 0.7}, {"label": "Speech", "score": 0.3}],
        ]
    ]
    # win_length=0.96, hop_length=0.96 → window 0 = [0, 0.96], window 1 = [0.96, 1.92]
    diar = _mk_diar_result([(1.0, 1.9, "spk0")])
    grid = aa.ComparisonGrid(win_length=0.2, hop_length=0.1, name="cross_stream")
    allowlist = {"Speech", "Conversation"}
    rows = aa._diff_classification_vs_diarization(
        ast_result,
        diar,
        grid,
        duration_s=2.0,
        class_win_length=0.96,
        class_hop_length=0.96,
        allowlist=allowlist,
        class_model="ast",
        diar_model="pyannote",
        pass_label="raw_16k",
    )
    # Disagreements: in window 0 (AST says Speech, diar silent) and window 1 (AST says Music, diar speaks)
    assert any(not r["agree"] for r in rows[:5])  # some early bucket disagrees
    assert any(not r["agree"] for r in rows[15:])  # some late bucket disagrees too


def test_cross_stream_ppg_unavailable_degrades_gracefully(aa: types.ModuleType, tmp_path: Path) -> None:
    """When PPG is missing, compare_cross_stream records incomparable, not crash."""
    summary = {
        "duration_s": 1.0,
        "asr": {"by_model": {"whisper": {"status": "ok", "result": _mk_asr_result([(0.0, 1.0, "hi")])}}},
        "diarization": {"by_model": {"pyannote": {"status": "ok", "result": _mk_diar_result([(0.0, 1.0, "s")])}}},
        # Note: no "ppgs" / "ppg" key
    }
    args = aa.parse_args(["/tmp/dummy.wav"])
    parquets, tracks, incomparable = aa.compare_cross_stream(
        pass_label="raw_16k",
        pass_summary=summary,
        args=args,
        run_dir=tmp_path,
        cache_dir=None,
        audio_sig="A" * 64,
        duration_s=1.0,
        wrapper_hash="W",
        senselab_ver="1.0",
    )
    # ASR↔diar parquet exists; PPG path is recorded as unavailable.
    assert any("asr__whisper__vs__diarization__pyannote" in str(p) for p in parquets)
    assert "raw_16k/asr_vs_ppg" in incomparable
    assert "PPG" in incomparable["raw_16k/asr_vs_ppg"]


def test_cross_stream_phoneme_per_two_tier(aa: types.ModuleType) -> None:
    """phoneme_per is continuous; phoneme_disagreement boolean fires at the threshold."""
    # Synthetic ASR result: a single chunk with text "hello" spanning [0.0, 1.0]
    asr = _mk_asr_result([(0.0, 1.0, "hello")])
    # Synthetic PPG result: 100 frames, all argmax to "M" (mismatch)
    ppg_frames = [[{"phoneme": "M", "score": 1.0}] for _ in range(100)]
    grid = aa.ComparisonGrid(win_length=0.5, hop_length=0.5, name="cross_stream")
    rows = aa._diff_asr_vs_ppg(
        asr,
        [ppg_frames],
        grid,
        duration_s=1.0,
        threshold=0.50,
        asr_model="whisper",
        pass_label="raw_16k",
    )
    # Both buckets emit a row with continuous phoneme_per and boolean disagreement.
    assert rows
    for r in rows:
        assert r["phoneme_per"] >= 0.0
        # PER == 1.0 (every phoneme differs) → flagged as disagreement.
        assert r["phoneme_disagreement"] is True


# ── Phase 6 (US4 uncertainty) tests ───────────────────────────────────


def test_uncertainty_whisper_avg_logprob_extracted(aa: types.ModuleType) -> None:
    """Whisper avg_logprob -> exp(avg_logprob) confidence; no_speech_prob preserved."""
    import math

    chunk = SimpleNamespace(text="hi", start=0.0, end=1.0, avg_logprob=-0.5, no_speech_prob=0.1)
    conf, nsp = aa._whisper_chunk_confidence(chunk)
    assert conf is not None
    assert conf == pytest.approx(math.exp(-0.5))
    assert nsp == 0.1
    # Missing signal returns (None, None)
    blank = SimpleNamespace(text="hi", start=0.0, end=1.0)
    assert aa._whisper_chunk_confidence(blank) == (None, None)


def test_uncertainty_aggregator_min_default_via_compare(aa: types.ModuleType, tmp_path: Path) -> None:
    """ASR-vs-ASR rows pick up combined_uncertainty = 1 - min(confidence_a, confidence_b)."""
    # Build two ASR results where each chunk carries an avg_logprob.
    chunks_a = [SimpleNamespace(text="hello", start=0.0, end=0.5, avg_logprob=-0.1, chunks=None, speaker=None)]
    chunks_b = [SimpleNamespace(text="hello", start=0.0, end=0.5, avg_logprob=-1.0, chunks=None, speaker=None)]
    res_a = [SimpleNamespace(text="hello", start=None, end=None, chunks=chunks_a, speaker=None)]
    res_b = [SimpleNamespace(text="hello", start=None, end=None, chunks=chunks_b, speaker=None)]
    summary = {
        "duration_s": 0.5,
        "asr": {
            "by_model": {
                "a": {"status": "ok", "result": res_a, "cache_key": "ka"},
                "b": {"status": "ok", "result": res_b, "cache_key": "kb"},
            }
        },
    }
    args = aa.parse_args(["/tmp/dummy.wav", "--cross-stream-win-length", "0.5", "--cross-stream-hop-length", "0.5"])
    parquets, _, _ = aa.compare_within_stream(
        pass_label="raw_16k",
        pass_summary=summary,
        args=args,
        run_dir=tmp_path,
        cache_dir=None,
        audio_sig="A" * 64,
        duration_s=0.5,
        wrapper_hash="W",
        senselab_ver="1.0",
    )
    import math

    import pandas as pd

    df = pd.read_parquet(parquets[0])
    # avg_logprob: a=-0.1 -> conf 0.905; b=-1.0 -> conf 0.368. min(...) = 0.368 → uncertainty 0.632.
    expected = 1.0 - math.exp(-1.0)
    assert df.iloc[0]["confidence_a"] == pytest.approx(math.exp(-0.1), rel=1e-6)
    assert df.iloc[0]["confidence_b"] == pytest.approx(math.exp(-1.0), rel=1e-6)
    assert df.iloc[0]["combined_uncertainty"] == pytest.approx(expected, rel=1e-6)


def test_uncertainty_missing_signal_dropped_not_zeroed(aa: types.ModuleType) -> None:
    """A row with one None confidence sees combined_uncertainty = 1 - the other confidence."""
    out = aa._aggregate_uncertainty([0.9, None], "min")
    assert out == pytest.approx(0.1)
    # Both None → None
    assert aa._aggregate_uncertainty([None, None], "min") is None


def test_disagreements_index_top_n_and_ordering(aa: types.ModuleType, tmp_path: Path) -> None:
    """Disagreements index: sort by combined_uncertainty desc, mismatch priority, start asc."""
    import pandas as pd

    rows = [
        {
            "start": 0.0,
            "end": 0.2,
            "comparison_kind": "within_stream",
            "task": "asr",
            "stream_pair": None,
            "model_a": "a",
            "model_b": "b",
            "pass_a": "raw_16k",
            "pass_b": "raw_16k",
            "agree": False,
            "mismatch_type": "text_edit",
            "comparison_status": "ok",
            "confidence_a": 0.5,
            "confidence_b": 0.5,
            "combined_uncertainty": 0.5,
        },
        {
            "start": 0.2,
            "end": 0.4,
            "comparison_kind": "within_stream",
            "task": "asr",
            "stream_pair": None,
            "model_a": "a",
            "model_b": "b",
            "pass_a": "raw_16k",
            "pass_b": "raw_16k",
            "agree": False,
            "mismatch_type": "text_edit",
            "comparison_status": "ok",
            "confidence_a": 0.1,
            "confidence_b": 0.1,
            "combined_uncertainty": 0.9,
        },
        {
            "start": 0.4,
            "end": 0.6,
            "comparison_kind": "within_stream",
            "task": "asr",
            "stream_pair": None,
            "model_a": "a",
            "model_b": "b",
            "pass_a": "raw_16k",
            "pass_b": "raw_16k",
            "agree": True,
            "mismatch_type": None,
            "comparison_status": "ok",
            "confidence_a": 0.95,
            "confidence_b": 0.95,
            "combined_uncertainty": 0.05,
        },
    ]
    parquet = tmp_path / "raw_16k" / "comparisons" / "asr" / "a_vs_b.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet, index=False)

    idx = aa._build_disagreements_index(
        parquet_paths=[parquet],
        top_n=10,
        aggregator="min",
        run_dir=tmp_path,
        config={"top_n": 10},
        incomparable_reasons={},
        missing_confidence_signals=[],
    )
    # Only the two disagreement rows make it; agree=True row is excluded.
    assert idx["totals"]["disagreement_rows"] == 2
    assert len(idx["entries"]) == 2
    # Sort: highest combined_uncertainty first.
    assert idx["entries"][0]["combined_uncertainty"] == 0.9
    assert idx["entries"][1]["combined_uncertainty"] == 0.5
    assert idx["entries"][0]["rank"] == 1
    assert idx["entries"][1]["rank"] == 2


def test_models_without_native_confidence_lists_known_null(aa: types.ModuleType) -> None:
    """The disagreements.json missing_confidence_signals list mentions known-null backends."""
    summaries = {
        "passes": {
            "raw_16k": {
                "duration_s": 1.0,
                "diarization": {
                    "by_model": {
                        "pyannote/speaker-diarization-community-1": {"status": "ok", "result": [[]]},
                        "openai/whisper-large-v3-turbo": {"status": "ok", "result": [[]]},  # not in known-null list
                    }
                },
                "asr": {
                    "by_model": {
                        "ibm-granite/granite-speech-3.3-8b": {"status": "ok", "result": []},
                        "openai/whisper-large-v3-turbo": {"status": "ok", "result": []},
                    }
                },
            }
        }
    }
    out = aa._models_without_native_confidence(summaries)
    assert "pyannote/speaker-diarization-community-1" in out
    assert "ibm-granite/granite-speech-3.3-8b" in out
    # Whisper is NOT in the null list (it has avg_logprob).
    assert "openai/whisper-large-v3-turbo" not in out


def test_raw_vs_enhanced_handles_failed_pass(aa: types.ModuleType, tmp_path: Path) -> None:
    """When one pass failed, comparator emits no parquet for that task/model (incomparable)."""
    summaries = {
        "passes": {
            "raw_16k": {
                "duration_s": 1.0,
                "diarization": {"by_model": {"M": {"status": "ok", "result": _mk_diar_result([(0.0, 1.0, "s")])}}},
            },
            "enhanced_16k": {
                "duration_s": 1.0,
                "diarization": {"by_model": {"M": {"status": "failed", "error": "boom"}}},
            },
        },
    }
    args = aa.parse_args(["/tmp/dummy.wav"])
    parquets, tracks, incomparable = aa.compare_raw_vs_enhanced(
        summaries=summaries,
        args=args,
        run_dir=tmp_path,
        cache_dir=None,
        audio_sig="X" * 64,
        duration_s=1.0,
        wrapper_hash="W",
        senselab_ver="1.0",
    )
    assert parquets == []
    assert tracks == []
    assert "diarization/M/raw_vs_enhanced" in incomparable


def test_write_comparison_parquet_persists_provenance(aa: types.ModuleType, tmp_path: Path) -> None:
    """Parquet file carries the comparator_provenance metadata blob."""
    import pyarrow.parquet as pq

    rows = [
        {
            "start": 0.0,
            "end": 0.2,
            "comparison_kind": "within_stream",
            "task": "asr",
            "stream_pair": None,
            "model_a": "a",
            "model_b": "b",
            "pass_a": "raw_16k",
            "pass_b": "raw_16k",
            "agree": False,
            "mismatch_type": "text_edit",
            "comparison_status": "ok",
            "confidence_a": 0.8,
            "confidence_b": 0.4,
            "combined_uncertainty": 0.6,
        }
    ]
    dest = tmp_path / "out.parquet"
    prov = {"schema_version": 1, "wrapper_hash": "abc", "senselab_version": "1.0"}
    aa.write_comparison_parquet(rows, dest, provenance=prov)
    assert dest.exists()
    table = pq.read_table(dest)
    metadata = table.schema.metadata or {}
    raw = metadata.get(b"comparator_provenance")
    assert raw is not None
    import json as _json

    parsed = _json.loads(raw.decode("utf-8"))
    assert parsed["schema_version"] == 1
    assert parsed["wrapper_hash"] == "abc"
