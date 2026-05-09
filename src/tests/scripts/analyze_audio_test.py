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
    summary = {"passes": {"raw_16k": {}}}
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
    base = dict(
        audio_sig="A" * 64,
        comparison_kind="within_stream",
        task_or_pair="asr",
        upstream_cache_keys=["k1", "k2"],
        params={"grid": "cross_stream", "agg": "min"},
        wrapper_hash="W" * 64,
        senselab_ver="1.0",
    )
    k0 = aa.comparison_cache_key(**base)
    # Same inputs → same key
    assert aa.comparison_cache_key(**base) == k0
    # Order of upstream keys is normalized
    assert aa.comparison_cache_key(**{**base, "upstream_cache_keys": ["k2", "k1"]}) == k0
    # Any input change → different key
    assert aa.comparison_cache_key(**{**base, "comparison_kind": "cross_stream"}) != k0
    assert aa.comparison_cache_key(**{**base, "task_or_pair": "diar"}) != k0
    assert aa.comparison_cache_key(**{**base, "params": {**base["params"], "agg": "mean"}}) != k0


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
