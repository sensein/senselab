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
    # Default model lists per spec FR-005 and contracts/cli.md (ASR overhaul:
    # CrisperWhisper 2.0 turbo replaces Whisper, Granite removed).
    assert "nyralabs/CrisperWhisper2.0_turbo" in args.asr_models
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

    Defaults reflect the finer-speaker-windows retuning: the cross-stream grid
    is 0.25 s non-overlapping (the 0.5 s grid under-resolved speaker changes),
    and ``--speech-presence-labels`` is ``nargs="+"`` since AudioSet labels
    themselves contain commas (e.g. ``"Narration, monologue"``).
    """
    args = aa.parse_args(["/tmp/dummy.wav"])
    assert args.cross_stream_win_length == 0.25
    assert args.cross_stream_hop_length == 0.25
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


def test_asr_grid_defaults_to_1s_window_05s_hop(aa: types.ModuleType) -> None:
    """Utterance has its own grid: 1.0 s window with 0.5 s hop (overlapping).

    Wider than speech_presence/speaker (0.5/0.5) so most words land fully inside at least
    one bucket — pairs with the fully-contained rule in harvest_asr_votes.
    """
    args = aa.parse_args(["/tmp/dummy.wav"])
    assert args.asr_win_length == 1.0
    assert args.asr_hop_length == 0.5


def test_asr_grid_validation(aa: types.ModuleType) -> None:
    """Out-of-range asr grid values are rejected."""
    with pytest.raises(SystemExit):
        aa.parse_args(["/tmp/dummy.wav", "--asr-win-length", "-1"])
    with pytest.raises(SystemExit):
        aa.parse_args(["/tmp/dummy.wav", "--asr-hop-length", "1.5", "--asr-win-length", "1.0"])


# ── CLI → library adapters (T051 step 5) ──────────────────────────────
#     These exist because the adapters are the one place argparse attribute
#     names are read by hand — a typo there is invisible to the library tests
#     and only shows up at runtime (it did: `args.align_asr` vs `no_align_asr`).


def test_pass_plan_reads_every_arg_it_needs(aa: types.ModuleType) -> None:
    """_pass_plan must not reference an argparse attribute that doesn't exist."""
    plan = aa._pass_plan(aa.parse_args(["x.wav"]))
    assert plan.asr_models, "defaults should populate ASR models"
    assert plan.align_asr is True, "alignment is on unless --no-align-asr"


def test_pass_plan_honors_no_align_asr(aa: types.ModuleType) -> None:
    """--no-align-asr is a store_true flag, so the plan must invert it."""
    assert aa._pass_plan(aa.parse_args(["x.wav", "--no-align-asr"])).align_asr is False


def test_pass_plan_translates_skip_into_absence(aa: types.ModuleType) -> None:
    """The library has no skip set — skipping is empty tuples and None ids."""
    plan = aa._pass_plan(aa.parse_args(["x.wav", "--skip", "diarization", "asr", "ast", "features"]))
    assert plan.diarization_models == ()
    assert plan.asr_models == ()
    assert plan.ast_model is None
    assert plan.features is False


def test_pass_plan_reflects_post_triage_mutation(aa: types.ModuleType) -> None:
    """Built AFTER triage: a no-speech clip must not run diarization or ASR.

    `main` mutates args.skip / args.ppg on the no-speech path, so a plan
    constructed before that would run the full suite on silence.
    """
    args = aa.parse_args(["x.wav"])
    args.skip = ["diarization", "asr", "features"]  # what triage does
    args.ppg = False
    plan = aa._pass_plan(args)
    assert plan.diarization_models == () and plan.asr_models == () and plan.ppg is False


def test_stage_context_carries_provenance_fields(aa: types.ModuleType, tmp_path: types.ModuleType) -> None:
    """The context must record the resolved source path and the perturbation it ran under."""
    import torch

    from senselab.audio.data_structures import Audio
    from senselab.audio.workflows.audio_analysis.perturbations import identity

    args = aa.parse_args(["x.wav"])
    audio = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000)
    ctx = aa._stage_context(identity(), audio, args, device=None, out_dir=tmp_path, cache_dir=None, senselab_ver="v")
    assert ctx.perturbation == "raw"
    assert ctx.device_label == "auto"
    assert ctx.audio_source.endswith("x.wav")
    assert len(ctx.audio_signature) == 64


def test_the_variant_is_the_declared_transform_not_a_guess_from_the_name(
    aa: types.ModuleType, tmp_path: Path
) -> None:
    """``variant`` comes from the perturbation's declaration, never from how its name is spelled.

    It used to be ``"speech_enhanced" if label.startswith("enhanced")``. A perturbation named for
    its model would then have claimed to be unmodified — and the background mask, which is only
    meaningful on unmodified audio, gates on exactly this field.
    """
    import torch

    from senselab.audio.data_structures import Audio
    from senselab.audio.workflows.audio_analysis.perturbations import speech_enhancement

    args = aa.parse_args(["x.wav"])
    audio = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000)
    perturbation = speech_enhancement("speechbrain/sepformer-wham16k-enhancement", name="sepformer")
    ctx = aa._stage_context(perturbation, audio, args, device=None, out_dir=tmp_path, cache_dir=None, senselab_ver="v")
    assert ctx.perturbation == "sepformer"
    assert ctx.variant == "speech_enhanced"
    assert ctx.out_dir == tmp_path / "L1" / "perturbation" / "sepformer"


def test_policy_overrides_are_absent_when_no_flags_given(aa: types.ModuleType) -> None:
    """Unset adaptive flags must produce all-None overrides, which load_policy drops."""
    from senselab.audio.workflows.audio_analysis.adaptive.policy import load_policy

    overrides = aa._policy_overrides(aa.parse_args(["x.wav"]))
    assert load_policy(None, overrides)["policy_hash"] == load_policy()["policy_hash"]


def test_policy_overrides_map_budget_and_region_flags(aa: types.ModuleType) -> None:
    """--budget-* and --region-* land on the policy keys the loop reads."""
    args = aa.parse_args(
        ["x.wav", "--budget-medium", "8", "--budget-heavy", "0", "--region-top-n", "16", "--max-region-rounds", "3"]
    )
    o = aa._policy_overrides(args)
    assert o["budget"] == {"medium_per_run": 8, "heavy_per_run": 0}
    assert o["regions"] == {"top_n_per_round": 16, "max_region_rounds": 3}


def test_policy_overrides_pass_the_reserve_pool_in_order(aa: types.ModuleType) -> None:
    """U2 escalation tries the reserve models in the order given."""
    args = aa.parse_args(["x.wav", "--reserve-asr-models", "a/one", "b/two"])
    assert aa._policy_overrides(args)["reserve_asr_models"] == ["a/one", "b/two"]


def test_overlap_flag_only_appears_when_passed(aa: types.ModuleType) -> None:
    """Without the flag the rules block is untouched, so the policy default stands."""
    assert "rules" not in aa._policy_overrides(aa.parse_args(["x.wav"]))
    on = aa._policy_overrides(aa.parse_args(["x.wav", "--enable-overlap-separation"]))
    assert on["rules"]["I4_overlap_detection"]["enabled"] is True


def test_max_rounds_defaults_to_three(aa: types.ModuleType) -> None:
    """contracts/cli.md: default 3 rounds including baseline."""
    assert aa.parse_args(["x.wav"]).max_rounds == 3
