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


def test_parse_args_takes_an_audio_file_and_where_results_go(aa: types.ModuleType) -> None:
    """Three arguments, and the third is a whole config file rather than a knob.

    Seventy flags preceded this. They are gone deliberately: the run recipes in the repo's own docs
    differed only in flags whose right value a reader had no basis to choose, and the shipped defaults
    of the four *grid* flags put the four uncertainty axes on four spacings that shared no bucket keys
    — which disabled every cross-axis coupling in the pipeline, silently.
    """
    args = aa.parse_args(["/tmp/dummy.wav"])
    assert str(args.audio).endswith("dummy.wav")
    assert args.out is None, "no --out means the config's output_dir"
    assert args.config is None, "no --config means the packaged one"
    assert set(vars(args)) == {"audio", "out", "config"}, (
        f"the CLI grew an argument: {sorted(set(vars(args)) - {'audio', 'out', 'config'})}. "
        "Values belong in data/run_config/default.yaml with their derivation."
    )


def test_out_overrides_the_configs_output_dir(aa: types.ModuleType) -> None:
    """``--out`` survives because a caller genuinely has a basis to choose where results land."""
    args = aa.parse_args(["/tmp/dummy.wav", "--out", "/tmp/somewhere"])
    assert str(args.out) == "/tmp/somewhere"


def test_a_removed_flag_is_rejected_rather_than_ignored(aa: types.ModuleType) -> None:
    """argparse errors on an unknown flag, so an old command line fails loudly.

    Worth pinning for the *grid* flags in particular: a run that accepted ``--asr-win-length`` and
    ignored it would report a grid it did not use.
    """
    for stale in ("--asr-win-length", "--speech-presence-grid-hop-length", "--cross-stream-win-length", "--policy"):
        with pytest.raises(SystemExit):
            aa.parse_args(["/tmp/dummy.wav", stale, "1.0"])
    for stale_switch in ("--no-cache", "--no-enhancement", "--skip"):
        with pytest.raises(SystemExit):
            aa.parse_args(["/tmp/dummy.wav", stale_switch])


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


def test_the_packaged_config_is_the_documented_default_run(aa: types.ModuleType) -> None:
    """The values that used to be argparse defaults are now config values, and still those values."""
    cfg = aa.load_run_config(None)
    assert cfg.device == "auto"
    assert cfg.cache_enabled is True
    assert cfg.align_asr is True
    assert cfg.mms_aligner_model == "facebook/mms-1b-all"
    assert cfg.asr_language is None
    assert "nyralabs/CrisperWhisper2.0_turbo" in cfg.asr_models
    assert "nvidia/canary-qwen-2.5b" in cfg.asr_models
    assert "Qwen/Qwen3-ASR-1.7B" in cfg.asr_models
    # Both scene classifiers on YAMNet's native frame. AST ran at 10.24 s until 2026-08-06, on the
    # reasoning that 1024 mel frames is "its native frame" — but that is its required *input size*,
    # not its temporal precision: `ASTFeatureExtractor` zero-pads a shorter window to 1024 frames
    # (rectangular, unattenuated), so AST can be slid at any hop. Pinned because the coarse setting
    # cost both resolution and confidence — 3 windows at Speech 0.47 became 45 at 0.75-0.92 — and on
    # a 4.9 s recording the window exceeded the clip, so AST returned one value for every bucket.
    assert (cfg.ast_win_length, cfg.ast_hop_length) == (0.96, 0.48)
    assert (cfg.yamnet_win_length, cfg.yamnet_hop_length) == (0.96, 0.48)
    assert cfg.aggregator == "min"
    assert cfg.disagreements_top_n == 100
    assert cfg.max_rounds == 3, "contracts/cli.md: 3 rounds including baseline"
    assert any("Narration" in label for label in cfg.speech_presence_labels)


def test_every_axis_is_configured_on_one_grid(aa: types.ModuleType) -> None:
    """There is one grid in the config, and it is the declared one.

    Four grid pairs used to be configurable independently, and their *defaults* disagreed: presence
    and mask at 0.1/0.02, speaker at 0.25/0.25, asr at 1.0/0.5.
    """
    from senselab.audio.workflows.audio_analysis.axes import DEFAULT_TIME_GRID

    cfg = aa.load_run_config(None)
    assert (cfg.grid_win_length, cfg.grid_hop_length) == DEFAULT_TIME_GRID
    assert cfg.grid_hop_length == cfg.grid_win_length, "window equals hop, so rows do not overlap"
    grid = aa._bucket_grid(cfg)
    assert (grid.win_length, grid.hop_length) == DEFAULT_TIME_GRID


def test_the_configs_identity_is_hashed_over_the_merged_mapping(aa: types.ModuleType, tmp_path: Path) -> None:
    """An override cannot inherit the packaged file's identity, or provenance names the wrong run."""
    packaged = aa.load_run_config(None)
    override = tmp_path / "config.yaml"
    override.write_text("uncertainty:\n  aggregator: mean\n")
    merged = aa.load_run_config(override)

    assert merged.aggregator == "mean"
    assert merged.disagreements_top_n == packaged.disagreements_top_n, "deep-merge must keep the rest"
    assert merged.identity.config_hash != packaged.identity.config_hash
    assert str(override) in merged.identity.sources
    assert aa.load_run_config(None).identity.config_hash == packaged.identity.config_hash


def test_an_invalid_config_is_refused_at_load(aa: types.ModuleType, tmp_path: Path) -> None:
    """Each relation is checked before the run spends inference on it, not at the point of use."""
    cases = {
        "grid.hop_length": "grid:\n  hop_length: 0.5\n  win_length: 0.1\n",
        "aggregator": "uncertainty:\n  aggregator: median\n",
        "same_floor": "speaker:\n  same_floor: 0.9\n  diff_floor: 0.2\n",
        "enhancement.mode": "enhancement:\n  mode: sometimes\n",
        "speech_presence_labels": "uncertainty:\n  speech_presence_labels: []\n",
        "max_rounds": "rounds:\n  max_rounds: 0\n",
    }
    for name, body in cases.items():
        path = tmp_path / f"{name.replace('.', '_')}.yaml"
        path.write_text(body)
        with pytest.raises(ValueError):
            aa.load_run_config(path)


def test_disabling_a_stage_reads_as_a_skip(aa: types.ModuleType, tmp_path: Path) -> None:
    """``stages.comparisons: false`` is what ``main`` gates the workflow call on."""
    path = tmp_path / "config.yaml"
    path.write_text("stages:\n  comparisons: false\n")
    cfg = aa.load_run_config(path)
    assert "comparisons" in cfg.skipped_stages
    assert cfg.run_comparisons is False
    assert "asr" not in aa.load_run_config(None).skipped_stages


# ── config → library adapters ─────────────────────────────────────────
#     These exist because the adapters are the one place field names are read by hand — a typo there
#     is invisible to the library tests and only shows up at runtime (it did: `args.align_asr` vs
#     `no_align_asr`).


def test_pass_plan_reads_every_field_it_needs(aa: types.ModuleType) -> None:
    """_pass_plan must not reference a config field that doesn't exist."""
    plan = aa._pass_plan(aa.load_run_config(None))
    assert plan.asr_models, "defaults should populate ASR models"
    assert plan.align_asr is True, "alignment is on unless stages.align_asr is false"
    from senselab.audio.workflows.audio_analysis.axes import DEFAULT_TIME_GRID

    assert (plan.mask_grid.win_length, plan.mask_grid.hop_length) == DEFAULT_TIME_GRID


def test_pass_plan_honors_a_disabled_align_stage(aa: types.ModuleType, tmp_path: Path) -> None:
    """``stages.align_asr: false`` reaches the plan."""
    path = tmp_path / "config.yaml"
    path.write_text("stages:\n  align_asr: false\n")
    assert aa._pass_plan(aa.load_run_config(path)).align_asr is False


def test_pass_plan_translates_skip_into_absence(aa: types.ModuleType, tmp_path: Path) -> None:
    """The library has no skip set — skipping is empty tuples and None ids."""
    path = tmp_path / "config.yaml"
    path.write_text("stages:\n  diarization: false\n  asr: false\n  ast: false\n  features: false\n")
    plan = aa._pass_plan(aa.load_run_config(path))
    assert plan.diarization_models == ()
    assert plan.asr_models == ()
    assert plan.ast_model is None
    assert plan.features is False


def test_pass_plan_reflects_the_post_triage_config(aa: types.ModuleType) -> None:
    """Built AFTER triage: a no-speech clip must not run diarization or ASR.

    Triage returns a *new* config with the skip set widened, rather than mutating one every later
    stage has already read — so "what was configured" and "what the audio justified" stay distinct.
    """
    cfg = aa.load_run_config(None).with_skipped({"diarization", "asr", "features"})
    plan = aa._pass_plan(cfg)
    assert plan.diarization_models == () and plan.asr_models == ()
    assert aa.load_run_config(None).diarization_models, "the original config is untouched"


def test_stage_context_carries_provenance_fields(aa: types.ModuleType, tmp_path: Path) -> None:
    """The context must record the resolved source path and the perturbation it ran under."""
    import torch

    from senselab.audio.data_structures import Audio
    from senselab.audio.workflows.audio_analysis.perturbations import identity

    audio = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000)
    ctx = aa._stage_context(
        identity(), audio, Path("x.wav"), device=None, out_dir=tmp_path, cache_dir=None, senselab_ver="v"
    )
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

    audio = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000)
    perturbation = speech_enhancement("speechbrain/sepformer-wham16k-enhancement", name="sepformer")
    ctx = aa._stage_context(
        perturbation, audio, Path("x.wav"), device=None, out_dir=tmp_path, cache_dir=None, senselab_ver="v"
    )
    assert ctx.perturbation == "sepformer"
    assert ctx.variant == "speech_enhanced"
    assert ctx.out_dir == tmp_path / "L1" / "perturbation" / "sepformer"


def test_the_adaptive_policy_is_a_section_of_the_same_config(aa: types.ModuleType, tmp_path: Path) -> None:
    """One file, two identities: the config's hash and the policy's own.

    The budget / region / reserve-pool flags that used to override the policy from the command line
    are gone; those values are the policy, and the policy is this file.
    """
    from senselab.audio.workflows.audio_analysis.adaptive.policy import load_policy

    cfg = aa.load_run_config(None)
    assert cfg.adaptive["budget"]["heavy_per_run"] == 4
    assert cfg.adaptive["rules"]["I4_overlap_detection"]["enabled"] is True

    override = tmp_path / "config.yaml"
    override.write_text("adaptive:\n  budget:\n    heavy_per_run: 0\n")
    assert load_policy(override)["budget"]["heavy_per_run"] == 0
    assert load_policy(override)["policy_hash"] != load_policy()["policy_hash"]
