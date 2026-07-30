"""StageContext / PassPlan / STAGE_VERSIONS contract tests (T051 step 4).

The highest-value test here is `test_provenance_joins_to_build_cache_index`: the
adaptive loop indexes cached results on `provenance.audio_signature`, and a
mismatch against `summary.json` makes cache-replay escalation silently never
fire — no error, no log line.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from senselab.audio.workflows.audio_analysis.stage_context import (
    STAGE_VERSIONS,
    PassPlan,
    StageContext,
    stage_code_version,
)
from senselab.utils.tasks.cached_inference import audio_signature, cache_store


def _ctx(**kwargs: object) -> StageContext:
    base: dict[str, object] = {"pass_label": "raw_16k", "audio_signature": "a" * 64, "senselab_ver": "1.2.3"}
    base.update(kwargs)
    return StageContext(**base)  # type: ignore[arg-type]


# ── STAGE_VERSIONS ────────────────────────────────────────────────────


def test_stage_versions_are_pinned() -> None:
    """A bump invalidates that stage's cache, so it should be a visible diff."""
    assert dict(STAGE_VERSIONS) == {
        "diarization": 1,
        "ast": 1,
        "yamnet": 1,
        "features": 1,
        "asr": 1,
        "alignment": 1,
        "ppgs": 1,
        # Scene-quality level work. Each declares its own counter rather than
        # borrowing another stage's invalidation fate.
        "background_mask": 1,
        "noise_floor": 1,
        "background_sources": 1,
        "level_probe": 1,
    }


def test_stage_versions_is_immutable() -> None:
    """Nothing may mutate the table at runtime — keys must be reviewable in git."""
    with pytest.raises(TypeError):
        STAGE_VERSIONS["asr"] = 99  # type: ignore[index]


def test_stage_code_version_is_self_describing() -> None:
    """`cat`-ing a cache entry should tell you which stage version wrote it."""
    assert stage_code_version("asr") == "asr@1"


def test_unknown_stage_raises_rather_than_defaulting() -> None:
    """A new stage must declare a version, not inherit another stage's fate."""
    with pytest.raises(KeyError, match="STAGE_VERSIONS"):
        stage_code_version("brand_new_stage")


# ── device_label ──────────────────────────────────────────────────────


def test_device_label_is_auto_when_unset() -> None:
    """None → "auto". It's inside the cache key and the provenance."""
    assert _ctx().device_label == "auto"


def test_device_label_uses_the_enum_value() -> None:
    """A concrete device reports its senselab value."""
    from senselab.utils.data_structures import DeviceType

    assert _ctx(device=DeviceType.CPU).device_label == "cpu"


# ── keys ──────────────────────────────────────────────────────────────


def test_cache_key_is_stable_and_stage_scoped() -> None:
    """Same call → same key; a different task → a different key."""
    ctx = _ctx()
    first = ctx.cache_key_for("asr", "whisper", {"device": "auto"})
    assert first == ctx.cache_key_for("asr", "whisper", {"device": "auto"})
    assert first != ctx.cache_key_for("diarization", "whisper", {"device": "auto"})


def test_cache_key_tracks_the_audio_signature() -> None:
    """Different audio must never replay another clip's result."""
    a = _ctx(audio_signature="a" * 64).cache_key_for("asr", "m", {})
    b = _ctx(audio_signature="b" * 64).cache_key_for("asr", "m", {})
    assert a != b


def test_align_key_differs_from_the_task_key() -> None:
    """Alignment keying stays independent of the ASR cache."""
    ctx = _ctx()
    align = ctx.align_key_for(
        transcript_sha="c" * 64, language="en", aligner_model_id="facebook/mms-1b-all", aligner_params={}
    )
    assert align != ctx.cache_key_for("alignment", "facebook/mms-1b-all", {})


def test_align_key_tracks_the_transcript() -> None:
    """A changed transcript must re-align rather than replay stale timestamps."""
    ctx = _ctx()
    kwargs = {"language": "en", "aligner_model_id": "facebook/mms-1b-all", "aligner_params": {}}
    assert ctx.align_key_for(transcript_sha="a", **kwargs) != ctx.align_key_for(transcript_sha="b", **kwargs)  # type: ignore[arg-type]


# ── provenance ────────────────────────────────────────────────────────


def test_provenance_records_the_stage_code_version() -> None:
    """The whole point of STAGE_VERSIONS is that a stale replay is diagnosable."""
    prov = _ctx().provenance_for("asr", "whisper", {"device": "auto"})
    assert prov["code_version"] == "asr@1"
    assert prov["cache_schema_version"] == 4
    assert prov["pass"] == "raw_16k"
    assert prov["device"] == "auto"


def test_provenance_joins_to_build_cache_index(tmp_path: Path) -> None:
    """summary.json's audio_signature must resolve in the adaptive cache index.

    `adaptive/loop.py` reads summary["passes"][label]["audio_signature"];
    `build_cache_index` keys entries on provenance.audio_signature. If the two
    ever diverge the index misses silently and U2 escalation never fires.
    """
    from senselab.audio.workflows.audio_analysis.adaptive.interventions import build_cache_index

    audio = SimpleNamespace(waveform=torch.tensor([[0.1, 0.2, 0.3]]), sampling_rate=16000)
    sig = audio_signature(audio)
    ctx = StageContext(pass_label="raw_16k", audio_signature=sig, cache_dir=tmp_path, senselab_ver="v")

    key = ctx.cache_key_for("asr", "openai/whisper-tiny", {})
    cache_store(
        tmp_path,
        key,
        {"status": "ok", "result": [], "provenance": ctx.provenance_for("asr", "openai/whisper-tiny", {})},
    )

    index = build_cache_index(tmp_path)
    assert (sig, "asr", "openai/whisper-tiny") in index, f"index keys: {list(index)[:3]}"


# ── sidecars ──────────────────────────────────────────────────────────


def test_write_sidecar_creates_nested_paths(tmp_path: Path) -> None:
    """Stages write per-model sidecars under the pass dir."""
    _ctx(out_dir=tmp_path).write_sidecar(Path("diarization") / "m.json", {"a": 1})
    assert (tmp_path / "diarization" / "m.json").exists()


def test_write_sidecar_is_a_noop_without_out_dir(tmp_path: Path) -> None:
    """Headless mode for the adaptive loop: no out_dir → no files, no error."""
    _ctx(out_dir=None).write_sidecar("x.json", {"a": 1})
    assert not list(tmp_path.iterdir())


# ── PassPlan ──────────────────────────────────────────────────────────


def test_pass_plan_defaults_to_running_nothing_expensive() -> None:
    """Absence means skip — an empty plan must not imply "run every model"."""
    plan = PassPlan()
    assert plan.diarization_models == () and plan.asr_models == ()
    assert plan.ast_model is None and plan.yamnet_model is None
    assert plan.features is False and plan.ppg is False


def test_pass_plan_is_frozen() -> None:
    """Immutable so a plan can't be mutated mid-pass (the args.skip bug class)."""
    with pytest.raises(Exception):  # noqa: B017 — dataclasses raise FrozenInstanceError
        PassPlan().features = True  # type: ignore[misc]


def test_stage_context_is_frozen() -> None:
    """Same reasoning: the run environment must not drift between stages."""
    with pytest.raises(Exception):  # noqa: B017
        _ctx().pass_label = "other"  # type: ignore[misc]


# ── import weight ─────────────────────────────────────────────────────


def test_stage_context_import_stays_light() -> None:
    """Importing the config types must not drag in torch/transformers.

    `DeviceType` is behind TYPE_CHECKING precisely for this: a caller that only
    wants a cache key shouldn't pay for the ML stack. Run in a subprocess because
    the parent test session has already imported everything.
    """
    code = (
        "import sys; "
        "import senselab.audio.workflows.audio_analysis.stage_context as m; "
        "print('transformers' in sys.modules, 'torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    transformers_loaded, torch_loaded = out.stdout.strip().split()
    assert transformers_loaded == "False", "stage_context pulled in transformers"
    assert torch_loaded == "False", "stage_context pulled in torch"
