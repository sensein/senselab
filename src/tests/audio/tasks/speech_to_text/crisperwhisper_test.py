"""Tests for the CrisperWhisper 2.0 subprocess-venv backend.

The assembly test is hermetic (worker + venv mocked): it verifies the
worker-output → ScriptLine mapping, including per-word / line ``score`` and the
line span derived from word timestamps. The conversion-cache tests are pure
filesystem tests over ``tmp_path``. The integration test runs the real model
only when the ``crisperwhisper`` venv is already provisioned (skipped in
default CI).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import senselab.audio.tasks.speech_to_text.crisperwhisper as cw
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
from senselab.utils.data_structures import HFModel
from senselab.utils.subprocess_venv import provisioned_venv_dirs

REPO_ROOT = Path(__file__).resolve().parents[5]
FIXTURE_WAV = REPO_ROOT / "src" / "tests" / "data_for_testing" / "audio_48khz_mono_16bits.wav"
CRISPER_VENVS = provisioned_venv_dirs("crisperwhisper")


def test_worker_output_maps_to_scriptlines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake worker output assembles into a ScriptLine with word chunks + scores."""
    # model=None now still constructs a default HFModel (to get a resolved commit_sha for
    # staging), so both the constructor's Hub validation and the module's own
    # resolve_model call (which would otherwise download the real snapshot) are faked.
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "f" * 40)
    monkeypatch.setattr(cw, "resolve_model", lambda *a, **k: ("f" * 40, Path("/fake/snapshot")))
    monkeypatch.setattr(cw, "ensure_venv", lambda *a, **k: "/fake/venv")
    monkeypatch.setattr(cw, "venv_python", lambda *a, **k: "/fake/venv/bin/python")
    monkeypatch.setattr(
        cw,
        "serve_in_venv",
        lambda *a, **k: {
            "results": [
                {
                    "text": "This is Peter",
                    "language": "en",
                    "score": 0.9,
                    "words": [
                        {"text": "This", "start": 0.0, "end": 0.2, "score": 0.95},
                        {"text": "is", "start": 0.2, "end": 0.3, "score": None},
                        {"text": "Peter", "start": 0.3, "end": 0.9, "score": 0.8},
                    ],
                }
            ]
        },
    )
    audio = Audio(waveform=torch.zeros(1, 16000, dtype=torch.float32), sampling_rate=16000)
    # model=None now builds a default HFModel internally (see the mocks above).
    out = cw.CrisperWhisperASR.transcribe_with_crisperwhisper([audio], model=None)

    assert len(out) == 1
    sl = out[0]
    assert sl.text == "This is Peter"
    assert sl.score == 0.9
    assert sl.chunks is not None and len(sl.chunks) == 3
    assert sl.chunks[0].text == "This" and sl.chunks[0].score == 0.95
    assert sl.chunks[1].score is None  # native confidence absent for a word → None
    assert sl.start == 0.0 and sl.end == 0.9


def _stub_out_staging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the backend at a fake venv + snapshot so no Hub or venv work happens."""
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "f" * 40)
    monkeypatch.setattr(cw, "resolve_model", lambda *a, **k: ("f" * 40, Path("/fake/snapshot")))
    monkeypatch.setattr(cw, "ensure_venv", lambda *a, **k: "/fake/venv")
    monkeypatch.setattr(cw, "venv_python", lambda *a, **k: "/fake/venv/bin/python")


def test_ct2_position_limit_becomes_a_typed_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """CTranslate2's 448-position overrun is re-raised as a ValueError, so it records as an absence."""
    _stub_out_staging(monkeypatch)

    def _raise(*a: object, **k: object) -> dict:
        raise RuntimeError("No position encodings are defined for positions >= 448, but got position 448")

    monkeypatch.setattr(cw, "serve_in_venv", _raise)
    audio = Audio(waveform=torch.zeros(1, 16000, dtype=torch.float32), sampling_rate=16000)

    with pytest.raises(cw.CrisperWhisperDecoderPositionsExceeded) as caught:
        cw.CrisperWhisperASR.transcribe_with_crisperwhisper([audio], model=None)
    assert isinstance(caught.value, ValueError)
    assert "448" in str(caught.value)


def test_other_worker_failures_stay_hard(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unrelated worker RuntimeError is not reclassified as an absence."""
    _stub_out_staging(monkeypatch)

    def _raise(*a: object, **k: object) -> dict:
        raise RuntimeError("CrisperWhisper 2.0 venv failed:\nCUDA out of memory")

    monkeypatch.setattr(cw, "serve_in_venv", _raise)
    audio = Audio(waveform=torch.zeros(1, 16000, dtype=torch.float32), sampling_rate=16000)

    with pytest.raises(RuntimeError) as caught:
        cw.CrisperWhisperASR.transcribe_with_crisperwhisper([audio], model=None)
    assert not isinstance(caught.value, ValueError)


_FAKE_PROMPT = """
class PromptBuilder:
    def _build(self, mode, hotwords=None, context=None):
        return list(range(4 + 10 * len((context or "").split())))
"""

_FAKE_LIBRARY = """
import os

from crisperwhisper.prompt import PromptBuilder


class _Word:
    def __init__(self, word, start, end):
        self.word, self.start, self.end, self.probability = word, start, end, 0.5


class _Result:
    def __init__(self, text):
        self.text, self.language, self.words = text, "en", [_Word(text, 0.0, 0.5)]


class CrisperWhisperModel:
    def __init__(self, model_id, backend, device, compute_type):
        pass

    def transcribe(self, path, language, word_timestamps, longform_strategy, max_new_tokens):
        name = os.path.basename(path)
        if name.startswith("broken"):
            raise RuntimeError("CUDA out of memory")
        context = "w " * 30 if name.startswith("overrun") else "w"
        prompt = PromptBuilder()._build("verbatim", context=context)
        if len(prompt) + max_new_tokens >= 448:
            raise RuntimeError("No position encodings are defined for positions >= 448, but got position 448")
        return _Result(str(len(prompt)))
"""


def _run_worker(tmp_path: Path, names: list[str]) -> dict:
    """Serve the real worker script against a fake ``crisperwhisper`` library."""
    import sys

    from senselab.utils.venv_worker import serve_in_venv, shutdown_venv_workers

    library = tmp_path / "lib" / "crisperwhisper"
    library.mkdir(parents=True)
    (library / "__init__.py").write_text(_FAKE_LIBRARY)
    (library / "prompt.py").write_text(_FAKE_PROMPT)
    try:
        return serve_in_venv(
            ("crisperwhisper-test", str(tmp_path)),
            python=sys.executable,
            script=cw._CRISPER_WORKER_SCRIPT,
            init={"model_id": "/fake/snapshot", "backend": "transformers", "device": "cpu", "compute_type": "float32"},
            request={
                "audio_paths": [str(tmp_path / name) for name in names],
                "longform_strategy": cw.LONGFORM_STRATEGY,
                "max_new_tokens": cw.MAX_NEW_TOKENS,
                "prompt_token_budget": cw.DECODER_POSITIONS - cw.MAX_NEW_TOKENS - 1,
                "capped_strategy": cw.CONTEXT_CAPPED,
                "position_limit": cw._CT2_POSITION_LIMIT,
            },
            env={"PYTHONPATH": str(tmp_path / "lib")},
            label="CrisperWhisper test",
            load_timeout_s=60,
            request_timeout_s=60,
        )
    finally:
        shutdown_venv_workers()


def test_a_position_overrun_is_decoded_again_with_its_context_capped(tmp_path: Path) -> None:
    """Only the overrunning input is decoded again, its prompt cut within the budget; the patch is undone."""
    out = _run_worker(tmp_path, ["overrun.wav", "fine.wav", "overrun_again.wav"])
    assert [entry["decode_strategy"] for entry in out["results"]] == [
        "continuation_context_capped",
        "continuation",
        "continuation_context_capped",
    ]
    capped = int(out["results"][0]["text"])
    assert capped + cw.MAX_NEW_TOKENS < cw.DECODER_POSITIONS
    assert out["results"][1]["text"] == "14", "an input under the budget keeps its whole context"


def test_any_other_worker_error_is_not_redecoded(tmp_path: Path) -> None:
    """The fallback is keyed on CTranslate2's position-limit message alone."""
    with pytest.raises(RuntimeError, match="out of memory"):
        _run_worker(tmp_path, ["broken.wav"])


def test_the_strategy_used_reaches_the_scriptline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The host carries the worker's ``decode_strategy`` onto the line it returns."""
    _stub_out_staging(monkeypatch)
    monkeypatch.setattr(
        cw,
        "serve_in_venv",
        lambda *a, **k: {"results": [{"text": "a", "words": [], "decode_strategy": "continuation_context_capped"}]},
    )
    audio = Audio(waveform=torch.zeros(1, 16000, dtype=torch.float32), sampling_rate=16000)
    [line] = cw.CrisperWhisperASR.transcribe_with_crisperwhisper([audio], model=None)
    assert line.decode_strategy == "continuation_context_capped"


def test_ct2_cache_key_matches_the_library_layout() -> None:
    """The computed key is the directory name the library's converter writes."""
    snapshot = (
        "/orcd/data/satra/002/huggingface/hub/models--nyralabs--CrisperWhisper2.0_turbo"
        "/snapshots/de0369c8a68025b7f6e86387b6eb5a3b369787c8"
    )
    assert cw._ct2_cache_key(snapshot, "float32") == (
        "--orcd--data--satra--002--huggingface--hub--models--nyralabs--CrisperWhisper2.0_turbo"
        "--snapshots--de0369c8a68025b7f6e86387b6eb5a3b369787c8_float32_6794fe16e2f2"
    )
    assert cw._ct2_cache_key(snapshot, "float16").endswith("_float16_6794fe16e2f2")


def test_torn_ct2_entry_is_discarded(tmp_path: Path) -> None:
    """A cache entry stamped complete without weights is torn, and is deleted."""
    entry = tmp_path / "model_float32_abc"
    entry.mkdir()
    (entry / ".conversion_complete").touch()
    (entry / "config.json").write_text("{}")

    assert cw._ct2_entry_is_torn(entry) is True
    assert cw._discard_torn_ct2_entry(entry) is True
    assert not entry.exists()
    assert list(tmp_path.iterdir()) == []


def test_complete_ct2_entry_is_kept(tmp_path: Path) -> None:
    """A cache entry carrying weights is left alone."""
    entry = tmp_path / "model_float32_abc"
    entry.mkdir()
    (entry / ".conversion_complete").touch()
    (entry / "model.bin").write_bytes(b"weights")

    assert cw._ct2_entry_is_torn(entry) is False
    assert cw._discard_torn_ct2_entry(entry) is False
    assert (entry / "model.bin").read_bytes() == b"weights"


def test_backend_selection_is_platform_appropriate() -> None:
    """CT2 on Linux x86_64, transformers elsewhere (both valid crisperwhisper backends)."""
    assert cw._CRISPER_BACKEND in ("ct2", "transformers")
    if cw._IS_LINUX_X86:
        assert cw._CRISPER_BACKEND == "ct2"
    else:
        assert cw._CRISPER_BACKEND == "transformers"


@pytest.mark.skipif(not CRISPER_VENVS, reason="crisperwhisper venv not provisioned for this host's device key")
def test_crisperwhisper_transcribes_when_venv_present() -> None:
    """Integration: real model yields verbatim text + word-level chunks (shape only)."""
    audio = Audio(filepath=str(FIXTURE_WAV))
    audio = downmix_audios_to_mono([audio])[0]
    if audio.sampling_rate != 16000:
        audio = resample_audios([audio], resample_rate=16000)[0]
    out = cw.CrisperWhisperASR.transcribe_with_crisperwhisper(
        [audio], model=HFModel(path_or_uri="nyralabs/CrisperWhisper2.0_turbo")
    )
    assert len(out) == 1
    line = out[0]
    assert isinstance(line.text, str) and line.text
    chunks = line.chunks or []
    assert len(chunks) >= 1
    assert chunks[0].start is not None and chunks[0].end is not None
