"""The synthetic multi-speaker corpus generator: reproducibility and ground-truth guarantees.

No test here synthesizes audio or builds the Qwen3-TTS subprocess venv -- ``synthesize_texts_with_qwen``
is replaced with a deterministic stub for every test, and :func:`test_never_touches_the_real_tts_venv`
asserts ``ensure_venv`` is never reached. The one test that would exercise real synthesis is marked
``skip`` with the reason stated, per the task's requirement that any real run stays gated and explicit.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pytest
import torch

from senselab.audio.data_structures import Audio

# `scripts/` is deliberately not an importable package -- pyproject sets `pythonpath = ["src"]`,
# so the repo root is not on sys.path. Load by file location instead, the convention
# speaker_ceiling_derive_test.py (Task 1) already established for the same reason.
_GENERATE = Path(__file__).resolve().parents[3] / "scripts" / "speaker_ceiling" / "generate.py"
_spec = importlib.util.spec_from_file_location("speaker_ceiling_generate_under_test", _GENERATE)
assert _spec is not None and _spec.loader is not None, f"could not load {_GENERATE}"
_generate = importlib.util.module_from_spec(_spec)
sys.modules["speaker_ceiling_generate_under_test"] = _generate
_spec.loader.exec_module(_generate)

_VOICE_POOL = ("aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian")


def _fake_synthesize_texts_with_qwen(
    texts: List[str],
    model: object = None,
    language: Union[str, List[str]] = "Auto",
    speaker: Optional[Union[str, List[str]]] = None,
    instruct: Optional[Union[str, List[str]]] = None,
    device: object = None,
) -> List[Audio]:
    """Stand in for real TTS: one waveform per text, length a pure function of word count.

    Deterministic and side-effect-free by construction -- the same text always produces the
    same duration -- which is exactly what the reproducibility guarantee needs: two runs
    with the same seed must plan the same texts, and this stub then must turn identical
    texts into identical durations.
    """
    sample_rate = 24000
    audios = []
    for text in texts:
        n_samples = max(1, len(text.split())) * int(0.2 * sample_rate)
        waveform = torch.zeros((1, n_samples), dtype=torch.float32)
        audios.append(Audio(waveform=waveform, sampling_rate=sample_rate))
    return audios


@pytest.fixture(autouse=True)
def _mock_hf_and_tts(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may construct an unmocked HFModel or reach the real TTS backend.

    ``check_hf_repo_exists``/``resolve_revision`` are patched at the modules that actually
    define them (not re-exported names), matching this repo's established convention --
    patching the re-exported name in ``generate.py`` would miss the deferred
    ``from senselab.utils.model_revision import resolve_revision`` import inside
    ``HFModel._resolve_commit_sha``.
    """
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "f" * 40)
    monkeypatch.setattr(_generate, "supported_speakers", lambda model=None: list(_VOICE_POOL))
    monkeypatch.setattr(_generate, "synthesize_texts_with_qwen", _fake_synthesize_texts_with_qwen)


def _rttm_speakers(path: Path) -> set:
    return {line.split()[7] for line in path.read_text().splitlines() if line.strip()}


def test_enforce_num_speakers_is_guaranteed_by_construction(tmp_path: Path) -> None:
    """Every session's written RTTM contains exactly k distinct speakers, for several k.

    This is the hard requirement the whole probe rests on: a session requested at k=8 that
    silently contains fewer distinct speakers would corrupt the ground truth every later
    ceiling is measured against. Checked against the file on disk, not the in-memory plan,
    since that is what the evaluation script will actually read.
    """
    out_dir = _generate.generate_corpus(out_dir=tmp_path / "corpus", counts=[1, 2, 5, 8], sessions_per_count=2, seed=17)
    for k in (1, 2, 5, 8):
        for i in range(2):
            rttm_path = out_dir / f"k={k}" / f"session_{i}.rttm"
            assert rttm_path.exists()
            speakers = _rttm_speakers(rttm_path)
            assert len(speakers) == k, f"k={k} session_{i}: found {len(speakers)} distinct speakers, wanted {k}"


def test_corpus_is_reproducible_from_the_seed(tmp_path: Path) -> None:
    """Same seed, same (mocked) synthesis -> byte-identical RTTM and manifest content.

    A re-run that is not comparable to the profile it produced would make every measured
    ceiling unreproducible -- this is the property the whole probe depends on.
    """
    out_a = _generate.generate_corpus(out_dir=tmp_path / "a", counts=[1, 3], sessions_per_count=2, seed=42)
    out_b = _generate.generate_corpus(out_dir=tmp_path / "b", counts=[1, 3], sessions_per_count=2, seed=42)

    for k in (1, 3):
        for i in range(2):
            rttm_a = (out_a / f"k={k}" / f"session_{i}.rttm").read_text()
            rttm_b = (out_b / f"k={k}" / f"session_{i}.rttm").read_text()
            assert rttm_a == rttm_b

            wav_a = Audio(filepath=str(out_a / f"k={k}" / f"session_{i}.wav"))
            wav_b = Audio(filepath=str(out_b / f"k={k}" / f"session_{i}.wav"))
            assert torch.equal(wav_a.waveform, wav_b.waveform)

    manifest_a = json.loads((out_a / "manifest.json").read_text())
    manifest_b = json.loads((out_b / "manifest.json").read_text())
    # Paths are recorded relative to each corpus's own root, so they match regardless of
    # the tmp_path prefix; everything else -- seed, session params, per-session structure
    # -- must be identical between the two runs.
    assert manifest_a == manifest_b


def test_a_different_seed_produces_a_different_corpus(tmp_path: Path) -> None:
    """Sanity check on the reproducibility test itself: seed must actually matter.

    Without this, test_corpus_is_reproducible_from_the_seed could pass vacuously if
    generate_corpus ignored its seed argument entirely.
    """
    out_a = _generate.generate_corpus(out_dir=tmp_path / "a", counts=[3], sessions_per_count=2, seed=1)
    out_b = _generate.generate_corpus(out_dir=tmp_path / "b", counts=[3], sessions_per_count=2, seed=2)

    rttm_a = (out_a / "k=3" / "session_0.rttm").read_text()
    rttm_b = (out_b / "k=3" / "session_0.rttm").read_text()
    assert rttm_a != rttm_b


def test_rejects_k_beyond_the_voice_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A k the voice pool cannot cover is refused outright, not silently truncated or cloned.

    This generator assigns one distinct voice per speaker and does not clone additional
    identities from reference audio -- so a request beyond the pool size cannot be honored
    at all, and honoring it partially would silently corrupt the ground truth (fewer
    distinct voices than the RTTM would claim).
    """
    monkeypatch.setattr(_generate, "supported_speakers", lambda model=None: ["only_one", "only_two"])
    with pytest.raises(ValueError, match="exposes only 2 named voices"):
        _generate.generate_corpus(out_dir=tmp_path / "corpus", counts=[3], sessions_per_count=1, seed=1)


def test_rejects_a_count_below_one(tmp_path: Path) -> None:
    """k=0 or negative is not a speaker count; refuse rather than emit a degenerate session."""
    with pytest.raises(ValueError, match="speaker count must be >= 1"):
        _generate.generate_corpus(out_dir=tmp_path / "corpus", counts=[0], sessions_per_count=1, seed=1)


def test_manifest_records_method_model_and_session_params(tmp_path: Path) -> None:
    """The manifest records generation method, resolved TTS model, seed, and session params.

    These are the fields a reader of a later profile needs to judge the ceiling it produced:
    the method (so a constructive-vs-estimated ground truth is not confused for the other),
    the exact model pin (so "which checkpoint" is never ambiguous), and the session
    parameters with their source (so a reader can tell NeMo's own defaults apart from this
    module's judgement calls) -- per the task's explicit requirement that judgements be
    recorded as such, not buried as bare literals.
    """
    out_dir = _generate.generate_corpus(out_dir=tmp_path / "corpus", counts=[2], sessions_per_count=1, seed=99)
    manifest = json.loads((out_dir / "manifest.json").read_text())

    assert manifest["seed"] == 99
    assert manifest["counts"] == [2]
    assert manifest["sessions_per_count"] == 1
    assert "tts-composed" in manifest["method"]
    assert manifest["tts_model"]["resolved_commit_sha"] == "f" * 40
    assert set(manifest["tts_model"]["voice_pool"]) == set(_VOICE_POOL)

    params = manifest["session_params"]
    assert params["turn_prob"] == _generate.TURN_PROB
    assert params["dominance_var"] == _generate.DOMINANCE_VAR
    assert params["enforce_num_speakers"] is True
    assert "judgement" in params["params_source"]

    assert len(manifest["sessions"]) == 1
    session = manifest["sessions"][0]
    assert session["k"] == 2
    assert len(session["speakers"]) == 2
    assert session["wav"] == "k=2/session_0.wav"
    assert session["rttm"] == "k=2/session_0.rttm"


def test_never_touches_the_real_tts_venv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real subprocess-venv machinery is never reached while synthesis is mocked.

    Mocking `synthesize_texts_with_qwen` wholesale (as every other test here does) should
    make `ensure_venv` unreachable, since that call lives entirely inside the real
    function. Asserted directly rather than assumed, so a future refactor that calls
    `ensure_venv` from somewhere else in this module would be caught here.
    """

    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("ensure_venv must never be called from a mocked-synthesis test")

    monkeypatch.setattr("senselab.utils.subprocess_venv.ensure_venv", _explode)
    _generate.generate_corpus(out_dir=tmp_path / "corpus", counts=[2], sessions_per_count=1, seed=1)


@pytest.mark.skip(
    reason=(
        "Exercises real Qwen3-TTS synthesis, which downloads a 1.7B checkpoint and builds a "
        "subprocess venv -- both explicitly disallowed in this suite. Run manually on a GPU "
        "host with network access, e.g.: "
        "uv run python scripts/speaker_ceiling/generate.py --out /tmp/ceiling-smoke "
        "--counts 2 --sessions 2 --seed 17"
    )
)
def test_real_generation_end_to_end() -> None:
    """Placeholder documenting the manual/cluster smoke test this suite deliberately skips."""
    _generate.generate_corpus(out_dir=Path("/tmp/ceiling-smoke"), counts=[2], sessions_per_count=2, seed=17)


def test_dominance_cdf_sums_to_one_and_is_nondecreasing() -> None:
    """The dominance CDF must be a real CDF: nondecreasing and ending at 1.

    `_next_speaker` samples from this by inverse-CDF lookup -- if it did not reach 1, the
    lookup could run off the end of the array for a draw close to 1.0.
    """
    rng = np.random.default_rng(0)
    cdf = _generate._dominance_cdf(rng, num_speakers=5)
    assert cdf[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(cdf, cdf[1:]))
    # Every speaker's share (the CDF's own increments) is at least MIN_DOMINANCE.
    shares = np.diff(np.concatenate(([0.0], cdf)))
    assert all(share >= _generate.MIN_DOMINANCE - 1e-9 for share in shares)


def test_plan_session_covers_every_speaker_in_the_first_k_turns() -> None:
    """The enforcement guarantee lives here: the first k planned turns are a permutation of 0..k-1.

    This is what makes enforce_num_speakers true by construction rather than by a
    probabilistic near-certainty the rest of the module merely hopes holds.
    """
    rng = np.random.default_rng(7)
    k = 6
    plan = _generate._plan_session(rng, num_speakers=k)
    first_k_speakers = {speaker_idx for speaker_idx, _ in plan[:k]}
    assert first_k_speakers == set(range(k))


def test_lay_out_session_never_starts_a_turn_before_the_session_begins() -> None:
    """Every placed turn starts at or after 0.0, even when the overlap draw always fires."""
    rng = np.random.default_rng(3)
    plan: Sequence[tuple] = [(0, "a"), (1, "b"), (0, "c")]
    durations = [0.5, 0.3, 0.4]
    turns = _generate._lay_out_session(plan, durations, rng)
    assert all(turn.start >= 0.0 for turn in turns)
    assert all(turn.end > turn.start for turn in turns)
