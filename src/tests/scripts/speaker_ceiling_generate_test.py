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
from collections import defaultdict
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

    Renders at exactly ``ASSUMED_WORDS_PER_SECOND``, not an arbitrary rate: `_plan_session`'s
    ``MIN_SPEAKER_SPEECH_SECONDS`` floor is a *planned*-word estimate built on that constant
    (see the module docstring), and `generate_corpus` re-verifies it against each session's
    *real* synthesized duration. A stub rendering faster than that assumption (an earlier
    version of this stub used 5 words/s against the planner's 2.5) would make every test in
    this file trip that re-verification, since a planned floor built on a slower assumed rate
    is not guaranteed to survive being rendered at a faster real one -- exactly the gap the
    module docstring calls a probabilistic, not deterministic, guarantee. Matching the rate
    here keeps that real-hardware question out of tests that are not about it.
    """
    sample_rate = 24000
    audios = []
    for text in texts:
        words = max(1, len(text.split()))
        n_samples = max(1, int(round(words / _generate.ASSUMED_WORDS_PER_SECOND * sample_rate)))
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
    assert params["overlap_trigger_prob"] == _generate.OVERLAP_TRIGGER_PROB
    assert params["min_speaker_speech_seconds"] == _generate.MIN_SPEAKER_SPEECH_SECONDS
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


def test_a_fully_resumed_run_never_calls_synthesis_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling generate_corpus a second time over a fully-complete corpus touches no TTS at all.

    Proves the skip branch actually skips the expensive part, not just that it takes some
    different code path: `synthesize_texts_with_qwen` is swapped for a version that fails
    the test outright if called, then the exact same generate_corpus call is repeated.
    """
    out_dir = tmp_path / "corpus"
    _generate.generate_corpus(out_dir=out_dir, counts=[1, 3], sessions_per_count=2, seed=11)

    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("synthesis must not run when every session is already complete")

    monkeypatch.setattr(_generate, "synthesize_texts_with_qwen", _explode)
    _generate.generate_corpus(out_dir=out_dir, counts=[1, 3], sessions_per_count=2, seed=11)


def test_resuming_after_partial_deletion_matches_an_uninterrupted_run(tmp_path: Path) -> None:
    """A run interrupted mid-corpus and resumed produces byte-identical output to one that never was.

    Simulates the two shapes a task killed mid-write leaves behind (per _session_is_complete's
    docstring): one session loses its wav entirely, another's rttm is truncated to zero bytes.
    Only those two sessions should be regenerated; every file -- regenerated or left alone --
    must end up identical to an uninterrupted reference run, and the manifest (including the
    reconstructed sessions' "speakers" field) must match exactly.
    """
    reference = _generate.generate_corpus(out_dir=tmp_path / "reference", counts=[1, 3], sessions_per_count=3, seed=5)

    resumed_dir = tmp_path / "resumed"
    _generate.generate_corpus(out_dir=resumed_dir, counts=[1, 3], sessions_per_count=3, seed=5)

    # Simulate a preempted task: an unlinked wav (never got that far) and a truncated rttm
    # (killed mid-write) are the two partial shapes _session_is_complete must catch.
    (resumed_dir / "k=3" / "session_1.wav").unlink()
    (resumed_dir / "k=1" / "session_2.rttm").write_text("")

    _generate.generate_corpus(out_dir=resumed_dir, counts=[1, 3], sessions_per_count=3, seed=5)

    for k in (1, 3):
        for i in range(3):
            resumed_rttm = (resumed_dir / f"k={k}" / f"session_{i}.rttm").read_text()
            reference_rttm = (reference / f"k={k}" / f"session_{i}.rttm").read_text()
            assert resumed_rttm == reference_rttm

            resumed_wav = Audio(filepath=str(resumed_dir / f"k={k}" / f"session_{i}.wav"))
            reference_wav = Audio(filepath=str(reference / f"k={k}" / f"session_{i}.wav"))
            assert torch.equal(resumed_wav.waveform, reference_wav.waveform)

    manifest_resumed = json.loads((resumed_dir / "manifest.json").read_text())
    manifest_reference = json.loads((reference / "manifest.json").read_text())
    assert manifest_resumed == manifest_reference


def test_a_zero_byte_wav_is_not_trusted_as_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero-length wav (a kill mid-encode, before any bytes landed) must be regenerated.

    Distinct from the rttm-truncation half of the interruption test above: this covers the
    wav side of the same partial-write hazard, checked directly against _session_is_complete
    rather than through a full generate_corpus round trip.
    """
    out_dir = tmp_path / "corpus"
    _generate.generate_corpus(out_dir=out_dir, counts=[2], sessions_per_count=1, seed=3)
    wav_path = out_dir / "k=2" / "session_0.wav"
    rttm_path = out_dir / "k=2" / "session_0.rttm"
    assert _generate._session_is_complete(wav_path, rttm_path)

    wav_path.write_bytes(b"")
    assert not _generate._session_is_complete(wav_path, rttm_path)


def test_sharded_generation_writes_a_distinct_manifest_name(tmp_path: Path) -> None:
    """A caller generating one k in isolation can point the manifest at a shard-scoped name.

    This is what lets several shard tasks write into the same corpus directory without one
    task's manifest.json clobbering another's -- see the manifest_name docstring.
    """
    out_dir = tmp_path / "corpus"
    _generate.generate_corpus(
        out_dir=out_dir, counts=[4], sessions_per_count=1, seed=7, manifest_name="manifest.k4.json"
    )
    assert not (out_dir / "manifest.json").exists()
    manifest = json.loads((out_dir / "manifest.k4.json").read_text())
    assert manifest["counts"] == [4]


def test_sessions_are_written_at_the_rate_diarizers_expect(tmp_path: Path) -> None:
    """Sessions land at 16 kHz, not the TTS model's native 24 kHz.

    Measured on an H100: Qwen3-TTS emits 24 kHz and pyannote rejects it outright with
    "Audio sampling rate 24000 does not match expected 16000", so a 24 kHz corpus scored zero
    successes and the probe correctly refused to emit any profile at all. Resampling once here
    is deterministic; leaving it to each backend would let the measured ceiling depend on whose
    resampler ran. The RTTM carries seconds, which resampling preserves exactly, so ground truth
    stays exact by construction.
    """
    import soundfile as sf

    out_dir = _generate.generate_corpus(out_dir=tmp_path / "corpus", counts=[2], sessions_per_count=1, seed=17)
    wav_path = out_dir / "k=2" / "session_0.wav"
    assert wav_path.exists()
    info = sf.info(str(wav_path))
    assert info.samplerate == _generate.CORPUS_SAMPLE_RATE == 16000, (
        f"corpus written at {info.samplerate} Hz; pyannote and the other diarizers require 16000"
    )


def test_synthesis_is_chunked_so_peak_memory_does_not_scale_with_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No synthesis call exceeds _SYNTH_BATCH_SIZE texts, however many sessions are requested.

    Batching every session's utterances into one call amortizes the model load but makes peak GPU
    memory scale with sessions_per_count, and at 20 sessions (~350 texts) it OOMs an 80 GB H100:
    "Tried to allocate 19.96 GiB ... 76.35 GiB already in use" (job 20125423). Asserting the batch
    ceiling here is what stops that regressing -- the failure only appears on real hardware at full
    sweep size, which no local run reaches.
    """
    seen_batch_sizes: list[int] = []
    real = _generate.synthesize_texts_with_qwen

    def _recording(*args: object, **kwargs: object) -> object:
        texts = kwargs.get("texts") or (args[0] if args else [])
        seen_batch_sizes.append(len(texts))  # type: ignore[arg-type]
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_generate, "synthesize_texts_with_qwen", _recording)
    _generate.generate_corpus(out_dir=tmp_path / "corpus", counts=[3], sessions_per_count=6, seed=17)

    assert seen_batch_sizes, "synthesis was never called"
    assert max(seen_batch_sizes) <= _generate._SYNTH_BATCH_SIZE, (
        f"a synthesis call carried {max(seen_batch_sizes)} texts, above the "
        f"{_generate._SYNTH_BATCH_SIZE} ceiling that keeps peak memory bounded"
    )


def test_per_speaker_speech_floor_holds_in_the_written_rttm_at_k8(tmp_path: Path) -> None:
    """At k=8, every speaker's total speech in the written RTTM clears MIN_SPEAKER_SPEECH_SECONDS.

    Before this fix, `_plan_session` stopped once a *total* word budget was reached, with no
    per-speaker floor at all: at k=8 on the real sweep, MIN_DOMINANCE's tail left the quietest
    speaker with ~1.4s of total speech in a ~50s session (job 20125423's corpus statistics) --
    far below what an embedding-based speaker model needs, so every k>=5 accuracy this probe
    measured was measuring the task's impossibility rather than a diarizer's competence. This
    would fail against that code (no floor enforced at all).

    Uses the module's default (autouse) synthesis stub, which renders at exactly
    ASSUMED_WORDS_PER_SECOND -- see that stub's docstring for why the rate matters here: a
    planned floor built on an assumed rate is not guaranteed to survive being rendered at a
    different real one, and this test is about the planning-and-verification logic, not about
    whether Qwen3-TTS's real speaking rate happens to match the assumption.
    """
    out_dir = _generate.generate_corpus(out_dir=tmp_path / "corpus", counts=[8], sessions_per_count=3, seed=123)
    for i in range(3):
        rttm_path = out_dir / "k=8" / f"session_{i}.rttm"
        speech_seconds: dict = defaultdict(float)
        for line in rttm_path.read_text().splitlines():
            if not line.strip():
                continue
            fields = line.split()
            speech_seconds[fields[7]] += float(fields[4])

        assert len(speech_seconds) == 8, f"session_{i}: expected 8 distinct speakers, found {len(speech_seconds)}"
        for speaker, secs in speech_seconds.items():
            assert secs >= _generate.MIN_SPEAKER_SPEECH_SECONDS, (
                f"session_{i} speaker {speaker!r}: {secs:.2f}s of total speech, below the "
                f"{_generate.MIN_SPEAKER_SPEECH_SECONDS}s floor"
            )


def test_realized_overlap_is_near_its_target() -> None:
    """The realized overlapped-time proportion lands near MEAN_OVERLAP's 10% target.

    Before this fix, MEAN_OVERLAP doubled as both `_lay_out_session`'s Bernoulli-fire
    probability *and* the target proportion of overlapped time -- their product, not
    either alone, is what determines the realized proportion, and the old (0.10 fire,
    ~0.225 mean fraction) combination realizes to ~0.0225 of speech time, matching the
    1.2-2.9% measured on the full sweep's RTTMs (a 4-8x undershoot from the stated 10%
    target). This drives the real `_plan_session` / `_lay_out_session` pair through many
    simulated sessions and checks the realized figure computed from the placed turns'
    start/end times, not from the constants' stated intent, so a future change to either
    OVERLAP_TRIGGER_PROB or OVERLAP_FRACTION_RANGE that drifts the realized proportion away
    from target is caught here rather than trusted.

    Overlapped seconds are measured between consecutive turns only (`_lay_out_session` only
    ever shifts a new turn into the immediately preceding one's tail), matching how the
    corpus's own overlap statistic came out nonzero even at k=1: a single speaker's own
    successive turns can still overlap under this placement model.
    """
    rng = np.random.default_rng(2024)
    total_speech = 0.0
    total_overlap = 0.0

    for _ in range(200):
        plan = _generate._plan_session(rng, num_speakers=4)
        durations = [max(1, len(text.split())) / _generate.ASSUMED_WORDS_PER_SECOND for _, text in plan]
        turns = _generate._lay_out_session(plan, durations, rng)

        total_speech += sum(turn.end - turn.start for turn in turns)
        total_overlap += sum(max(0.0, turns[j - 1].end - turns[j].start) for j in range(1, len(turns)))

    realized_proportion = total_overlap / total_speech
    assert realized_proportion == pytest.approx(_generate.MEAN_OVERLAP, rel=0.35), (
        f"realized overlap proportion {realized_proportion:.4f} is not near the "
        f"MEAN_OVERLAP={_generate.MEAN_OVERLAP} target it is meant to hit -- the old constants "
        "realized to roughly a fifth of this target"
    )
