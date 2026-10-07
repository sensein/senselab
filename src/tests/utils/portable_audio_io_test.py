"""The one audio I/O policy, checked against libsndfile and against its second caller.

Every assertion goes through a real write and a real read-back rather than comparing strings,
because the defect is that ``sf.write(path, waveform, sr)`` and
``AudioEncoder(...).to_file(path)`` both look right and substitute different samples.

Reasoning and the measurements: ``specs/20260819-091500-wav-subtype-sweep/design.md``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from senselab.audio.data_structures import Audio
from senselab.utils import portable_audio_io as pio
from senselab.utils.portable_audio_io import (
    LOSSLESS_WAV_SUBTYPE,
    apply_range_policy,
    is_float_subtype,
    out_of_range_fraction,
    read_audio,
    resolve_subtype,
    widest_subtype,
    write_audio,
)

_SR = 16000


def _over_unity_signal(peak: float = 1.6) -> np.ndarray:
    """A float32 signal peaking past full scale, the excursion DriftSE's plain checkpoint reached."""
    t = np.linspace(0.0, 1.0, _SR, endpoint=False, dtype=np.float32)
    return (peak * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


# --------------------------------------------------------------------------------------------
# Portability: the property that lets a worker venv use this module at all
# --------------------------------------------------------------------------------------------


def test_the_policy_module_imports_nothing_from_senselab() -> None:
    """A worker venv has no senselab, so one convenient import here breaks every worker."""
    tree = ast.parse(Path(pio.__file__).read_text())
    offenders: list[str] = []
    allowed = {"numpy", "soundfile", "logging", "os", "typing", "__future__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.split(".")[0] not in allowed]
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in allowed:
                offenders.append(node.module or f"relative level {node.level}")

    assert not offenders, (
        "portable_audio_io.py must import only numpy, soundfile and the standard library -- it is "
        f"staged into venvs where senselab is absent. Offending imports: {sorted(set(offenders))}"
    )


def test_the_policy_module_is_importable_from_a_bare_copy(tmp_path: Path) -> None:
    """Staging is a file copy, so the file must work standing alone, not as part of a package."""
    from senselab.utils.subprocess_venv import stage_portable_audio_io

    staged_dir = stage_portable_audio_io(tmp_path)
    copied = Path(staged_dir) / "portable_audio_io.py"
    assert copied.is_file()

    import importlib.util

    spec = importlib.util.spec_from_file_location("staged_portable_audio_io", copied)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    path = tmp_path / "staged.wav"
    report = module.write_audio(str(path), _over_unity_signal(), _SR)
    assert report.subtype == "FLOAT"
    assert sf.info(str(path)).subtype == "FLOAT"


# --------------------------------------------------------------------------------------------
# The premise: what the defaults actually do to samples
# --------------------------------------------------------------------------------------------


def test_the_wav_default_really_does_clip_and_float_really_does_not(tmp_path: Path) -> None:
    """The premise of the whole module, measured rather than asserted from the docs."""
    signal = _over_unity_signal()

    default_path = tmp_path / "default.wav"
    sf.write(str(default_path), signal, _SR)  # the defect itself, reproduced deliberately
    assert sf.info(str(default_path)).subtype == "PCM_16"
    clipped, _ = sf.read(str(default_path), dtype="float32")
    assert float(np.abs(clipped).max()) == pytest.approx(1.0, abs=1e-4)
    assert float(np.count_nonzero(np.abs(clipped) >= 0.9999)) / clipped.size > 0.5

    preserved = tmp_path / "preserved.wav"
    report = write_audio(preserved, signal, _SR)
    assert report.subtype == "FLOAT"
    recovered, _ = sf.read(str(preserved), dtype="float32")
    assert np.array_equal(recovered, signal), "a FLOAT WAV must round-trip float32 bit-exactly"


def _rms_dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))) + 1e-30))


def test_the_16_bit_floor_replaces_quiet_content_rather_than_dropping_it(tmp_path: Path) -> None:
    """At 16 bits, both -100 and -120 dBFS read back at the -93 dBFS 1-LSB floor.

    libsndfile truncates toward negative infinity rather than rounding, so at -120 dBFS the
    surviving int16 codes are ``{0, -1}`` -- a rectified artifact, not silence.
    """
    rng = np.random.default_rng(0)
    pcm16 = tmp_path / "quiet_pcm16.wav"

    for level_dbfs, minimum_error_db in [(-100.0, 3.0), (-120.0, 20.0)]:
        quiet = (rng.standard_normal(_SR).astype(np.float32) * 10 ** (level_dbfs / 20)).astype(np.float32)
        write_audio(pcm16, quiet, _SR, subtype="PCM_16")
        back, _ = sf.read(str(pcm16), dtype="float32")
        assert _rms_dbfs(back) - _rms_dbfs(quiet) > minimum_error_db, (
            f"a {level_dbfs:g} dBFS signal must read back louder than itself at 16 bits"
        )
        assert _rms_dbfs(back) == pytest.approx(-93.3, abs=1.0), "the read-back level is the 1-LSB floor"

        lossless = tmp_path / "quiet_float.wav"
        write_audio(lossless, quiet, _SR)
        recovered, _ = read_audio(lossless)
        assert np.array_equal(recovered, quiet), "the default must round-trip a -120 dBFS signal exactly"

    codes, _ = sf.read(str(pcm16), dtype="int16")
    assert set(np.unique(codes).tolist()) <= {0, -1}, "at -120 dBFS nothing but the truncation artifact survives"


# --------------------------------------------------------------------------------------------
# Decision 1: which subtype a destination gets
# --------------------------------------------------------------------------------------------


def test_flac_cannot_take_the_wav_subtype() -> None:
    """FLAC has no float subtype, which is why the sweep is not "write FLOAT everywhere"."""
    assert not sf.check_format("FLAC", LOSSLESS_WAV_SUBTYPE)
    assert sorted(sf.available_subtypes("FLAC")) == ["PCM_16", "PCM_24", "PCM_S8"]


@pytest.mark.parametrize(
    ("fmt", "dtype", "expected"),
    [
        ("wav", np.float32, "FLOAT"),
        ("WAV", np.float64, "FLOAT"),
        ("wav", np.int16, "PCM_16"),
        ("wav", np.int32, "PCM_32"),
        ("flac", np.float32, "PCM_24"),
        ("flac", np.int16, "PCM_16"),
        ("aiff", np.float32, "FLOAT"),
        ("ogg", np.float32, "VORBIS"),
        ("ogg", np.int16, "VORBIS"),
        ("mp3", np.float32, "MPEG_LAYER_III"),
    ],
)
def test_widest_subtype_picks_something_the_container_accepts(fmt: str, dtype: type, expected: str) -> None:
    """Case-insensitive, dtype-aware, and never a subtype the format would reject."""
    subtype = widest_subtype(fmt, dtype)
    assert subtype == expected
    assert sf.check_format(fmt.upper(), subtype)


def test_resolve_subtype_refuses_preserve_exactly_plus_flac() -> None:
    """Preserving exactly and writing FLAC are mutually exclusive, and must fail loudly."""
    with pytest.raises(ValueError, match="no float subtype at any depth"):
        resolve_subtype("FLAC", np.float32, "FLOAT")
    assert resolve_subtype("FLAC", np.float32, None) == "PCM_24"
    assert resolve_subtype("WAV", np.float32, "pcm_16") == "PCM_16", "a request is normalised, not rejected"


def test_widest_subtype_refuses_what_it_cannot_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown format must raise; so must a format libsndfile gives no default for.

    The second case is unreachable as libsndfile 1.2 ships (``RAW`` is its only format with no
    default subtype, and it supports ``FLOAT``), so it is monkeypatched: the alternative to
    raising is returning ``None``, which ``sf.write`` reads as "use the default".
    """
    with pytest.raises(ValueError, match="not a format soundfile can write"):
        widest_subtype("not-a-format", np.float32)

    monkeypatch.setattr("senselab.utils.portable_audio_io.sf.available_subtypes", lambda fmt: {"ULAW"})
    monkeypatch.setattr("senselab.utils.portable_audio_io.sf.default_subtype", lambda fmt: None)
    with pytest.raises(ValueError, match="no default subtype"):
        widest_subtype("wav", np.float32)


def test_write_audio_refuses_a_container_libsndfile_cannot_write(tmp_path: Path) -> None:
    """The shim has one backend, so an m4a is refused here rather than silently mis-encoded."""
    with pytest.raises(ValueError, match="not a format soundfile can write"):
        write_audio(tmp_path / "x.m4a", _over_unity_signal(0.5), _SR)


# --------------------------------------------------------------------------------------------
# Decision 2: what happens when the samples do not fit
# --------------------------------------------------------------------------------------------


def test_out_of_range_fraction_counts_only_what_a_fixed_point_write_would_lose() -> None:
    """Exactly ±1 is representable, so it must not be counted; an empty array is 0.0."""
    assert out_of_range_fraction(np.array([0.0, 1.0, -1.0])) == 0.0
    assert out_of_range_fraction(np.array([0.0, 1.5, -2.0, 0.5])) == pytest.approx(0.5)
    assert out_of_range_fraction(np.zeros((2, 4))) == 0.0
    assert out_of_range_fraction(np.array([], dtype=np.float32)) == 0.0
    assert out_of_range_fraction(_over_unity_signal()) > 0.5


def test_is_float_subtype_separates_the_two_that_hold_floats() -> None:
    """The predicate the range policy short-circuits on."""
    assert is_float_subtype(LOSSLESS_WAV_SUBTYPE)
    assert is_float_subtype("DOUBLE")
    for fixed in ("PCM_16", "PCM_24", "PCM_32", "ULAW", "ALAW", "VORBIS", "MPEG_LAYER_III"):
        assert not is_float_subtype(fixed)


def test_a_float_subtype_makes_the_range_policy_a_no_op() -> None:
    """Nothing is checked, warned about or rescaled when the destination can hold the values."""
    signal = _over_unity_signal(3.0)
    samples, peak, fraction, gain = apply_range_policy(signal, "WAV", "FLOAT", pio.RAISE)
    assert samples is signal
    assert (peak, fraction, gain) == (pytest.approx(3.0), 0.0, 1.0)


def test_raise_is_the_default_and_names_the_ways_out() -> None:
    """A refused write is the default, because a silently rescaled measurement is worse."""
    with pytest.raises(ValueError) as excinfo:
        apply_range_policy(_over_unity_signal(3.0), "FLAC", "PCM_24", destination="/tmp/x.flac")
    message = str(excinfo.value)
    assert "peak 3" in message and "FLAC/PCM_24" in message and "/tmp/x.flac" in message
    assert "out_of_range='normalize'" in message and "float-capable" in message


def test_normalize_records_a_gain_that_recovers_the_original(tmp_path: Path) -> None:
    """The rescale must be reversible, or it is just a quieter loss."""
    signal = _over_unity_signal(3.0)
    path = tmp_path / "normalized.wav"
    report = write_audio(path, signal, _SR, subtype="PCM_16", out_of_range="normalize")

    assert report.gain == pytest.approx(1.0 / 3.0)
    assert report.peak == pytest.approx(3.0)
    assert report.out_of_range_fraction > 0.5

    back, _ = read_audio(path)
    assert float(np.abs(back).max()) == pytest.approx(1.0, abs=1e-4)
    assert np.allclose(back / report.gain, signal, atol=1e-4), "dividing by the gain must recover the signal"


def test_warn_clips_but_says_so(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The only policy that loses data does it in the log, not silently."""
    path = tmp_path / "warned.wav"
    with caplog.at_level("WARNING", logger="senselab"):
        report = write_audio(path, _over_unity_signal(3.0), _SR, subtype="PCM_16", out_of_range="warn")
    assert report.gain == 1.0
    assert report.out_of_range_fraction > 0.5
    assert any("clipped" in record.message or "clipped" in record.getMessage() for record in caplog.records)
    back, _ = read_audio(path)
    assert float(np.abs(back).max()) == pytest.approx(1.0, abs=1e-4)


def test_an_unknown_policy_is_rejected() -> None:
    """A typo in the policy name must not fall through to the permissive branch."""
    with pytest.raises(ValueError, match="out_of_range must be one of"):
        apply_range_policy(_over_unity_signal(3.0), "WAV", "PCM_16", "clip")


# --------------------------------------------------------------------------------------------
# The agreement test: two callers, one policy
# --------------------------------------------------------------------------------------------

_AGREEMENT_CASES = [
    ("in-range wav", 0.9, "test.wav", None, "raise"),
    ("out-of-range wav, preserved by default", 3.0, "test.wav", None, "raise"),
    ("out-of-range wav, subtype forces the question", 3.0, "test.wav", "PCM_16", "normalize"),
    ("out-of-range wav, clipping accepted", 3.0, "test.wav", "PCM_16", "warn"),
    ("in-range flac", 0.9, "test.flac", None, "raise"),
    ("out-of-range flac", 3.0, "test.flac", None, "raise"),
    ("float-preserving wav, asked for by name", 3.0, "test.wav", "FLOAT", "raise"),
]


@pytest.mark.parametrize(("label", "peak", "name", "subtype", "policy"), _AGREEMENT_CASES)
def test_audio_and_the_shim_make_the_same_decision(
    tmp_path: Path, label: str, peak: float, name: str, subtype: str | None, policy: str
) -> None:
    """``Audio.save_to_file`` and ``write_audio`` must agree, byte for byte and error for error.

    ``save_to_file`` delegates today, so this passes structurally. It is here for the day someone
    reimplements one of the two: a shim that has drifted is worse than no shim, because the policy
    then only looks applied.
    """
    signal = _over_unity_signal(peak)
    audio = Audio(waveform=torch.from_numpy(signal).unsqueeze(0), sampling_rate=_SR)

    shim_path = tmp_path / f"shim_{name}"
    audio_path = tmp_path / f"audio_{name}"

    shim_error: Exception | None = None
    audio_error: Exception | None = None
    shim_report = audio_report = None
    try:
        shim_report = write_audio(shim_path, signal, _SR, subtype=subtype, out_of_range=policy)
    except Exception as exc:  # noqa: BLE001 -- the error itself is what is being compared
        shim_error = exc
    try:
        audio_report = audio.save_to_file(audio_path, subtype=subtype, out_of_range=policy)
    except Exception as exc:  # noqa: BLE001
        audio_error = exc

    assert (shim_error is None) == (audio_error is None), f"{label}: one path raised and the other did not"
    if shim_error is not None:
        assert type(shim_error) is type(audio_error)
        assert str(shim_error).replace(str(shim_path), "") == str(audio_error).replace(str(audio_path), "")
        return

    assert shim_report is not None and audio_report is not None
    assert (shim_report.format, shim_report.subtype) == (audio_report.format, audio_report.subtype), label
    assert (shim_report.peak, shim_report.gain) == (audio_report.peak, audio_report.gain), label
    assert shim_report.out_of_range_fraction == audio_report.out_of_range_fraction, label
    assert shim_path.read_bytes() == audio_path.read_bytes(), f"{label}: the files differ"


def test_read_audio_returns_senselab_axis_order(tmp_path: Path) -> None:
    """Channels first for 2-D, and 1-D left alone, so a worker's array matches the host's."""
    stereo = np.stack([_over_unity_signal(0.5), _over_unity_signal(0.25)])
    path = tmp_path / "stereo.wav"
    write_audio(path, stereo, _SR, channels_first=True)

    channels_first, sampling_rate = read_audio(path, channels_first=True)
    assert sampling_rate == _SR
    assert channels_first.shape == (2, _SR)
    assert np.allclose(channels_first, stereo)

    time_first, _ = read_audio(path, channels_first=False)
    assert time_first.shape == (_SR, 2)

    mono_path = tmp_path / "mono.wav"
    write_audio(mono_path, _over_unity_signal(0.5), _SR)
    mono, _ = read_audio(mono_path)
    assert mono.ndim == 1, "a mono read stays 1-D unless always_2d is asked for"
    mono_2d, _ = read_audio(mono_path, always_2d=True)
    assert mono_2d.shape == (1, _SR)
