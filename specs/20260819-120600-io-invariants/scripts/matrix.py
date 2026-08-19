"""Audio I/O path matrix: encode paths x decode paths x formats x signals.

Read-only w.r.t. the senselab repo; writes only into its own tmp dir.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

TMP = os.environ.get("IOAUDIT_TMP", "/Users/satra/.claude/jobs/295c3f8a/tmp/io-audit/work")
os.makedirs(TMP, exist_ok=True)

SR = 22050

# ---------------------------------------------------------------- versions
import librosa  # noqa: E402
import torchaudio  # noqa: E402
import torchcodec  # noqa: E402
from torchcodec.decoders import AudioDecoder  # noqa: E402
from torchcodec.encoders import AudioEncoder  # noqa: E402

VERSIONS = {
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "torch": torch.__version__,
    "torchaudio": torchaudio.__version__,
    "torchcodec": torchcodec.__version__,
    "soundfile": sf.__version__,
    "libsndfile": sf.__libsndfile_version__,
    "librosa": librosa.__version__,
    "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0],
}

# ---------------------------------------------------------------- signals
rng = np.random.default_rng(0)


def _q16(n: int) -> np.ndarray:
    """Values exactly representable as PCM_16 under the /32768 convention."""
    k = rng.integers(-32768, 32768, size=n).astype(np.int64)
    return (k.astype(np.float64) / 32768.0).astype(np.float32)


def _f32(n: int) -> np.ndarray:
    return rng.uniform(-0.9, 0.9, size=n).astype(np.float32)


N = 2000

SIGNALS: Dict[str, np.ndarray] = {
    # mono, exactly representable in 16-bit
    "q16": _q16(N).reshape(1, -1),
    # mono, arbitrary float32 in range
    "f32": _f32(N).reshape(1, -1),
    # mono, out of range: peak 3.0 plus a flat DC block at 1.5
    "oor": np.concatenate([_f32(N) * (3.0 / 0.9), np.full(400, 1.5, dtype=np.float32)]).reshape(1, -1),
    # stereo, exactly representable in 16-bit, channels differ
    "q16_stereo": np.stack([_q16(N), _q16(N)]),
}

# ---------------------------------------------------------------- targets
TARGETS: List[Dict[str, Any]] = [
    dict(
        id="wav_pcm16",
        ext="wav",
        sf=("WAV", "PCM_16"),
        ffmpeg=["-c:a", "pcm_s16le"],
        ta=dict(format="wav", encoding="PCM_S", bits_per_sample=16),
    ),
    dict(
        id="wav_pcm24",
        ext="wav",
        sf=("WAV", "PCM_24"),
        ffmpeg=["-c:a", "pcm_s24le"],
        ta=dict(format="wav", encoding="PCM_S", bits_per_sample=24),
    ),
    dict(
        id="wav_pcm32",
        ext="wav",
        sf=("WAV", "PCM_32"),
        ffmpeg=["-c:a", "pcm_s32le"],
        ta=dict(format="wav", encoding="PCM_S", bits_per_sample=32),
    ),
    dict(
        id="wav_float32",
        ext="wav",
        sf=("WAV", "FLOAT"),
        ffmpeg=["-c:a", "pcm_f32le"],
        ta=dict(format="wav", encoding="PCM_F", bits_per_sample=32),
    ),
    dict(
        id="wav_float64",
        ext="wav",
        sf=("WAV", "DOUBLE"),
        ffmpeg=["-c:a", "pcm_f64le"],
        ta=dict(format="wav", encoding="PCM_F", bits_per_sample=64),
    ),
    dict(
        id="flac_16",
        ext="flac",
        sf=("FLAC", "PCM_16"),
        ffmpeg=["-c:a", "flac", "-sample_fmt", "s16"],
        ta=dict(format="flac", bits_per_sample=16),
    ),
    dict(
        id="flac_24",
        ext="flac",
        sf=("FLAC", "PCM_24"),
        ffmpeg=["-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24"],
        ta=dict(format="flac", bits_per_sample=24),
    ),
    dict(
        id="ogg_vorbis",
        ext="ogg",
        sf=("OGG", "VORBIS"),
        ffmpeg=["-c:a", "vorbis", "-strict", "-2"],
        ta=dict(format="ogg"),
    ),
    dict(
        id="opus",
        ext="opus",
        sf=("OGG", "OPUS"),
        ffmpeg=["-c:a", "libopus"],
        ta=dict(format="opus"),
    ),
    dict(
        id="mp3",
        ext="mp3",
        sf=("MP3", "MPEG_LAYER_III"),
        ffmpeg=["-c:a", "libmp3lame"],
        ta=dict(format="mp3"),
    ),
    dict(
        id="m4a_aac",
        ext="m4a",
        sf=None,
        ffmpeg=["-c:a", "aac"],
        ta=dict(format="m4a"),
    ),
]

LOSSY = {"ogg_vorbis", "opus", "mp3", "m4a_aac"}


# ---------------------------------------------------------------- encoders
def enc_torchcodec(path: str, x: np.ndarray, sr: int, t: Dict) -> None:
    AudioEncoder(samples=torch.from_numpy(x), sample_rate=sr).to_file(path)


def enc_torchaudio(path: str, x: np.ndarray, sr: int, t: Dict) -> None:
    kw = dict(t["ta"])
    torchaudio.save(uri=path, src=torch.from_numpy(x), sample_rate=sr, channels_first=True, **kw)


def enc_soundfile(path: str, x: np.ndarray, sr: int, t: Dict) -> None:
    if t["sf"] is None:
        raise NotImplementedError("libsndfile has no writer for this container")
    fmt, sub = t["sf"]
    sf.write(path, x.T, sr, format=fmt, subtype=sub)


def enc_ffmpeg(path: str, x: np.ndarray, sr: int, t: Dict) -> None:
    raw = np.ascontiguousarray(x.T.astype("<f4")).tobytes()
    cmd = (
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-f", "f32le", "-ar", str(sr), "-ac", str(x.shape[0]), "-i", "-"]
        + t["ffmpeg"]
        + [path]
    )
    p = subprocess.run(cmd, input=raw, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[-600:])
    enc_ffmpeg.last_stderr = p.stderr.decode()  # type: ignore[attr-defined]


ENCODERS = {
    "torchcodec.AudioEncoder": enc_torchcodec,
    "torchaudio.save": enc_torchaudio,
    "soundfile.write": enc_soundfile,
    "ffmpeg-cli": enc_ffmpeg,
}


# ---------------------------------------------------------------- decoders
def dec_torchcodec(path: str) -> Dict[str, Any]:
    d = AudioDecoder(path)
    s = d.get_all_samples()
    a = s.data.numpy()
    return dict(data=a, sr=int(s.sample_rate), dtype=str(a.dtype))


def dec_torchaudio(path: str) -> Dict[str, Any]:
    a, sr = torchaudio.load(path)
    return dict(data=a.numpy(), sr=int(sr), dtype=str(a.numpy().dtype))


def dec_soundfile_f32(path: str) -> Dict[str, Any]:
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    return dict(data=a.T, sr=int(sr), dtype="float32")


def dec_soundfile_f64(path: str) -> Dict[str, Any]:
    a, sr = sf.read(path, dtype="float64", always_2d=True)
    return dict(data=a.T.astype(np.float32), sr=int(sr), dtype="float64->float32")


def dec_librosa(path: str) -> Dict[str, Any]:
    a, sr = librosa.load(path, sr=None, mono=False)
    a = np.atleast_2d(a)
    return dict(data=a, sr=int(sr), dtype=str(a.dtype))


def _ffprobe(path: str) -> Dict[str, Any]:
    p = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_fmt,sample_rate,channels,bits_per_raw_sample,bit_rate",
            "-show_entries",
            "format=format_name",
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        text=True,
    )
    try:
        j = json.loads(p.stdout)
        st = j.get("streams", [{}])[0]
        st["format_name"] = j.get("format", {}).get("format_name")
        return st
    except Exception:
        return {"error": p.stderr[-300:]}


def dec_ffmpeg(path: str) -> Dict[str, Any]:
    info = _ffprobe(path)
    ch = int(info.get("channels", 1) or 1)
    sr = int(info.get("sample_rate", SR) or SR)
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", path, "-f", "f64le", "-c:a", "pcm_f64le", "-"],
        capture_output=True,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[-400:])
    a = np.frombuffer(p.stdout, dtype="<f8").reshape(-1, ch).T.astype(np.float32)
    return dict(data=a, sr=sr, dtype="float64->float32")


DECODERS = {
    "torchcodec.AudioDecoder": dec_torchcodec,
    "torchaudio.load": dec_torchaudio,
    "soundfile.read(f32)": dec_soundfile_f32,
    "soundfile.read(f64)": dec_soundfile_f64,
    "librosa.load": dec_librosa,
    "ffmpeg-cli": dec_ffmpeg,
}


# ---------------------------------------------------------------- compare
def compare(ref: np.ndarray, got: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(ref_shape=list(ref.shape), got_shape=list(got.shape))
    if ref.shape != got.shape:
        n = min(ref.shape[-1], got.shape[-1])
        if ref.shape[0] != got.shape[0]:
            out["exact"] = False
            out["note"] = "channel count differs"
            return out
        # lossy codecs pad/delay: compare the common prefix only, plus best-offset
        r, g = ref[:, :n], got[:, :n]
        out["exact"] = False
        out["note"] = f"length differs by {got.shape[-1] - ref.shape[-1]}"
        out["max_abs_diff_prefix"] = float(np.max(np.abs(r - g))) if n else None
        out["got_peak"] = float(np.max(np.abs(got))) if got.size else None
        out["got_distinct"] = int(np.unique(got).size) if got.size else 0
        return out
    d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    out["exact"] = bool(ref.tobytes() == got.astype(np.float32).tobytes())
    out["allclose_bitwise"] = bool(np.array_equal(ref, got.astype(np.float32)))
    out["max_abs_diff"] = float(d.max()) if d.size else 0.0
    out["got_peak"] = float(np.max(np.abs(got))) if got.size else None
    out["got_distinct"] = int(np.unique(got).size) if got.size else 0
    return out


# ---------------------------------------------------------------- run
results: List[Dict[str, Any]] = []

for t in TARGETS:
    for enc_name, enc in ENCODERS.items():
        for sig_name, x in SIGNALS.items():
            path = os.path.join(TMP, f"{t['id']}__{enc_name.replace('.', '_').replace('-', '_')}__{sig_name}.{t['ext']}")
            if os.path.exists(path):
                os.remove(path)
            rec: Dict[str, Any] = dict(
                target=t["id"], ext=t["ext"], encoder=enc_name, signal=sig_name, sr_in=SR, ch_in=int(x.shape[0])
            )
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter("always")
                try:
                    enc(path, x, SR, t)
                    rec["write"] = "ok"
                except Exception as e:  # noqa: BLE001
                    rec["write"] = "FAIL"
                    rec["write_error"] = f"{type(e).__name__}: {e}"[:400]
                rec["write_warnings"] = [f"{w.category.__name__}: {str(w.message)[:200]}" for w in wl]
            if rec["write"] == "ok":
                rec["probe"] = _ffprobe(path)
                try:
                    i = sf.info(path)
                    rec["sf_info"] = dict(format=i.format, subtype=i.subtype, sr=i.samplerate, ch=i.channels, frames=i.frames)
                except Exception as e:  # noqa: BLE001
                    rec["sf_info"] = {"error": f"{type(e).__name__}: {e}"[:200]}
                rec["size_bytes"] = os.path.getsize(path)
                rec["decode"] = {}
                for dec_name, dec in DECODERS.items():
                    with warnings.catch_warnings(record=True) as wl:
                        warnings.simplefilter("always")
                        try:
                            r = dec(path)
                            c = compare(x, r["data"])
                            c["sr_out"] = r["sr"]
                            c["sr_preserved"] = r["sr"] == SR
                            c["read_dtype"] = r["dtype"]
                            c["warnings"] = [f"{w.category.__name__}: {str(w.message)[:150]}" for w in wl]
                            rec["decode"][dec_name] = c
                        except Exception as e:  # noqa: BLE001
                            rec["decode"][dec_name] = {"error": f"{type(e).__name__}: {e}"[:300]}
            results.append(rec)
            print(
                f"{t['id']:<12} {enc_name:<24} {sig_name:<11} {rec['write']:<5} "
                f"{rec.get('sf_info', {}).get('subtype') or rec.get('probe', {}).get('codec_name', '')}",
                flush=True,
            )

out_path = os.path.join(os.path.dirname(TMP), "matrix.json")
with open(out_path, "w") as f:
    json.dump({"versions": VERSIONS, "sr": SR, "results": results}, f, indent=1)
print("wrote", out_path)
