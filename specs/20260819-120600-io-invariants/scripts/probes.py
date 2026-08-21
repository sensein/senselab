import json
import os
import subprocess
import warnings

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

W = "/Users/satra/.claude/jobs/295c3f8a/tmp/io-audit/work2"
os.makedirs(W, exist_ok=True)
SR = 22050


def hr(t):
    print("\n" + "=" * 100 + f"\n## {t}\n" + "=" * 100)


# ---------------------------------------------------------------- 1. rounding
hr("1. float -> int16 conversion: rounding rule and scale factor per encoder")
# values chosen to expose truncation vs round-to-nearest and 32767 vs 32768 scaling
vals = np.array([0.0, 1.0, -1.0, 0.5, 0.4999847, 1.0 / 32768, 1.5 / 32768, 2.5 / 32768, -1.5 / 32768], dtype=np.float32)
x = vals.reshape(1, -1)
rows = {}
for name, fn in [
    ("torchcodec", lambda p: AudioEncoder(samples=torch.from_numpy(x), sample_rate=SR).to_file(p)),
    ("soundfile PCM_16", lambda p: sf.write(p, x.T, SR, format="WAV", subtype="PCM_16")),
    ("ffmpeg pcm_s16le", lambda p: subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
         "-c:a", "pcm_s16le", p], input=np.ascontiguousarray(x.T.astype("<f4")).tobytes(), check=True)),
]:
    p = os.path.join(W, f"round_{name.split()[0]}.wav")
    fn(p)
    raw, _ = sf.read(p, dtype="int16")
    rows[name] = raw
print(f"{'input float':>14} " + " ".join(f"{k:>18}" for k in rows))
for i, v in enumerate(vals):
    print(f"{v:>14.9f} " + " ".join(f"{int(rows[k][i]):>18}" for k in rows))
print("\nnote: 1.0 -> 32767 means /32767-ish or clamped; -1.0 -> -32768 means /32768 scale.")

# ---------------------------------------------------------------- 2. input dtypes
hr("2. torchcodec AudioEncoder: accepted input dtypes / shapes")
cases = {
    "float32 (1,N)": torch.rand(1, 100, dtype=torch.float32) * 0.5,
    "float64 (1,N)": torch.rand(1, 100, dtype=torch.float64) * 0.5,
    "float16 (1,N)": (torch.rand(1, 100) * 0.5).to(torch.float16),
    "int16 (1,N)": (torch.rand(1, 100) * 1000).to(torch.int16),
    "int32 (1,N)": (torch.rand(1, 100) * 1000).to(torch.int32),
    "float32 1-D (N,)": torch.rand(100, dtype=torch.float32) * 0.5,
    "float32 (N,1) time-first": torch.rand(100, 1, dtype=torch.float32) * 0.5,
    "float32 non-contig": (torch.rand(2, 200, dtype=torch.float32) * 0.5)[:, ::2],
    "numpy float32": np.random.rand(1, 100).astype(np.float32),
}
for k, v in cases.items():
    p = os.path.join(W, "dt.wav")
    try:
        AudioEncoder(samples=v, sample_rate=SR).to_file(p)
        i = sf.info(p)
        print(f"  {k:<26} OK  -> {i.subtype} ch={i.channels} frames={i.frames}")
    except Exception as e:
        print(f"  {k:<26} {type(e).__name__}: {str(e)[:130]}")

hr("2b. soundfile.write accepted input dtypes (WAV/FLOAT)")
for k, v in {
    "float32": np.random.rand(100, 1).astype(np.float32),
    "float64": np.random.rand(100, 1),
    "int16": (np.random.rand(100, 1) * 1000).astype(np.int16),
    "int32": (np.random.rand(100, 1) * 1000).astype(np.int32),
    "torch float32 tensor": torch.rand(100, 1),
}.items():
    p = os.path.join(W, "dt2.wav")
    try:
        sf.write(p, v, SR, format="WAV", subtype="FLOAT")
        print(f"  {k:<26} OK -> {sf.info(p).subtype}")
    except Exception as e:
        print(f"  {k:<26} {type(e).__name__}: {str(e)[:130]}")

# ---------------------------------------------------------------- 3. extension -> codec
hr("3. torchcodec: which codec/container does each extension select? (no way to override)")
x2 = (torch.rand(1, 2000, dtype=torch.float32) * 0.5)
for ext in ["wav", "flac", "ogg", "oga", "opus", "mp3", "m4a", "mp4", "aac", "wv", "caf", "aiff", "w64", "mka", "webm"]:
    p = os.path.join(W, f"ext.{ext}")
    if os.path.exists(p):
        os.remove(p)
    try:
        AudioEncoder(samples=x2, sample_rate=48000 if ext in ("opus", "webm") else SR).to_file(p)
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=codec_name,sample_fmt,bits_per_raw_sample", "-show_entries", "format=format_name",
             "-of", "default=nw=1:nk=1", p], capture_output=True, text=True).stdout.split()
        try:
            si = sf.info(p)
            sfs = f"{si.format}/{si.subtype}"
        except Exception:
            sfs = "libsndfile: CANNOT READ"
        print(f"  .{ext:<6} -> ffprobe {pr}   |  {sfs}")
    except Exception as e:
        print(f"  .{ext:<6} -> {type(e).__name__}: {str(e)[:110]}")

# ---------------------------------------------------------------- 4. metadata
hr("4. metadata preservation (title/comment) on WAV and FLAC")
sig = np.random.rand(1000, 1).astype(np.float32) * 0.5
# soundfile: set via SoundFile attributes
p_sf = os.path.join(W, "meta_sf.flac")
with sf.SoundFile(p_sf, "w", SR, 1, format="FLAC", subtype="PCM_24") as f:
    f.title = "SENSELAB_TITLE"
    f.comment = "SENSELAB_COMMENT"
    f.write(sig)
print("  soundfile.write+SoundFile attrs -> read back:", sf.info(p_sf).__dict__.get("title", "?"), "|",
      subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "default=nw=1", p_sf],
                     capture_output=True, text=True).stdout.strip().replace("\n", " "))
p_ff = os.path.join(W, "meta_ff.flac")
subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                "-c:a", "flac", "-metadata", "title=SENSELAB_TITLE", "-metadata", "comment=SENSELAB_COMMENT", p_ff],
               input=sig.tobytes(), check=True)
print("  ffmpeg -metadata -> read back:",
      subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "default=nw=1", p_ff],
                     capture_output=True, text=True).stdout.strip().replace("\n", " "))
p_tc = os.path.join(W, "meta_tc.flac")
AudioEncoder(samples=torch.from_numpy(sig.T.copy()), sample_rate=SR).to_file(p_tc)
print("  torchcodec: no metadata parameter exists. Tags written:",
      subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "default=nw=1", p_tc],
                     capture_output=True, text=True).stdout.strip().replace("\n", " ") or "(none)")
# does a decode path expose tags at all?
print("  AudioDecoder.metadata fields:", [a for a in dir(AudioDecoder(p_ff).metadata) if not a.startswith('_')])
print("  soundfile can read tags?", "sf.info has no tags;", "SoundFile.title ->",
      sf.SoundFile(p_ff).title if hasattr(sf.SoundFile(p_ff), "title") else "n/a")

# ---------------------------------------------------------------- 5. video containers
hr("5. audio extraction from a video container (mp4: h264 video + aac audio; mkv: vp9 + opus)")
vids = {}
for name, args in {
    "mp4_h264_aac": ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"],
    "mkv_vp9_opus": ["-c:v", "libvpx-vp9", "-c:a", "libopus"],
    "mov_h264_pcm": ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le"],
}.items():
    ext = name.split("_")[0]
    p = os.path.join(W, f"vid_{name}.{ext}")
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1"] + args + [p],
        capture_output=True, text=True)
    vids[name] = p if r.returncode == 0 else None
    if r.returncode != 0:
        print(f"  {name}: could not build fixture: {r.stderr[-200:]}")

for name, p in vids.items():
    if not p:
        continue
    print(f"\n  --- {name} ({os.path.getsize(p)} bytes)")
    try:
        d = AudioDecoder(p)
        s = d.get_all_samples()
        print(f"    torchcodec.AudioDecoder      OK shape={tuple(s.data.shape)} sr={s.sample_rate}")
    except Exception as e:
        print(f"    torchcodec.AudioDecoder      {type(e).__name__}: {str(e)[:110]}")
    try:
        a, sr = torchaudio.load(p)
        print(f"    torchaudio.load              OK shape={tuple(a.shape)} sr={sr}")
    except Exception as e:
        print(f"    torchaudio.load              {type(e).__name__}: {str(e)[:110]}")
    try:
        a, sr = sf.read(p)
        print(f"    soundfile.read               OK shape={a.shape} sr={sr}")
    except Exception as e:
        print(f"    soundfile.read               {type(e).__name__}: {str(e)[:110]}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import librosa
            a, sr = librosa.load(p, sr=None, mono=False)
        print(f"    librosa.load                 OK shape={np.atleast_2d(a).shape} sr={sr}")
    except Exception as e:
        print(f"    librosa.load                 {type(e).__name__}: {str(e)[:110]}")
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-f", "f32le", "-"], capture_output=True)
    print(f"    ffmpeg-cli                   {'OK ' + str(len(r.stdout) // 4) + ' samples' if r.returncode == 0 else 'FAIL'}")

# ---------------------------------------------------------------- 6. senselab end-to-end
hr("6. senselab Audio: save_to_file / load round-trip (uses torchcodec when available)")
import sys
sys.path.insert(0, "/Users/satra/software/sensein/senselab/src")
from senselab.audio.data_structures.audio import Audio  # noqa: E402

for sig_name, arr in {
    "in-range float32 (peak .9)": (np.random.default_rng(1).uniform(-0.9, 0.9, 2000)).astype(np.float32),
    "16-bit exact grid": (np.random.default_rng(2).integers(-32768, 32768, 2000) / 32768.0).astype(np.float32),
    "out-of-range (peak 3.0)": (np.concatenate([np.full(500, 1.5), np.full(500, -3.0), np.full(1000, 0.25)])).astype(np.float32),
}.items():
    for ext in ["wav", "flac", "ogg"]:
        a = Audio(waveform=arr, sampling_rate=SR)
        p = os.path.join(W, f"sl_{ext}.{ext}")
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            a.save_to_file(p, encoding="PCM_F", bits_per_sample=32)
            nw = [str(w.message)[:60] for w in wl]
        b = Audio(filepath=p)
        got = b.waveform.numpy()[0]
        ref = arr
        n = min(len(got), len(ref))
        exact = got.shape == ref.shape and ref.tobytes() == got.tobytes()
        try:
            sub = f"{sf.info(p).format}/{sf.info(p).subtype}"
        except Exception as e:
            sub = f"libsndfile CANNOT READ ({type(e).__name__})"
        print(f"  {sig_name:<28} .{ext:<5} {sub:<38} exact={exact!s:<5} "
              f"maxdiff={np.max(np.abs(ref[:n] - got[:n])):.3e} in_peak={np.max(np.abs(ref)):.4f} "
              f"out_peak={np.max(np.abs(got)):.4f} warns={len(nw)}")

hr("7. AudioDecoder: implicit resample / channel-fold parameters (silent transform risk)")
p = os.path.join(W, "res.wav")
sf.write(p, np.random.rand(2000, 2).astype(np.float32) * 0.5, 22050, format="WAV", subtype="FLOAT")
for kw in [{}, {"sample_rate": 16000}, {"num_channels": 1}, {"sample_rate": 16000, "num_channels": 1}]:
    s = AudioDecoder(p, **kw).get_all_samples()
    print(f"  AudioDecoder(**{kw}) -> shape={tuple(s.data.shape)} sr={s.sample_rate}")
print("  soundfile equivalent: none (sf.read never resamples); librosa.load(sr=...) resamples; ffmpeg -ar resamples")
