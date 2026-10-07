"""Focused follow-up:
  (a) same-library baselines for chunk==slice (previous run mixed decoders into one baseline);
  (b) wide-window alignment search for mp3/aac to characterise the offset exactly;
  (c) does torchcodec's AudioSamples.pts_seconds let a caller recover the true position?
  (d) ffmpeg accurate seek (-ss AFTER -i) vs fast seek (-ss BEFORE -i).
"""

import os
import subprocess
import sys
import warnings

import numpy as np
import soundfile as sf
import torchaudio
from torchcodec.decoders import AudioDecoder

warnings.filterwarnings("ignore")
W = "/Users/satra/.claude/jobs/295c3f8a/tmp/io-audit/work3"
SR = 22050


def hr(t):
    print("\n" + "=" * 118 + f"\n## {t}\n" + "=" * 118)


FILES = {
    "wav PCM_16": ("c_pcm16.wav", SR),
    "wav FLOAT32": ("c_f32.wav", SR),
    "flac PCM_24": ("c_24.flac", SR),
    "mp3 192k": ("c.mp3", SR),
    "m4a aac": ("c.m4a", SR),
    "opus 48k": ("c.opus", 48000),
}


def full_tc(p):
    return AudioDecoder(p).get_all_samples().data.numpy()


def full_ta(p):
    return torchaudio.load(p)[0].numpy()


def full_sf(p):
    return sf.read(p, dtype="float32", always_2d=True)[0].T


def full_ff(p):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-f", "f32le", "-"], capture_output=True, check=True)
    ch = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                             "stream=channels", "-of", "default=nw=1:nk=1", p],
                            capture_output=True, text=True).stdout.strip())
    return np.frombuffer(r.stdout, dtype="<f4").reshape(-1, ch).T


def ch_tc(p, sr, a, n):
    return AudioDecoder(p).get_samples_played_in_range(
        start_seconds=a / sr, stop_seconds=(a + n) / sr).data.numpy()


def ch_ta(p, sr, a, n):
    return torchaudio.load(p, frame_offset=a, num_frames=n)[0].numpy()


def ch_sf(p, sr, a, n):
    return sf.read(p, dtype="float32", start=a, frames=n, always_2d=True)[0].T


def ch_ff_fast(p, sr, a, n):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{a / sr:.9f}", "-i", p, "-t", f"{n / sr:.9f}",
                        "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype="<f4").reshape(1, -1)


def ch_ff_accurate(p, sr, a, n):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ss", f"{a / sr:.9f}", "-t", f"{n / sr:.9f}",
                        "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype="<f4").reshape(1, -1)


PAIRS = [
    ("torchcodec", full_tc, ch_tc),
    ("torchaudio(=tc)", full_ta, ch_ta),
    ("soundfile", full_sf, ch_sf),
    ("ffmpeg -ss BEFORE -i", full_ff, ch_ff_fast),
    ("ffmpeg -ss AFTER -i", full_ff, ch_ff_accurate),
]

REQS = [("start 0", 0, 2048), ("odd 5001", 5001, 2048), ("mid", 44100 + 777, 4096)]


def align(full, ch, nominal, window=4000):
    n = ch.shape[-1]
    best = (None, np.inf, False)
    for d in range(-window, window + 1):
        a = nominal + d
        if a < 0 or a + n > full.shape[-1]:
            continue
        seg = full[:, a:a + n]
        m = float(np.max(np.abs(seg.astype(np.float64) - ch.astype(np.float64))))
        if m < best[1]:
            best = (d, m, bool(seg.tobytes() == ch.astype(np.float32).tobytes()))
    return best


hr("B4. chunk == slice, EACH LIBRARY AGAINST ITS OWN FULL DECODE, alignment searched over +/-4000 samples")
print("   'off' = how far the returned chunk actually sits from where it was requested (samples).")
for fname, (f, fsr) in FILES.items():
    p = os.path.join(W, f)
    print(f"\n  --- {fname} (sr={fsr})")
    for lname, ffull, fch in PAIRS:
        try:
            full = ffull(p)
        except Exception as e:
            print(f"      {lname:<22} full decode failed: {type(e).__name__}")
            continue
        out = [f"full_n={full.shape[1]}"]
        for tag, a0, n0 in REQS:
            a = int(round(a0 * fsr / SR))
            n = int(round(n0 * fsr / SR))
            try:
                ch = fch(p, fsr, a, n)
                d, m, ex = align(full, ch, a)
                out.append(f"{tag}: n={ch.shape[1]} off={d} diff={m:.2e} {'EXACT' if ex else '-'}")
            except Exception as e:
                out.append(f"{tag}: {type(e).__name__}")
        print(f"      {lname:<22} " + " | ".join(out))

hr("B5. Does AudioSamples.pts_seconds report the true start of the returned chunk?")
for fname, (f, fsr) in FILES.items():
    p = os.path.join(W, f)
    full = full_tc(p)
    print(f"\n  --- {fname}")
    for tag, a0, n0 in REQS:
        a = int(round(a0 * fsr / SR))
        n = int(round(n0 * fsr / SR))
        try:
            s = AudioDecoder(p).get_samples_played_in_range(start_seconds=a / fsr, stop_seconds=(a + n) / fsr)
            ch = s.data.numpy()
            pts_samp = s.pts_seconds * fsr
            d_req, m_req, ex_req = align(full, ch, a)
            d_pts, m_pts, ex_pts = align(full, ch, int(round(pts_samp)))
            print(f"      {tag:<10} requested_start={a:<7} pts_seconds={s.pts_seconds:.9f} "
                  f"(= sample {pts_samp:.1f})  n={ch.shape[1]:<5} "
                  f"align_vs_requested off={d_req} diff={m_req:.2e}  "
                  f"align_vs_pts off={d_pts} diff={m_pts:.2e} exact={ex_pts}")
        except Exception as e:
            print(f"      {tag:<10} {type(e).__name__}: {str(e)[:80]}")

hr("B6. mp3: what exactly does the range API drop? sweep chunk starts, report returned length")
p = os.path.join(W, "c.mp3")
full = full_tc(p)
print(f"  full decode n={full.shape[1]}; input was {SR*6}")
for a in [0, 1, 100, 576, 1105, 1106, 2048, 5001, 22050]:
    n = 2048
    s = AudioDecoder(p).get_samples_played_in_range(start_seconds=a / SR, stop_seconds=(a + n) / SR)
    ch = s.data.numpy()
    d, m, ex = align(full, ch, a, 4000)
    print(f"    request start={a:<7} n_req={n} -> n_got={ch.shape[1]:<6} pts={s.pts_seconds:.9f} "
          f"(sample {s.pts_seconds*SR:.1f})  best_off_vs_requested={d} diff={m:.2e} exact={ex}")

hr("B7. Cross-library agreement on a FULL decode of the same file (decoder identity, not chunking)")
for fname, (f, fsr) in FILES.items():
    p = os.path.join(W, f)
    dec = {}
    for lname, ffull, _ in PAIRS[:4]:
        try:
            dec[lname] = ffull(p)
        except Exception as e:
            dec[lname] = None
    base = dec.get("torchcodec")
    row = []
    for lname, a in dec.items():
        if a is None:
            row.append(f"{lname}=ERR")
        elif base is None:
            row.append(f"{lname}=n/a")
        elif a.shape == base.shape:
            row.append(f"{lname}: n={a.shape[1]} {'EXACT' if a.tobytes() == base.tobytes() else f'diff={np.abs(a-base).max():.2e}'}")
        else:
            row.append(f"{lname}: n={a.shape[1]} (len differs by {a.shape[1]-base.shape[1]})")
    print(f"  {fname:<14} " + " | ".join(row))

hr("B8. senselab Audio: does its offset/duration path inherit the mp3 defect? (uses torchcodec range)")
sys.path.insert(0, "/Users/satra/software/sensein/senselab/src")
from senselab.audio.data_structures.audio import Audio  # noqa: E402

for fname, (f, fsr) in FILES.items():
    p = os.path.join(W, f)
    full = Audio(filepath=p).waveform.numpy()
    for tag, a0, n0 in [("odd 5001", 5001, 2048)]:
        a = int(round(a0 * fsr / SR))
        n = int(round(n0 * fsr / SR))
        try:
            ch = Audio(filepath=p, offset_in_sec=a / fsr, duration_in_sec=n / fsr).waveform.numpy()
            d, m, ex = align(full, ch, a, 4000)
            print(f"  {fname:<14} full_n={full.shape[1]:<8} chunk_n={ch.shape[1]:<6} "
                  f"off_from_requested={d} max|diff|={m:.3e} bitexact={ex}")
        except Exception as e:
            print(f"  {fname:<14} {type(e).__name__}: {str(e)[:70]}")
