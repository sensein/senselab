"""Priority experiments.

A. Is decode source-independent?  dtype / amplitude convention / clamping / normalisation.
B. Does a streamed or seeked chunk equal the corresponding slice of a full decode, bit-for-bit?
"""

import os
import subprocess
import sys
import warnings

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchcodec.decoders import AudioDecoder

warnings.filterwarnings("ignore")
W = "/Users/satra/.claude/jobs/295c3f8a/tmp/io-audit/work3"
os.makedirs(W, exist_ok=True)


def hr(t):
    print("\n" + "=" * 118 + f"\n## {t}\n" + "=" * 118)


def ff_write(path, x, sr, codec):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", str(x.shape[0]),
         "-i", "-"] + codec + [path],
        input=np.ascontiguousarray(x.T.astype("<f4")).tobytes(), check=True)


# =====================================================================================
# A. decode standardisation
# =====================================================================================
hr("A1. Amplitude convention: full-scale integer sources. Is int16 -32768 -> -1.0 and 32767 -> 32767/32768?")
sr = 22050
ints = np.array([-32768, -32767, -16384, 0, 16384, 32766, 32767], dtype=np.int16)
p = os.path.join(W, "fs16.wav")
sf.write(p, ints, sr, format="WAV", subtype="PCM_16")
readers = {
    "torchcodec.AudioDecoder": lambda q: AudioDecoder(q).get_all_samples().data.numpy(),
    "torchaudio.load": lambda q: torchaudio.load(q)[0].numpy(),
    "soundfile.read(f32)": lambda q: sf.read(q, dtype="float32", always_2d=True)[0].T,
    "soundfile.read(f64)": lambda q: sf.read(q, dtype="float64", always_2d=True)[0].T,
    "librosa.load": lambda q: np.atleast_2d(__import__("librosa").load(q, sr=None, mono=False)[0]),
    "ffmpeg-cli(f64)": lambda q: np.frombuffer(
        subprocess.run(["ffmpeg", "-v", "error", "-i", q, "-f", "f64le", "-"], capture_output=True,
                       check=True).stdout, dtype="<f8").astype(np.float32).reshape(1, -1),
}
print(f"{'reader':<24} " + " ".join(f"{int(v):>12}" for v in ints))
print(f"{'/32768 reference':<24} " + " ".join(f"{v / 32768:>12.8f}" for v in ints))
for name, fn in readers.items():
    try:
        a = fn(p)[0]
        print(f"{name:<24} " + " ".join(f"{v:>12.8f}" for v in a))
    except Exception as e:
        print(f"{name:<24} {type(e).__name__}: {e}")
print(f"{'dtype returned':<24} " + ", ".join(f"{n}={fn(p).dtype}" for n, fn in readers.items() if n != 'ffmpeg-cli(f64)'))

hr("A2. THE KEY TEST: float32 WAV with samples beyond +/-1 -- does decode clamp?")
oor = np.concatenate([np.full(200, 3.0), np.full(200, -2.5), np.full(200, 1.5), np.linspace(-4, 4, 400)]).astype(np.float32)
for sub, ffc in [("FLOAT", ["-c:a", "pcm_f32le"]), ("DOUBLE", ["-c:a", "pcm_f64le"])]:
    p = os.path.join(W, f"oor_{sub}.wav")
    sf.write(p, oor, sr, format="WAV", subtype=sub)
    print(f"\n  source: WAV/{sub} written by soundfile, true peak = {np.abs(oor).max():.4f}")
    for name, fn in readers.items():
        try:
            a = fn(p)[0]
            print(f"    {name:<24} peak={np.abs(a).max():<10.5f} min={a.min():<10.5f} max={a.max():<10.5f} "
                  f"clamped={'YES' if np.abs(a).max() <= 1.0001 else 'no':<4} "
                  f"exact_vs_source={'YES' if a.shape == oor.shape and a.astype(np.float32).tobytes() == oor.tobytes() else 'no'}")
        except Exception as e:
            print(f"    {name:<24} {type(e).__name__}: {str(e)[:80]}")
    p2 = os.path.join(W, f"oor_ff_{sub}.wav")
    ff_write(p2, oor.reshape(1, -1), sr, ffc)
    a = readers["torchcodec.AudioDecoder"](p2)[0]
    print(f"    (same source written by ffmpeg) torchcodec peak={np.abs(a).max():.5f}")

hr("A3. Does any decode path normalise? Two files differing only by a known gain of 0.1")
base = (np.random.default_rng(7).uniform(-0.9, 0.9, 20000)).astype(np.float32)
pairs = {}
for tag, g in [("loud", 1.0), ("quiet", 0.1)]:
    for fmt, sub, ext in [("WAV", "FLOAT", "wav"), ("WAV", "PCM_16", "wav"), ("FLAC", "PCM_24", "flac")]:
        q = os.path.join(W, f"gain_{tag}_{fmt}_{sub}.{ext}")
        sf.write(q, (base * g).astype(np.float32), sr, format=fmt, subtype=sub)
        pairs[(tag, f"{fmt}/{sub}")] = q
for fs in ["WAV/FLOAT", "WAV/PCM_16", "FLAC/PCM_24"]:
    print(f"\n  {fs}: expected peak ratio quiet/loud = 0.1 exactly (no normalisation)")
    for name, fn in readers.items():
        try:
            pl, pq = np.abs(fn(pairs[("loud", fs)])).max(), np.abs(fn(pairs[("quiet", fs)])).max()
            rl, rq = float(np.sqrt((fn(pairs[("loud", fs)]) ** 2).mean())), float(np.sqrt((fn(pairs[("quiet", fs)]) ** 2).mean()))
            print(f"    {name:<24} peak {pl:.6f}/{pq:.6f} ratio={pq / pl:.6f}   rms ratio={rq / rl:.6f}")
        except Exception as e:
            print(f"    {name:<24} {type(e).__name__}")

hr("A4. Source-independence: identical audio in 8 containers, decoded by torchcodec. dtype / peak / RMS / length")
ref = (np.random.default_rng(11).uniform(-0.8, 0.8, 22050 * 2)).astype(np.float32)
srcs = {}
sf.write(os.path.join(W, "s_pcm16.wav"), ref, sr, format="WAV", subtype="PCM_16"); srcs["wav PCM_16"] = "s_pcm16.wav"
sf.write(os.path.join(W, "s_pcm24.wav"), ref, sr, format="WAV", subtype="PCM_24"); srcs["wav PCM_24"] = "s_pcm24.wav"
sf.write(os.path.join(W, "s_pcm32.wav"), ref, sr, format="WAV", subtype="PCM_32"); srcs["wav PCM_32"] = "s_pcm32.wav"
sf.write(os.path.join(W, "s_f32.wav"), ref, sr, format="WAV", subtype="FLOAT"); srcs["wav FLOAT32"] = "s_f32.wav"
sf.write(os.path.join(W, "s_f64.wav"), ref, sr, format="WAV", subtype="DOUBLE"); srcs["wav FLOAT64"] = "s_f64.wav"
sf.write(os.path.join(W, "s_16.flac"), ref, sr, format="FLAC", subtype="PCM_16"); srcs["flac 16"] = "s_16.flac"
sf.write(os.path.join(W, "s_24.flac"), ref, sr, format="FLAC", subtype="PCM_24"); srcs["flac 24"] = "s_24.flac"
ff_write(os.path.join(W, "s.mp3"), ref.reshape(1, -1), sr, ["-c:a", "libmp3lame", "-b:a", "192k"]); srcs["mp3 192k"] = "s.mp3"
ff_write(os.path.join(W, "s.opus"), ref.reshape(1, -1), 48000, ["-c:a", "libopus"]); srcs["opus (48k src)"] = "s.opus"
ff_write(os.path.join(W, "s.m4a"), ref.reshape(1, -1), sr, ["-c:a", "aac"]); srcs["m4a aac"] = "s.m4a"
subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "lavfi", "-i",
                "testsrc=size=160x120:rate=10:duration=2", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", os.path.join(W, "s.mp4")],
               input=np.ascontiguousarray(ref.astype("<f4")).tobytes(), check=True)
srcs["mp4 (video+aac)"] = "s.mp4"
print(f"  reference: dtype={ref.dtype} n={ref.size} peak={np.abs(ref).max():.6f} rms={np.sqrt((ref**2).mean()):.6f}")
print(f"\n  {'source':<18} {'dtype':<9} {'n':<8} {'sr':<7} {'peak':<10} {'rms':<10} {'peak/ref':<9} {'exact':<6}")
for k, f in srcs.items():
    q = os.path.join(W, f)
    try:
        d = AudioDecoder(q)
        s = d.get_all_samples()
        a = s.data.numpy()
        ex = a.shape[1] == ref.size and a[0].tobytes() == ref.tobytes()
        print(f"  {k:<18} {str(a.dtype):<9} {a.shape[1]:<8} {s.sample_rate:<7} {np.abs(a).max():<10.6f} "
              f"{np.sqrt((a**2).mean()):<10.6f} {np.abs(a).max() / np.abs(ref).max():<9.5f} {str(ex):<6}")
    except Exception as e:
        print(f"  {k:<18} {type(e).__name__}: {str(e)[:70]}")

# =====================================================================================
# B. chunk == slice
# =====================================================================================
hr("B. CHUNK == SLICE.  full = decode whole file; chunk = decode [t0, t1). Compare bit-for-bit.")

SR2 = 22050
NS = SR2 * 6
sig = (np.random.default_rng(23).integers(-32768, 32768, NS) / 32768.0).astype(np.float32)  # 16-bit-exact

FILES = {}
sf.write(os.path.join(W, "c_pcm16.wav"), sig, SR2, format="WAV", subtype="PCM_16"); FILES["wav PCM_16"] = "c_pcm16.wav"
sf.write(os.path.join(W, "c_f32.wav"), sig, SR2, format="WAV", subtype="FLOAT"); FILES["wav FLOAT32"] = "c_f32.wav"
sf.write(os.path.join(W, "c_24.flac"), sig, SR2, format="FLAC", subtype="PCM_24"); FILES["flac PCM_24"] = "c_24.flac"
ff_write(os.path.join(W, "c.mp3"), sig.reshape(1, -1), SR2, ["-c:a", "libmp3lame", "-b:a", "192k"]); FILES["mp3 192k"] = "c.mp3"
ff_write(os.path.join(W, "c.m4a"), sig.reshape(1, -1), SR2, ["-c:a", "aac", "-b:a", "128k"]); FILES["m4a aac"] = "c.m4a"
ff_write(os.path.join(W, "c.opus"), sig.reshape(1, -1), 48000, ["-c:a", "libopus"]); FILES["opus 48k"] = "c.opus"

# chunk requests: (start_sample, n_samples) at the file's own rate
REQS = [("aligned 0", 0, 2048), ("odd offset", 5001, 2048), ("mid-file", SR2 * 2 + 777, 4096),
        ("late", SR2 * 4, 8192)]


def chunk_torchcodec(path, srate, a, n):
    return AudioDecoder(path).get_samples_played_in_range(
        start_seconds=a / srate, stop_seconds=(a + n) / srate).data.numpy()


def chunk_torchaudio(path, srate, a, n):
    return torchaudio.load(path, frame_offset=a, num_frames=n)[0].numpy()


def chunk_soundfile(path, srate, a, n):
    x, _ = sf.read(path, dtype="float32", start=a, frames=n, always_2d=True)
    return x.T


def chunk_librosa(path, srate, a, n):
    import librosa
    x, _ = librosa.load(path, sr=None, mono=False, offset=a / srate, duration=n / srate)
    return np.atleast_2d(x)


def chunk_ffmpeg(path, srate, a, n):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{a / srate:.9f}", "-i", path, "-t", f"{n / srate:.9f}",
                        "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype="<f4").reshape(1, -1)


def chunk_senselab(path, srate, a, n):
    sys.path.insert(0, "/Users/satra/software/sensein/senselab/src")
    from senselab.audio.data_structures.audio import Audio
    return Audio(filepath=path, offset_in_sec=a / srate, duration_in_sec=n / srate).waveform.numpy()


CHUNKERS = {
    "torchcodec range": chunk_torchcodec,
    "torchaudio.load(off)": chunk_torchaudio,
    "soundfile.read(start)": chunk_soundfile,
    "librosa(offset)": chunk_librosa,
    "ffmpeg -ss/-t": chunk_ffmpeg,
    "senselab Audio(off)": chunk_senselab,
}


def best_align(full, ch, nominal, window=200):
    """Return (best_offset_delta, max_abs_diff at that offset, exact_bool) over +/- window samples."""
    n = ch.shape[-1]
    best = (None, np.inf, False)
    for d in range(-window, window + 1):
        a = nominal + d
        if a < 0 or a + n > full.shape[-1]:
            continue
        seg = full[:, a:a + n]
        if seg.shape != ch.shape:
            continue
        m = float(np.max(np.abs(seg.astype(np.float64) - ch.astype(np.float64))))
        if m < best[1]:
            best = (d, m, bool(seg.tobytes() == ch.astype(np.float32).tobytes()))
    return best


for fname, f in FILES.items():
    path = os.path.join(W, f)
    fsr = sf.info(path).samplerate if not f.endswith((".m4a", ".opus")) else int(
        subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
                        "-of", "default=nw=1:nk=1", path], capture_output=True, text=True).stdout.strip())
    full = AudioDecoder(path).get_all_samples().data.numpy()
    print(f"\n  --- {fname}   (file sr={fsr}, full decode n={full.shape[1]}, input n={NS})")
    print(f"      {'chunker':<22} {'request':<12} {'got_n':<7} {'nominal_off_delta':<18} {'max|diff|':<12} {'bitexact'}")
    for tag, a, n in REQS:
        aa = int(round(a * fsr / SR2))
        nn = int(round(n * fsr / SR2))
        for cname, cfn in CHUNKERS.items():
            try:
                ch = cfn(path, fsr, aa, nn)
                if ch.shape[0] != full.shape[0]:
                    print(f"      {cname:<22} {tag:<12} channel mismatch {ch.shape} vs {full.shape}")
                    continue
                d, m, ex = best_align(full, ch, aa)
                print(f"      {cname:<22} {tag:<12} {ch.shape[1]:<7} {str(d):<18} {m:<12.3e} {ex}")
            except Exception as e:
                print(f"      {cname:<22} {tag:<12} {type(e).__name__}: {str(e)[:60]}")

hr("B2. Contiguity: do consecutive chunks concatenate back to the full decode exactly?")
for fname, f in [("wav PCM_16", "c_pcm16.wav"), ("wav FLOAT32", "c_f32.wav"), ("flac PCM_24", "c_24.flac"),
                 ("mp3 192k", "c.mp3"), ("m4a aac", "c.m4a")]:
    path = os.path.join(W, f)
    full = AudioDecoder(path).get_all_samples().data.numpy()
    CH = 2048
    for cname, cfn in [("torchcodec range", chunk_torchcodec), ("soundfile.read(start)", chunk_soundfile),
                       ("torchaudio.load(off)", chunk_torchaudio)]:
        try:
            parts = []
            a = 0
            while a < full.shape[1]:
                parts.append(cfn(path, SR2, a, min(CH, full.shape[1] - a)))
                a += CH
            cat = np.concatenate(parts, axis=1)
            n = min(cat.shape[1], full.shape[1])
            ex = cat.shape == full.shape and cat.astype(np.float32).tobytes() == full.tobytes()
            m = float(np.max(np.abs(cat[:, :n].astype(np.float64) - full[:, :n].astype(np.float64))))
            nmis = int(np.sum(cat[:, :n] != full[:, :n]))
            print(f"  {fname:<14} {cname:<22} concat_n={cat.shape[1]:<8} full_n={full.shape[1]:<8} "
                  f"bitexact={str(ex):<6} max|diff|={m:.3e} mismatched_samples={nmis}")
        except Exception as e:
            print(f"  {fname:<14} {cname:<22} {type(e).__name__}: {str(e)[:70]}")

hr("B3. senselab Audio.from_stream (soundfile blocks) vs full decode")
sys.path.insert(0, "/Users/satra/software/sensein/senselab/src")
from senselab.audio.data_structures.audio import Audio  # noqa: E402

for fname, f in [("wav PCM_16", "c_pcm16.wav"), ("wav FLOAT32", "c_f32.wav"), ("flac PCM_24", "c_24.flac")]:
    path = os.path.join(W, f)
    full = Audio(filepath=path).waveform.numpy()
    cat = np.concatenate([c.waveform.numpy() for c in Audio.from_stream(path, chunk_duration_in_sec=0.25)], axis=1)
    n = min(cat.shape[1], full.shape[1])
    print(f"  {fname:<14} stream_n={cat.shape[1]:<8} full_n={full.shape[1]:<8} "
          f"bitexact={cat.shape == full.shape and cat.tobytes() == full.tobytes()} "
          f"max|diff|={np.max(np.abs(cat[:, :n] - full[:, :n])):.3e}")
