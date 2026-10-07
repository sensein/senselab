"""Closing two decode-side gaps:
  C. PER-SEGMENT normalisation: a file whose halves differ by 60 dB. If any path normalised
     per segment, the quiet chunk would come back boosted relative to its slice.
  D. Frame-boundary dependence of seeked chunks for mp3 / aac / opus: sweep the offset across
     a codec frame and report where chunk != slice and by how much.
"""
import os, subprocess, warnings
import numpy as np, soundfile as sf, torchaudio
from torchcodec.decoders import AudioDecoder
warnings.filterwarnings("ignore")
W = "/Users/satra/.claude/jobs/295c3f8a/tmp/io-audit/work5"; os.makedirs(W, exist_ok=True)
SR = 22050

def hr(t): print("\n" + "=" * 112 + f"\n## {t}\n" + "=" * 112)

def ffw(p, x, sr, codec):
    subprocess.run(["ffmpeg","-hide_banner","-v","error","-y","-f","f32le","-ar",str(sr),"-ac","1","-i","-"]
                   + codec + [p], input=np.ascontiguousarray(x.astype("<f4")).tobytes(), check=True)

# ---- C. per-segment normalisation ----
hr("C. Per-segment normalisation test: loud half (peak 0.9) then quiet half (peak 0.0009, -60 dB)")
n = SR * 3
rng = np.random.default_rng(41)
loud  = rng.uniform(-0.9, 0.9, n).astype(np.float32)
quiet = (rng.uniform(-0.9, 0.9, n) * 1e-3).astype(np.float32)
sig = np.concatenate([loud, quiet]).astype(np.float32)
print(f"  file: loud-half peak={np.abs(loud).max():.6f}  quiet-half peak={np.abs(quiet).max():.9f}  ratio={np.abs(quiet).max()/np.abs(loud).max():.3e}")

files = {}
sf.write(os.path.join(W,"d_f32.wav"), sig, SR, format="WAV", subtype="FLOAT"); files["wav FLOAT32"]=("d_f32.wav",SR)
sf.write(os.path.join(W,"d_p16.wav"), sig, SR, format="WAV", subtype="PCM_16"); files["wav PCM_16"]=("d_p16.wav",SR)
sf.write(os.path.join(W,"d_24.flac"), sig, SR, format="FLAC", subtype="PCM_24"); files["flac PCM_24"]=("d_24.flac",SR)
ffw(os.path.join(W,"d.mp3"), sig, SR, ["-c:a","libmp3lame","-b:a","192k"]); files["mp3 192k"]=("d.mp3",SR)
ffw(os.path.join(W,"d.m4a"), sig, SR, ["-c:a","aac","-b:a","128k"]);        files["m4a aac"]=("d.m4a",SR)
ffw(os.path.join(W,"d.opus"), sig, 48000, ["-c:a","libopus"]);              files["opus 48k"]=("d.opus",48000)

READERS = {
  "torchcodec": (lambda p: AudioDecoder(p).get_all_samples().data.numpy(),
                 lambda p,sr,a,nn: AudioDecoder(p).get_samples_played_in_range(start_seconds=a/sr, stop_seconds=(a+nn)/sr).data.numpy()),
  "torchaudio": (lambda p: torchaudio.load(p)[0].numpy(),
                 lambda p,sr,a,nn: torchaudio.load(p, frame_offset=a, num_frames=nn)[0].numpy()),
  "soundfile":  (lambda p: sf.read(p, dtype="float32", always_2d=True)[0].T,
                 lambda p,sr,a,nn: sf.read(p, dtype="float32", start=a, frames=nn, always_2d=True)[0].T),
  "ffmpeg -ss after -i": (
      lambda p: np.frombuffer(subprocess.run(["ffmpeg","-v","error","-i",p,"-f","f32le","-"],capture_output=True,check=True).stdout,dtype="<f4").reshape(1,-1),
      lambda p,sr,a,nn: np.frombuffer(subprocess.run(["ffmpeg","-v","error","-i",p,"-ss",f"{a/sr:.9f}","-t",f"{nn/sr:.9f}","-f","f32le","-"],capture_output=True,check=True).stdout,dtype="<f4").reshape(1,-1)),
}
print(f"\n  {'format':<13} {'reader':<20} {'quiet-chunk peak':<18} {'slice peak':<14} {'chunk/slice':<13} {'bitexact'}")
for fname,(f,fsr) in files.items():
    p = os.path.join(W,f)
    a = int(round(n * fsr/SR)); nn = int(round(SR*1.0 * fsr/SR))
    for rname,(rfull,rch) in READERS.items():
        try:
            full = rfull(p); ch = rch(p,fsr,a,nn)
            sl = full[:, a:a+ch.shape[1]]
            pk_c, pk_s = float(np.abs(ch).max()), float(np.abs(sl).max())
            ex = sl.shape==ch.shape and sl.tobytes()==ch.astype(np.float32).tobytes()
            print(f"  {fname:<13} {rname:<20} {pk_c:<18.9f} {pk_s:<14.9f} {pk_c/pk_s if pk_s else float('nan'):<13.6f} {ex}")
        except Exception as e:
            print(f"  {fname:<13} {rname:<20} {type(e).__name__}: {str(e)[:50]}")

# ---- D. frame-boundary sweep ----
hr("D. Frame-boundary dependence: sweep seek offset over one codec frame, chunk vs own-full-decode slice")
def align(full, ch, nominal, window=3000):
    m0 = ch.shape[-1]; best=(None,np.inf,False)
    for d in range(-window,window+1):
        s=nominal+d
        if s<0 or s+m0>full.shape[-1]: continue
        seg=full[:, s:s+m0]
        if seg.shape!=ch.shape: continue
        v=float(np.max(np.abs(seg.astype(np.float64)-ch.astype(np.float64))))
        if v<best[1]: best=(d,v,bool(seg.tobytes()==ch.astype(np.float32).tobytes()))
    return best

# frame sizes: mp3=1152, aac=1024, opus=960 @48k
for fname,(f,fsr,frame) in {"mp3 192k":("d.mp3",SR,1152), "m4a aac":("d.m4a",SR,1024), "opus 48k":("d.opus",48000,960)}.items():
    p=os.path.join(W,f)
    print(f"\n  --- {fname} (frame={frame} samples).  base offset = 1.0 s, then +0..frame in steps")
    for rname,(rfull,rch) in READERS.items():
        try: full=rfull(p)
        except Exception as e:
            print(f"      {rname:<20} full decode {type(e).__name__}"); continue
        base=int(round(SR*1.0*fsr/SR)); nn=2048
        offs=[base+k for k in [0,1,frame//4,frame//2,frame-1,frame,frame+1,2*frame]]
        cells=[]
        for a in offs:
            try:
                ch=rch(p,fsr,a,nn); d,m,ex=align(full,ch,a)
                cells.append(f"+{a-base}:n={ch.shape[1]},off={d},d={m:.1e}{'!' if not ex else ''}")
            except Exception as e: cells.append(f"+{a-base}:{type(e).__name__}")
        print(f"      {rname:<20} " + "  ".join(cells))
print("\n  ('!' marks a chunk that is NOT bit-identical to the slice of that same library's full decode)")
