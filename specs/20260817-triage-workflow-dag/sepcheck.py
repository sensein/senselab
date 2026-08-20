import numpy as np, soundfile as sf, warnings
warnings.filterwarnings("ignore")
from scipy.signal import resample_poly
T="/Users/satra/.claude/jobs/295c3f8a/tmp/"
y,sr=sf.read("/Users/satra/Downloads/streaming-audio-2026-07-30T04-21-56-487Z.wav",dtype="float32",always_2d=True)
mix=resample_poly(y.mean(axis=1),16000,sr).astype(np.float64); fs=16000
s0,_=sf.read(T+"lab_src0.wav",dtype="float64"); s1,_=sf.read(T+"lab_src1.wav",dtype="float64")
n=min(len(mix),len(s0),len(s1)); mix,s0,s1=mix[:n],s0[:n],s1[:n]
print(f"lengths equal: {len(s0)==len(s1)}  n={n} ({n/fs:.2f}s)")
print(f"src0 == src1 exactly: {np.array_equal(s0,s1)}")
print(f"corr(src0,src1) = {np.corrcoef(s0,s1)[0,1]:+.4f}")
print(f"RMS  src0 {20*np.log10(np.sqrt((s0**2).mean())):.3f} dB   src1 {20*np.log10(np.sqrt((s1**2).mean())):.3f} dB")
print(f"max|src0-src1| = {np.abs(s0-s1).max():.6f}   max|mix| = {np.abs(mix).max():.4f}")
print(f"corr(src0,mix) {np.corrcoef(s0,mix)[0,1]:+.4f}   corr(src1,mix) {np.corrcoef(s1,mix)[0,1]:+.4f}")
print(f"\nper-region RMS (dB) -- does either stream concentrate one kind?")
R={"speech 11.62-13.20":(11.62,13.20),"cough1 7.93-8.49":(7.926,8.494),
   "cough2 9.61-10.25":(9.610,10.250),"exhal1 2.30-3.52":(2.2995,3.5205),"silence 3.6-4.6":(3.6,4.6)}
print(f"  {'region':22s} {'mix':>8s} {'src0':>8s} {'src1':>8s} {'s0-mix':>8s} {'s1-mix':>8s}")
for k,(a,b) in R.items():
    i,j=int(a*fs),int(b*fs)
    f=lambda v: 20*np.log10(np.sqrt((v[i:j]**2).mean())+1e-12)
    print(f"  {k:22s} {f(mix):8.2f} {f(s0):8.2f} {f(s1):8.2f} {f(s0)-f(mix):+8.2f} {f(s1)-f(mix):+8.2f}")
print(f"\nsum check: corr(src0+src1, mix) = {np.corrcoef(s0+s1,mix)[0,1]:+.4f}")
