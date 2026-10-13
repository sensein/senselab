import numpy as np, soundfile as sf, warnings, sys
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from labels import label, SCORED
from scipy.signal import resample_poly, hilbert, butter, filtfilt
fs=16000; bb,aa=butter(4,40/(fs/2),"low")
def env(f, add_transient=False):
    y,sr=sf.read(f,dtype="float32",always_2d=True)
    x=resample_poly(y.mean(axis=1),fs,sr).astype(np.float64)
    if add_transient:                      # a 30 ms full-scale click at 1.0 s -- a door slam
        i=int(1.0*fs); x=x.copy(); x[i:i+int(.03*fs)] += 0.95*np.sign(np.random.default_rng(3).standard_normal(int(.03*fs)))
    xp=np.empty_like(x); xp[0]=x[0]; xp[1:]=x[1:]-0.97*x[:-1]
    E=np.maximum(filtfilt(bb,aa,np.abs(hilbert(xp))),1e-12)
    return x, 20*np.log10(E/E.max())
LAB="/Users/satra/Downloads/streaming-audio-2026-07-30T04-21-56-487Z.wav"
HAP="/Users/satra/Downloads/HAPPY_ASK.wav"

print("1. does the peak vary WITHIN a recording? sliding 1 s window max, relative to the global peak\n")
for name,f in (("labelled (coughs+speech)",LAB),("HAPPY_ASK (speech only)",HAP)):
    x,Edb=env(f); w=int(1.0*fs)
    loc=np.array([Edb[i:i+w].max() for i in range(0,len(Edb)-w,w//2)])
    print(f"  {name:26s} global 0.0 dB | local peaks: min {loc.min():6.1f}  median {np.median(loc):6.1f}  max {loc.max():6.1f}  spread {loc.max()-loc.min():5.1f} dB")

print("\n2. where does the global peak sit relative to the SPEECH peak?\n")
x,Edb=env(LAB); sp=label("speech")
sp_pk=Edb[int(sp['span_lo']*fs):int(sp['span_hi']*fs)].max()
print(f"  labelled file: global peak 0.0 dB, speech peak {sp_pk:.1f} dB  -> speech sits {-sp_pk:.1f} dB below peak")
xh,Eh=env(HAP)
print(f"  HAPPY_ASK: global peak 0.0 dB, and the peak IS speech -> speech sits 0.0 dB below peak")
print(f"     so `peak - 25 dB` is {-25 - sp_pk:+.1f} dB relative to the speech peak on the labelled file,")
print(f"     and -25.0 dB relative to it on HAPPY_ASK. The same constant means two different things.")

print("\n3. what one loud transient does to a peak-anchored gate\n")
x2,E2=env(LAB, add_transient=True)
sp_pk2=E2[int(sp['span_lo']*fs):int(sp['span_hi']*fs)].max()
print(f"  with a 30 ms full-scale click injected at 1.0 s:")
print(f"    speech peak moves {sp_pk:.1f} -> {sp_pk2:.1f} dB below the new global peak  (shift {sp_pk2-sp_pk:+.1f} dB)")
print(f"    a `peak - 25 dB` gate now sits {(-25)-sp_pk2:+.1f} dB relative to the speech peak")
print(f"    -> speech {'still proposable' if sp_pk2 > -25 else 'NO LONGER PROPOSABLE'}")
print("\n4. a robust alternative: high percentile instead of max\n")
for name,f in (("labelled",LAB),("HAPPY_ASK",HAP)):
    x,Edb=env(f)
    x2,E2=env(f,add_transient=True)
    for q in (99.9,99.0,95.0):
        a=np.percentile(Edb,q); b=np.percentile(E2,q)
        print(f"  {name:10s} p{q:<5} clean {a:7.2f}  with click {b:7.2f}   shift {b-a:+6.2f} dB")
    print(f"  {name:10s} max    clean    0.00  with click    0.00   shift  (by definition 0, the click IS the max)")
