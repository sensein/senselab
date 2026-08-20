import numpy as np, soundfile as sf, warnings, sys
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from labels import SCORED, label
from scipy.signal import resample_poly, hilbert, butter, filtfilt
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.classification.api import classify_audios
y,sr=sf.read("/Users/satra/Downloads/streaming-audio-2026-07-30T04-21-56-487Z.wav",dtype="float32",always_2d=True)
x=resample_poly(y.mean(axis=1),16000,sr).astype(np.float64); fs=16000
SP=label("speech"); sig=(x[int(SP["span_lo"]*fs):int(SP["span_hi"]*fs)]**2).mean()
rng=np.random.default_rng(17); w=rng.standard_normal(len(x)); W=np.fft.rfft(w)
f=np.fft.rfftfreq(len(x),1/fs); W[1:]/=np.sqrt(f[1:]); nz=np.fft.irfft(W,len(x)); nz/=np.sqrt((nz**2).mean())
LEV=[None,20,10,5,0,-5]; mixes=[]; auds=[]
for L in LEV:
    m=x.copy() if L is None else x+nz*np.sqrt(sig/(10**(L/10)))
    m=m/max(1.0,np.abs(m).max()); mixes.append(m)
    auds.append(Audio(waveform=m[None,:].astype(np.float32),sampling_rate=fs))
res=classify_audios(auds,model="yamnet",top_k=521)
bb,aa=butter(4,40/(fs/2),"low")
print("peak above the silence floor, per event, as noise rises (dB)\n")
names=[l["name"][:14] for l in SCORED]
print(f"  {'SNR':>7s} {'floor':>7s} "+"".join(f"{n:>15s}" for n in names)+f"{'speech margin':>15s}")
for L,m,w_ in zip(LEV,mixes,res):
    xp=np.empty_like(m); xp[0]=m[0]; xp[1:]=m[1:]-0.97*m[:-1]
    E=np.maximum(filtfilt(bb,aa,np.abs(hilbert(xp))),1e-12); Edb=20*np.log10(E/E.max())
    sc={win["start"]:{k:v for e in win["label_scores"] for k,v in e.items()} for win in w_}
    st=sorted(sc); sil=np.array([sc[s].get("Silence",0.0) for s in st])
    mask=np.zeros(len(Edb),bool)
    for s,v in zip(st,sil):
        if v>=.5: mask[int(s*fs):min(len(Edb),int((s+.96)*fs))]=True
    floor=float(np.median(Edb[mask])) if mask.any() else float(np.percentile(Edb,10))
    vals=[]
    for l in SCORED:
        i,j=int(l["span_lo"]*fs),int(l["span_hi"]*fs)
        vals.append(Edb[i:j].max()-floor)
    sp_i=[k for k,l in enumerate(SCORED) if l["name"]=="speech"][0]
    others=[v for k,v in enumerate(vals) if k!=sp_i]
    margin=vals[sp_i]  # speech contrast is what a gate must not exceed
    print(f"  {('orig' if L is None else f'{L:+d}'):>7s} {floor:7.1f} "+"".join(f"{v:15.1f}" for v in vals)+f"{margin:15.1f}")
print("\n  a gate must sit BELOW the speech column to propose speech at all.")
print("  the airway columns show how much headroom a lower gate gives away.")
