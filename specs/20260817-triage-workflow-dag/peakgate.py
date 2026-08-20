import numpy as np, soundfile as sf, warnings, sys, torch, os
warnings.filterwarnings("ignore")
sys.path.insert(0,"/Users/satra/software/sensein/senselab/.claude/worktrees/design/specs/20260817-triage-workflow-dag")
from labels import label
from scipy.signal import resample_poly, hilbert, butter, filtfilt, find_peaks
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.classification.api import classify_audios
from torchaudio.pipelines import SQUIM_OBJECTIVE
y,sr=sf.read("/Users/satra/Downloads/streaming-audio-2026-07-30T04-21-56-487Z.wav",dtype="float32",always_2d=True)
x=resample_poly(y.mean(axis=1),16000,sr).astype(np.float64); fs=16000
SP=label("speech"); a_,b_=SP["span_lo"],SP["span_hi"]
sig=(x[int(a_*fs):int(b_*fs)]**2).mean()
rng=np.random.default_rng(17); w=rng.standard_normal(len(x)); W=np.fft.rfft(w)
f=np.fft.rfftfreq(len(x),1/fs); W[1:]/=np.sqrt(f[1:]); nz=np.fft.irfft(W,len(x)); nz/=np.sqrt((nz**2).mean())
LEV=[None,20,10,5,0,-5]; mixes=[];auds=[]
for L in LEV:
    m=x.copy() if L is None else x+nz*np.sqrt(sig/(10**(L/10)))
    m=m/max(1.0,np.abs(m).max()); mixes.append(m)
    auds.append(Audio(waveform=m[None,:].astype(np.float32),sampling_rate=fs))
res=classify_audios(auds,model="yamnet",top_k=521)
mo=SQUIM_OBJECTIVE.get_model()
def squim(s):
    s=np.asarray(s,dtype=np.float32)
    if len(s)<fs//2: s=np.pad(s,(0,fs//2-len(s)))
    with torch.no_grad(): u,v,w2=mo(torch.from_numpy(s).unsqueeze(0))
    return float(u.item()),float(w2.item())
bb,aa=butter(4,40/(fs/2),"low")
K=float(os.environ.get("K","25"))
print(f"propose gate = peak - {K:.0f} dB  (peak-anchored, floor-independent)\n")
print(f"  {'SNR':>6s} {'floor':>7s} {'gate':>7s} {'spans':>6s} {'span over speech':>18s} {'IoU':>5s} {'cov':>5s} {'STOI':>6s} {'SI-SDR':>7s} {'verdict':>10s}")
for L,m,w_ in zip(LEV,mixes,res):
    xp=np.empty_like(m); xp[0]=m[0]; xp[1:]=m[1:]-0.97*m[:-1]
    E=np.maximum(filtfilt(bb,aa,np.abs(hilbert(xp))),1e-12); Edb=20*np.log10(E/E.max())
    sc={win["start"]:{k:v for e in win["label_scores"] for k,v in e.items()} for win in w_}
    st=sorted(sc); sil=np.array([sc[s].get("Silence",0.) for s in st])
    mask=np.zeros(len(Edb),bool)
    for s,v in zip(st,sil):
        if v>=.5: mask[int(s*fs):min(len(Edb),int((s+.96)*fs))]=True
    floor=float(np.median(Edb[mask])) if mask.any() else float(np.percentile(Edb,10))
    gate=-K
    if gate<=floor:
        print(f"  {('orig' if L is None else f'{L:+d}'):>6s} {floor:7.1f} {gate:7.1f} {'--':>6s} {'GATE BELOW FLOOR':>18s} {'-':>5s} {'-':>5s} {'-':>6s} {'-':>7s} {'FAIL':>10s}")
        continue
    pk,_=find_peaks(Edb,height=gate,distance=int(.150*fs)); sp=[]
    for p in pk:
        th=Edb[p]-15.; i=p
        while i>0 and Edb[i]>th: i-=1
        tho=Edb[p]-.7*(Edb[p]-floor); hang=int(.120*fs); j=p
        while j<len(Edb)-1:
            ww=Edb[j:j+hang]
            if len(ww) and ww.max()<=tho: break
            j+=1
        if j-i>=.050*fs: sp.append([i/fs,j/fs])
    sp.sort(); mg=[]
    for s0,e0 in sp:
        if mg and s0<=mg[-1][1]: mg[-1][1]=max(mg[-1][1],e0)
        else: mg.append([s0,e0])
    hit=[s for s in mg if s[0]<b_ and s[1]>a_]
    if not hit:
        print(f"  {('orig' if L is None else f'{L:+d}'):>6s} {floor:7.1f} {gate:7.1f} {len(mg):6d} {'NONE':>18s} {'-':>5s} {'-':>5s} {'-':>6s} {'-':>7s} {'FAIL':>10s}")
        continue
    s0,e0=max(hit,key=lambda s:min(s[1],b_)-max(s[0],a_))
    iou=max(0,min(e0,b_)-max(s0,a_))/(max(e0,b_)-min(s0,a_))
    stq,sd=squim(m[int(s0*fs):int(e0*fs)])
    ov=[sc[s].get("Speech",0.) for s in st if s<e0 and s+0.96>s0]
    cov=float(np.mean([v>=.5 for v in ov])) if ov else 0.
    vd="SPEECH" if (cov>0 and stq>=.8) else ("uncertain" if (cov>0)!=(stq>=.8) else "not speech")
    print(f"  {('orig' if L is None else f'{L:+d}'):>6s} {floor:7.1f} {gate:7.1f} {len(mg):6d} {f'{s0:.2f}-{e0:.2f}':>18s} {iou:5.2f} {cov*100:4.0f}% {stq:6.3f} {sd:+7.2f} {vd:>10s}")
