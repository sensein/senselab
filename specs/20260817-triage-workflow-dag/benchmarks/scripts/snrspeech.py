import numpy as np, soundfile as sf, warnings, torch, sys, json
warnings.filterwarnings("ignore")
sys.path.insert(0,"/Users/satra/software/sensein/agentharness/scripts")
from score_against_labels import LABELS
from scipy.signal import resample_poly, hilbert, butter, filtfilt, find_peaks
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.classification.api import classify_audios
T="/Users/satra/.claude/jobs/295c3f8a/tmp/"
y,sr=sf.read("/Users/satra/Downloads/streaming-audio-2026-07-30T04-21-56-487Z.wav",dtype="float32",always_2d=True)
x=resample_poly(y.mean(axis=1),16000,sr).astype(np.float64); fs=16000; dur=len(x)/fs
SP=[l for l in LABELS if l["name"]=="speech"][0]; a_,b_=SP["span_lo"],SP["span_hi"]
sig_pow=(x[int(a_*fs):int(b_*fs)]**2).mean()
rng=np.random.default_rng(17)
# pink-ish noise: white through a 1/f filter, so it is not trivially separable by spectrum
w=rng.standard_normal(len(x)); W=np.fft.rfft(w); f=np.fft.rfftfreq(len(x),1/fs)
W[1:]/=np.sqrt(f[1:]); noise=np.fft.irfft(W,len(x)); noise/=np.sqrt((noise**2).mean())
LEV=[None,20,10,5,0,-5]
print(f"GATE={__import__('os').environ.get('GATE','18')} dB")
print(f"speech label {a_:.2f}-{b_:.2f}s | SNR is defined over the speech span, pink noise, seed 17\n")
auds=[];mixes=[]
for L in LEV:
    if L is None: m=x.copy()
    else:
        npow=sig_pow/(10**(L/10)); m=x+noise*np.sqrt(npow)
    m=m/max(1.0,np.abs(m).max())      # avoid clipping, record that we rescaled
    mixes.append(m); auds.append(Audio(waveform=m[None,:].astype(np.float32),sampling_rate=fs))
res=classify_audios(auds,model="yamnet",top_k=521)
from torchaudio.pipelines import SQUIM_OBJECTIVE
mo=SQUIM_OBJECTIVE.get_model()
def squim(s):
    s=np.asarray(s,dtype=np.float32)
    if len(s)<fs//2: s=np.pad(s,(0,fs//2-len(s)))
    with torch.no_grad(): u,v,w2=mo(torch.from_numpy(s).unsqueeze(0))
    return float(u.item()),float(v.item()),float(w2.item())
bb,aa=butter(4,40/(fs/2),"low")
out=[]
print(f"  {'SNR':>7s} {'floor':>7s} {'spans':>6s} {'span over speech?':>18s} {'IoU':>6s} {'YAM cov':>8s} {'YAM max':>8s} {'STOI':>6s} {'SI-SDR':>7s} {'verdict':>9s}")
for L,m,w_ in zip(LEV,mixes,res):
    xp=np.empty_like(m); xp[0]=m[0]; xp[1:]=m[1:]-0.97*m[:-1]
    E=np.maximum(filtfilt(bb,aa,np.abs(hilbert(xp))),1e-12); Edb=20*np.log10(E/E.max())
    sc={}; 
    for win in w_:
        d={k:v for e in win["label_scores"] for k,v in e.items()}
        sc[win["start"]]=d
    sil=np.array([sc[s].get("Silence",0.0) for s in sorted(sc)]); st=np.array(sorted(sc))
    mask=np.zeros(len(Edb),bool)
    for s,v in zip(st,sil):
        if v>=.5: mask[int(s*fs):min(len(Edb),int((s+.96)*fs))]=True
    floor=float(np.median(Edb[mask])) if mask.any() else float(np.percentile(Edb,10))
    GATE=float(__import__("os").environ.get("GATE","18"))
    pk,_=find_peaks(Edb,height=floor+GATE,distance=int(.150*fs)); sp=[]
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
    if hit:
        s0,e0=max(hit,key=lambda s:min(s[1],b_)-max(s[0],a_))
        inter=max(0,min(e0,b_)-max(s0,a_)); union=max(e0,b_)-min(s0,a_); iou=inter/union
        stq,pq,sd=squim(m[int(s0*fs):int(e0*fs)])
        ov=[sc[s].get("Speech",0.0) for s in st if s<e0 and s+0.96>s0]
        cov=float(np.mean([v>=0.5 for v in ov])) if ov else 0.0; mx=max(ov) if ov else 0.0
        verdict = "SPEECH" if (cov>0 and stq>=0.8) else ("uncertain" if (cov>0)!=(stq>=0.8) else "not speech")
        print(f"  {('orig' if L is None else f'{L:+d} dB'):>7s} {floor:7.1f} {len(mg):6d} {f'{s0:.2f}-{e0:.2f}':>18s} {iou:6.2f} {cov*100:7.0f}% {mx:8.3f} {stq:6.3f} {sd:+7.2f} {verdict:>9s}")
    else:
        print(f"  {('orig' if L is None else f'{L:+d} dB'):>7s} {floor:7.1f} {len(mg):6d} {'NONE':>18s} {'-':>6s} {'-':>8s} {'-':>8s} {'-':>6s} {'-':>7s} {'FAIL':>9s}")
