import numpy as np, warnings
warnings.filterwarnings("ignore")
T="/Users/satra/.claude/jobs/295c3f8a/tmp/"
d=np.load(T+"stage1.npz"); h=np.load(T+"hear.npz")
x=d["x"].astype(np.float32); fs=int(d["fs"])
hl=[str(l) for l in h["labels"]]; HR=h["R"]; cen=h["centres"]
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.health_acoustics import detect_health_acoustic_events
from senselab.audio.tasks.classification.api import classify_audios
spans=[(2.32,3.29),(5.32,6.22),(7.92,8.51),(9.61,9.96),(11.75,13.16)]
N2=2*fs
auds=[];yaud=[]
for a,b in spans:
    seg=x[int(a*fs):int(b*fs)]
    buf=np.zeros(N2,np.float32); off=(N2-len(seg))//2; buf[off:off+len(seg)]=seg
    auds.append(Audio(waveform=buf[None,:],sampling_rate=fs))
    ys=seg if len(seg)>=int(0.96*fs) else np.pad(seg,(0,int(0.96*fs)-len(seg)))
    yaud.append(Audio(waveform=ys.astype(np.float32)[None,:],sampling_rate=fs))
hres=detect_health_acoustic_events(auds,hop_length=2.0)
yres=classify_audios(yaud,model="yamnet",top_k=521)
print(f"{'span':13s} {'dur':>7s} | {'HeAR whole-span input':>34s} | {'HeAR sweep coverage':>24s} | {'YAMNet whole-span':>26s}")
for (a,b),hr,yr in zip(spans,hres,yres):
    hs={k:v for e in hr[0]["label_scores"] for k,v in e.items()}
    top=sorted(hs.items(),key=lambda kv:-kv[1])[:2]
    ys={}
    for w in yr:
        for e in w["label_scores"]:
            for k,v in e.items(): ys[k]=max(ys.get(k,0),v)
    ytop=sorted(ys.items(),key=lambda kv:-kv[1])[:2]
    m=(cen<b)&(cen+0.5>a)
    cov={l:( (HR[hl.index(l)][m]>=0.5).mean() ) for l in ("Cough","Breathe")}
    cl=max(cov,key=cov.get)
    print(f"{a:5.2f}-{b:5.2f} {(b-a)*1000:6.0f}ms | "
          f"{top[0][0]:>13s} {top[0][1]:.3f}  ({top[1][0]} {top[1][1]:.2f}) | "
          f"{cl:>10s} {cov[cl]*100:5.0f}% | {ytop[0][0]:>14s} {ytop[0][1]:.3f} ({ytop[1][0][:9]} {ytop[1][1]:.2f})")
