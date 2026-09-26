import numpy as np
from scipy.signal import find_peaks
d=np.load("/Users/satra/.claude/jobs/295c3f8a/tmp/stage1.npz")
y=np.load("/Users/satra/.claude/jobs/295c3f8a/tmp/yamnet.npz")
Edb=d["Edb"]; fs=int(d["fs"]); dur=float(d["dur"])
sil=y["sil"]; st=y["start"]
mask=np.zeros(len(Edb),bool)
for s,v in zip(st,sil):
    if v>=0.5:
        i,j=int(s*fs),min(len(Edb),int((s+0.96)*fs)); mask[i:j]=True
print(f"YAMNet-silence covers {mask.sum()/fs:.2f} s of {dur:.2f} s ({mask.mean()*100:.0f}%)\n")
f_pct=np.percentile(Edb,10)
f_yam=np.median(Edb[mask]); f_yam95=np.percentile(Edb[mask],95)
print(f"  floor, 10th percentile of whole file : {f_pct:7.2f} dB")
print(f"  floor, median over YAMNet Silence    : {f_yam:7.2f} dB   ({f_yam-f_pct:+.2f} dB)")
print(f"  floor, 95th pct over YAMNet Silence  : {f_yam95:7.2f} dB   ({f_yam95-f_pct:+.2f} dB)")
print(f"  envelope inside silence: min {Edb[mask].min():.1f}  max {Edb[mask].max():.1f} dB")

def spans(floor,peak_db=18.,onset_drop=15.,frac=.7,hang_ms=120,min_ms=50):
    pk,_=find_peaks(Edb,height=floor+peak_db,distance=int(.150*fs)); out=[]
    for p in pk:
        th=Edb[p]-onset_drop; i=p
        while i>0 and Edb[i]>th: i-=1
        tho=Edb[p]-frac*(Edb[p]-floor); hang=int(hang_ms*fs/1000); j=p
        while j<len(Edb)-1:
            w=Edb[j:j+hang]
            if len(w) and w.max()<=tho: break
            j+=1
        if j-i>=min_ms*fs/1000: out.append([i/fs,j/fs])
    out.sort(); m=[]
    for s0,e0 in out:
        if m and s0<=m[-1][1]: m[-1][1]=max(m[-1][1],e0)
        else: m.append([s0,e0])
    return m
for name,fl in (("10th percentile",f_pct),("YAMNet Silence median",f_yam),("YAMNet Silence p95",f_yam95)):
    sp=spans(fl)
    print(f"\n  {name} (floor {fl:.1f} dB) -> {len(sp)} spans, {sum(e-s for s,e in sp):.2f} s")
    print("     "+", ".join(f"{s:.2f}-{e:.2f}" for s,e in sp))
np.save("/Users/satra/.claude/jobs/295c3f8a/tmp/silmask.npy",mask)
