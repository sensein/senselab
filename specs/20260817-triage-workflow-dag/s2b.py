import numpy as np
from scipy.signal import find_peaks
d=np.load("/Users/satra/.claude/jobs/295c3f8a/tmp/stage1.npz")
Edb=d["Edb"]; floor=float(d["floor"]); fs=int(d["fs"])
pk,_=find_peaks(Edb,height=floor+18.0,distance=int(0.150*fs))
out=[]
for p in pk:
    th=Edb[p]-15.0; i=p
    while i>0 and Edb[i]>th: i-=1
    tho=Edb[p]-0.7*(Edb[p]-floor); hang=int(0.120*fs); j=p
    while j<len(Edb)-1:
        w=Edb[j:j+hang]
        if len(w) and w.max()<=tho: break
        j+=1
    if (j-i)>=0.050*fs: out.append([i/fs,j/fs,Edb[p]-floor])
out.sort(); m=[]
for s0,e0,pd in out:
    if m and s0<=m[-1][1]: m[-1]=[m[-1][0],max(m[-1][1],e0),max(m[-1][2],pd)]
    else: m.append([s0,e0,pd])
np.save("/Users/satra/.claude/jobs/295c3f8a/tmp/spans_v2b.npy",np.array(m))
print(f"{len(m)} spans, {sum(e-s for s,e,_ in m):.2f} s covered")
