import numpy as np, warnings
warnings.filterwarnings("ignore")
d=np.load("/Users/satra/.claude/jobs/295c3f8a/tmp/stage1.npz")
x=d["x"].astype(np.float32); fs=int(d["fs"])
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.classification.api import classify_audios
w=classify_audios([Audio(waveform=x[None,:],sampling_rate=fs)],model="yamnet",top_k=521)[0]
st=np.array([v["start"] for v in w])
sc=[{k:val for e in v["label_scores"] for k,val in e.items()} for v in w]
allk=sorted({k for s in sc for k in s})
M=np.array([[s.get(k,0.0) for s in sc] for k in allk])          # 521 x nwin
top=np.argsort(M.max(axis=1))[::-1][:12]
labs=[allk[i] for i in top]; R=M[top]
sil=M[allk.index("Silence")]
np.savez("/Users/satra/.claude/jobs/295c3f8a/tmp/yamnet.npz",sil=sil,start=st,R=R,labels=np.array(labs))
print(f"{len(w)} windows; top-12 by max score:")
for l,r in zip(labs,R): print(f"   {l:24s} max {r.max():.3f} @ {st[np.argmax(r)]+0.48:5.2f}s")
