import numpy as np, warnings, json
warnings.filterwarnings("ignore")
d=np.load("/Users/satra/.claude/jobs/295c3f8a/tmp/stage1.npz")
x=d["x"]; fs=int(d["fs"]); dur=float(d["dur"])
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.health_acoustics import detect_health_acoustic_events
from senselab.audio.tasks.health_acoustics.hear import HEAR_EVENT_LABELS
WIN=0.5; HOP=0.1; N2=2*fs; W=int(WIN*fs)
centres=np.arange(0, dur-WIN+1e-9, HOP)
auds=[]
for c in centres:
    i=int(c*fs); seg=x[i:i+W]
    if len(seg)<W: seg=np.pad(seg,(0,W-len(seg)))
    buf=np.zeros(N2); off=(N2-W)//2; buf[off:off+W]=seg      # 500 ms gate inside a 2 s buffer
    auds.append(Audio(waveform=buf.astype(np.float32)[None,:], sampling_rate=fs))
print(f"{len(auds)} gated 2 s windows, {WIN*1000:.0f} ms gate, {HOP*1000:.0f} ms hop",flush=True)
res=detect_health_acoustic_events(auds, hop_length=2.0)
R=np.zeros((len(HEAR_EVENT_LABELS),len(auds)))
for j,per in enumerate(res):
    if not per: continue
    sc={k:v for dd in per[0]["label_scores"] for k,v in dd.items()}
    for i,lab in enumerate(HEAR_EVENT_LABELS): R[i,j]=sc.get(lab,np.nan)
np.savez("/Users/satra/.claude/jobs/295c3f8a/tmp/hear.npz",R=R,centres=centres,labels=np.array(HEAR_EVENT_LABELS))
print("hear raster",R.shape,"max per class:",dict(zip(HEAR_EVENT_LABELS,R.max(axis=1).round(3))),flush=True)
