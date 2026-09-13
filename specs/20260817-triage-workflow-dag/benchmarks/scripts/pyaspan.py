import numpy as np, soundfile as sf, warnings, sys
warnings.filterwarnings("ignore")
sys.path.insert(0,"/Users/satra/software/sensein/senselab/.claude/worktrees/design/specs/20260817-triage-workflow-dag")
from labels import label
from scipy.signal import resample_poly
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_diarization import diarize_audios
y,sr=sf.read("/Users/satra/Downloads/streaming-audio-2026-07-30T04-21-56-487Z.wav",dtype="float32",always_2d=True)
x=resample_poly(y.mean(axis=1),16000,sr).astype(np.float32); fs=16000
SPAN=(11.75,13.16)              # the speech span PREPROCESS proposes
LAB=label("speech")
cases={
 "whole file (14.03 s)": (0.0, len(x)/fs),
 "speech span only (1.41 s)": SPAN,
 "span +/- 0.5 s pad (2.41 s)": (SPAN[0]-0.5, SPAN[1]+0.5),
 "span +/- 1.0 s pad (3.41 s)": (SPAN[0]-1.0, SPAN[1]+1.0),
 "label extent (1.58 s)": (LAB["span_lo"], LAB["span_hi"]),
}
for name,(a,b) in cases.items():
    a=max(0,a); b=min(len(x)/fs,b)
    clip=x[int(a*fs):int(b*fs)]
    try:
        d=diarize_audios([Audio(waveform=clip[None,:],sampling_rate=fs)])[0]
        segs=[(float(s.start)+a,float(s.end)+a,getattr(s,"speaker",None) or getattr(s,"label",None)) for s in d]
        spk=sorted({s[2] for s in segs})
        print(f"  {name:30s} -> {len(segs)} seg, {len(spk)} speaker(s) {spk}")
        for s0,e0,lb in segs: print(f"      {s0:6.2f}-{e0:6.2f}s  {lb}")
    except Exception as e:
        print(f"  {name:30s} -> FAILED {type(e).__name__}: {e}")
