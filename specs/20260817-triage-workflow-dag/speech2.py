import numpy as np, warnings, soundfile as sf, json, torch, sys
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from labels import LABELS
from scipy.signal import resample_poly
from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import HFModel
F="/Users/satra/Downloads/streaming-audio-2026-07-30T04-21-56-487Z.wav"; T="/Users/satra/.claude/jobs/295c3f8a/tmp/"
y,sr=sf.read(F,dtype="float32",always_2d=True); x=resample_poly(y.mean(axis=1),16000,sr).astype(np.float32)
fs=16000; dur=len(x)/fs
SPEECH=[l for l in LABELS if l["name"]=="speech"][0]
print(f"labelled recording: {dur:.2f}s | speech label {SPEECH['span_lo']:.2f}-{SPEECH['span_hi']:.2f}s "
      f"({SPEECH['span_hi']-SPEECH['span_lo']:.2f}s = {(SPEECH['span_hi']-SPEECH['span_lo'])/dur*100:.0f}% of file)",flush=True)

from torchaudio.pipelines import SQUIM_OBJECTIVE
m=SQUIM_OBJECTIVE.get_model()
def squim(sig):
    s=np.asarray(sig,dtype=np.float32)
    if len(s)<fs//2: s=np.pad(s,(0,fs//2-len(s)))
    with torch.no_grad(): a,b,c=m(torch.from_numpy(s).unsqueeze(0))
    return float(a.item()),float(b.item()),float(c.item())

print("\nSQUIM: whole file vs the speech region alone vs a cough alone",flush=True)
regions={"whole file":(0,dur),
         "speech label":(SPEECH["span_lo"],SPEECH["span_hi"]),
         "cough 1":(7.926,8.494),"exhalation 1":(2.2995,3.5205),
         "silence 3.6-4.6":(3.6,4.6)}
for name,(a,b) in regions.items():
    st,pe,sd=squim(x[int(a*fs):int(b*fs)])
    print(f"  {name:18s} {b-a:5.2f}s   STOI {st:.3f}   PESQ {pe:.2f}   SI-SDR {sd:+7.2f}",flush=True)

A=Audio(waveform=x[None,:],sampling_rate=fs)
from senselab.audio.tasks.speaker_diarization import diarize_audios
try:
    dia=diarize_audios([A])[0]
    print(f"\npyannote on the whole file: {len(dia)} segment(s)",flush=True)
    for seg in dia:
        lab=getattr(seg,"speaker",None) or getattr(seg,"label",None)
        print(f"   {float(seg.start):6.2f}-{float(seg.end):6.2f}s  {lab}",flush=True)
except Exception as e:
    print(f"\npyannote FAILED: {type(e).__name__}: {e}",flush=True)

from senselab.audio.tasks.source_separation import separate_audios
try:
    outs=separate_audios([A],model=HFModel(path_or_uri="alibabasglab/MossFormer2_SS_16K",revision="main"),n_sources=2)[0]
    print(f"\nMossFormer2_SS_16K -> {len(outs)} streams",flush=True)
    for i,o in enumerate(outs):
        w=o.waveform.squeeze().cpu().numpy() if hasattr(o.waveform,"cpu") else np.asarray(o.waveform).squeeze()
        sf.write(T+f"lab_src{i}.wav",w,fs)
        sp=squim(w[int(SPEECH['span_lo']*fs):int(SPEECH['span_hi']*fs)])
        cg=squim(w[int(7.926*fs):int(8.494*fs)])
        print(f"   src{i}: RMS {20*np.log10(np.sqrt((w**2).mean())+1e-12):7.2f} dB | "
              f"over speech STOI {sp[0]:.3f} PESQ {sp[1]:.2f} | over cough1 STOI {cg[0]:.3f} PESQ {cg[1]:.2f}",flush=True)
except Exception as e:
    print(f"\nMossFormer FAILED: {type(e).__name__}: {e}",flush=True)
