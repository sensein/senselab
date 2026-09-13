import numpy as np, soundfile as sf, warnings, sys, json
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from labels import LABELS
from scipy.signal import resample_poly, hilbert, butter, filtfilt, stft
T="/Users/satra/.claude/jobs/295c3f8a/tmp/"
y,sr=sf.read("/Users/satra/Downloads/streaming-audio-2026-07-30T04-21-56-487Z.wav",dtype="float32",always_2d=True)
x=resample_poly(y.mean(axis=1),16000,sr).astype(np.float64); fs=16000; dur=len(x)/fs
xp=np.empty_like(x); xp[0]=x[0]; xp[1:]=x[1:]-0.97*x[:-1]
b,a=butter(4,40/(fs/2),"low")
def env(s):
    e=np.maximum(filtfilt(b,a,np.abs(hilbert(s))),1e-12); return 20*np.log10(e/np.abs(e).max())
E=env(xp); t=np.arange(len(E))/fs
s0,_=sf.read(T+"lab_src0.wav",dtype="float64"); s1,_=sf.read(T+"lab_src1.wav",dtype="float64")
n=min(len(x),len(s0)); s0,s1=s0[:n],s1[:n]
E0,E1=env(s0),env(s1); t0=np.arange(len(E0))/fs
yv=np.load(T+"yamnet.npz"); yl=[str(l) for l in yv["labels"]]; YR=yv["R"]; yst=yv["start"]
SPK=[(7.95,9.01),(9.51,10.19),(11.62,13.06)]
SQ={"whole file":(0,dur,0.864,1.34,-12.92),"speech":(11.62,13.20,0.950,1.69,12.40),
    "cough 1":(7.926,8.494,0.443,1.12,-13.68),"exhal 1":(2.2995,3.5205,0.388,1.10,-15.84),
    "silence":(3.6,4.6,0.397,1.23,-13.51)}
fig,ax=plt.subplots(6,1,figsize=(15.5,15),sharex=True,
    gridspec_kw=dict(height_ratios=[1.4,0.85,0.85,1.5,1.4,2.1],hspace=.17))
fig.suptitle("SPEECH branch instruments on the labelled recording — 16 kHz, pre-emphasised\n"
  "the speech label is 11.62–13.20 s: 1.58 s, 11% of the file",fontsize=12,y=.975)
for l in LABELS:
    c={"scored":"#2a7"}.get(l["status"],"#bbb")
    for a_ in ax: a_.axvspan(l["span_lo"],l["span_hi"],color=c,alpha=.10,zorder=0)
sp=[l for l in LABELS if l["name"]=="speech"][0]
for a_ in ax: a_.axvspan(sp["span_lo"],sp["span_hi"],color="#1f4e79",alpha=.16,zorder=0)
ax[0].plot(t,E,lw=.7,c="#b03030"); ax[0].set_ylabel("envelope, dB\n(mixture)"); ax[0].set_ylim(-62,4)
ax[0].annotate("speech label",(12.4,0),ha="center",fontsize=8,c="#1f4e79")
spr=YR[yl.index("Speech")]; sil=YR[yl.index("Silence")]
ax[1].step(yst+.48,spr,where="mid",lw=1.4,c="#1f4e79",label="YAMNet Speech")
ax[1].step(yst+.48,sil,where="mid",lw=1.0,c="#0a7",ls="--",label="YAMNet Silence")
ax[1].axhline(.5,ls=":",c="k",lw=.8); ax[1].set_ylim(-.05,1.08); ax[1].set_ylabel("YAMNet")
ax[1].legend(loc="upper left",fontsize=7.5,ncol=2)
hot=np.where(spr>=.5)[0]
ax[1].annotate(f"Speech>=0.5 spans {yst[hot[0]]:.2f}-{yst[hot[-1]]+0.96:.2f}s\n"
               f"onset overshoots the label by {yst[hot[0]]-sp['span_lo']:+.2f}s",
               xy=(yst[hot[0]],.55),xytext=(4.6,.72),fontsize=7.5,c="#a00",
               arrowprops=dict(arrowstyle="->",color="#a00",lw=.8))
ax[2].set_ylim(0,1); ax[2].set_yticks([]); ax[2].set_ylabel("pyannote\ncommunity-1")
for (s_,e_) in SPK:
    is_speech = s_>11
    ax[2].add_patch(Rectangle((s_,.3),e_-s_,.4,facecolor="#1f4e79" if is_speech else "#c33",
                              edgecolor="k",lw=.6,alpha=.75))
    ax[2].annotate("SPEAKER_00", ((s_+e_)/2,.78),ha="center",fontsize=7)
ax[2].annotate("2 of 3 pyannote segments are coughs (red) — 1.74 s of the 3.18 s it calls speech",
               (0.15,.06),fontsize=7.5,c="#a00")
ax[3].set_ylim(0,1); ax[3].set_yticks([]); ax[3].set_ylabel("SQUIM\nper region")
for k,(a_,b_,st,pe,sd) in SQ.items():
    good = st>=0.8
    ax[3].add_patch(Rectangle((a_,.42),b_-a_,.26,facecolor="#2a7" if good else "#c33",alpha=.35,edgecolor="k",lw=.5))
    ax[3].annotate(f"{k}\nSTOI {st:.3f}\nSI-SDR {sd:+.1f}",((a_+b_)/2,.72),ha="center",fontsize=6.8)
ax[3].annotate("SI-SDR flips sign: -12.92 dB over the file, +12.40 dB over its speech",
               (0.15,.12),fontsize=8,c="#a00")
ax[4].plot(t0,E0,lw=.7,c="#1f4e79",label="MossFormer src0 (speech + exhalations)")
ax[4].plot(t0,E1,lw=.7,c="#c1571a",label="MossFormer src1 (coughs)")
ax[4].set_ylabel("separated\nenvelopes, dB"); ax[4].set_ylim(-95,12); ax[4].legend(loc="upper left",fontsize=7.5,ncol=2,framealpha=.9)
ax[4].annotate("src0 gains +9.17 dB on speech and loses 8-10.5 dB on the coughs; src1 keeps the coughs (-1 to -2 dB) and drops the exhalation by 47 dB",(0.15,-91),fontsize=7.5)
f_,tt,Z=stft(xp,fs,nperseg=80,noverlap=0,boundary=None)
M=np.abs(Z); S=20*np.log10(M+1e-10); S-=S.max()
ax[5].pcolormesh(tt,f_,np.clip(S,-80,0),shading="auto",cmap="magma",vmin=-80,vmax=0)
ax[5].set_ylim(0,8000); ax[5].set_ylabel("wideband spectrogram\n5 ms / 5 ms")
ax[5].set_xlabel("time (s)"); ax[5].set_xlim(0,dur)
fig.text(.5,.006,"blue band = the speech label. green bands = other scored labels. "
  "Separation cannot be scored for benefit here: speech and airway events never overlap in this file.",
  ha="center",fontsize=8.5,c="#444")
fig.savefig("/Users/satra/Downloads/speech_branch_instruments.png",dpi=118,bbox_inches="tight")
print("wrote /Users/satra/Downloads/speech_branch_instruments.png")
