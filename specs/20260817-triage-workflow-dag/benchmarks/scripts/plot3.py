import numpy as np, warnings, torch
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
T="/Users/satra/.claude/jobs/295c3f8a/tmp/"
d=np.load(T+"stage1.npz"); h=np.load(T+"hear.npz"); y=np.load(T+"yamnet.npz")
fs=int(d["fs"]); dur=float(d["dur"]); x=d["x"]; xp=d["xp"]; Edb=d["Edb"]; t=d["t"]
hl=[str(l) for l in h["labels"]]; HR=h["R"]; cen=h["centres"]
yl=[str(l) for l in y["labels"]]; YR=y["R"]; sil=y["sil"]; sst=y["start"]
mask=np.zeros(len(Edb),bool)
for s,v in zip(sst,sil):
    if v>=.5: mask[int(s*fs):min(len(Edb),int((s+.96)*fs))]=True
floor=float(np.median(Edb[mask])); f_pct=float(np.percentile(Edb,10))
pk,_=find_peaks(Edb,height=floor+18.,distance=int(.150*fs)); sp=[]
for p in pk:
    th=Edb[p]-15.; i=p
    while i>0 and Edb[i]>th: i-=1
    tho=Edb[p]-.7*(Edb[p]-floor); hang=int(.120*fs); j=p
    while j<len(Edb)-1:
        w=Edb[j:j+hang]
        if len(w) and w.max()<=tho: break
        j+=1
    if j-i>=.050*fs: sp.append([i/fs,j/fs])
sp.sort(); m=[]
for s0,e0 in sp:
    if m and s0<=m[-1][1]: m[-1][1]=max(m[-1][1],e0)
    else: m.append([s0,e0])
sp=m
peak=20*np.log10(np.abs(x).max()); rms=20*np.log10(np.sqrt((x**2).mean()))
try:
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    mm=SQUIM_OBJECTIVE.get_model()
    with torch.no_grad(): a,b,c=mm(torch.from_numpy(x.astype(np.float32)).unsqueeze(0))
    sq=f"STOI {a.item():.3f}  PESQ {b.item():.2f}  SI-SDR {c.item():+.1f} dB"
except Exception: sq="n/a"
try:
    import pyloudnorm as pyln; lu=f"{pyln.Meter(fs).integrated_loudness(x):+.1f} LUFS"
except Exception: lu="n/a"
keep=[i for i,l in enumerate(hl) if l in ("Cough","Breathe")]
fig,ax=plt.subplots(7,1,figsize=(16,18.5),sharex=True,
    gridspec_kw=dict(height_ratios=[1.0,1.6,0.7,2.6,2.3,2.3,1.0],hspace=.15))
fig.suptitle("PREPROCESS derivatives, aligned — 16 kHz, pre-emphasis a=0.97 (switchable)\n"
  f"level (plain): peak {peak:.1f} dBFS   RMS {rms:.1f} dBFS   {lu}      squim (plain): {sq}",fontsize=11.5,y=.987)
tw=np.arange(len(x))/fs
ax[0].plot(tw,x,lw=.3,c="#999",label="plain"); ax[0].plot(tw,xp,lw=.3,c="#1f4e79",label="pre-emphasised")
ax[0].set_ylabel("waveform"); ax[0].legend(loc="upper right",fontsize=8,ncol=2); ax[0].set_ylim(-1,1)
ax[1].plot(t,Edb,lw=.8,c="#b03030")
ax[1].axhline(floor,c="#0a7",lw=1.3); ax[1].axhline(floor+18,ls="--",c="#0a7",lw=.9); ax[1].axhline(f_pct,ls=":",c="k",lw=1)
ax[1].text(dur-.15,floor+18.6,"+18 dB peak-proposal gate",fontsize=7.5,c="#065",ha="right")
ax[1].text(dur-.15,floor+1.0,f"floor = median over YAMNet Silence = {floor:.1f} dB",fontsize=7.5,c="#065",ha="right")
ax[1].text(dur-.15,f_pct-3.0,f"10th-percentile floor {f_pct:.1f} dB (replaced)",fontsize=7.5,ha="right")
ax[1].set_ylabel("Hilbert envelope\n40 Hz LP, dB"); ax[1].set_ylim(floor-11,6)
ax[2].fill_between(t,0,mask.astype(float),step="mid",color="#0a7",alpha=.35)
ax[2].plot(sst+.48,sil,"o-",ms=3,lw=.8,c="#054"); ax[2].axhline(.5,ls="--",c="k",lw=.7)
ax[2].set_ylim(-.05,1.05); ax[2].set_ylabel("YAMNet\nSilence")
ax[2].text(.05,.58,"0.5 threshold sits in an empty gap (all scores <=0.36 or >=0.62)",fontsize=7)
for k,(s0,e0) in enumerate(sp):
    for a_ in ax: a_.axvspan(s0,e0,color="#ffb000",alpha=.16,zorder=0)
    ax[1].annotate(f"S{k} {(e0-s0)*1000:.0f} ms",((s0+e0)/2,3.0),ha="center",fontsize=7.5,c="#7a4a00")
ax[1].annotate("mouth click — not proposed under this floor",xy=(.87,floor+16),xytext=(1.6,floor-8.0),
    fontsize=7.5,c="#a00",arrowprops=dict(arrowstyle="->",color="#a00",lw=.9))
im1=ax[3].pcolormesh(sst+.48,np.arange(len(yl)),YR,shading="auto",cmap="cividis",vmin=0,vmax=1)
ax[3].set_yticks(np.arange(len(yl))); ax[3].set_yticklabels(yl,fontsize=8)
ax[3].set_ylabel("YAMNet top-12\n0.96 s win / 0.48 s hop")
ax[4].pcolormesh(d["t_wb"],d["f_wb"],np.clip(d["Swb"],-80,0),shading="auto",cmap="magma",vmin=-80,vmax=0)
ax[4].set_ylabel("wideband spectrogram\n5 ms win / 5 ms hop"); ax[4].set_ylim(0,8000)
cf=d["cf"]
ax[5].pcolormesh(d["t_g"],np.arange(len(cf)),np.clip(d["Gdb"],-80,0),shading="auto",cmap="viridis",vmin=-80,vmax=0)
tk=[0,9,19,29,39]; ax[5].set_yticks(tk); ax[5].set_yticklabels([f"{cf[i]:.0f}" for i in tk],fontsize=8)
ax[5].set_ylabel("gammatone\n40 ERB ch (Hz)")
for i,c in zip(keep,("#e8402a","#12a4d9")):
    ax[6].plot(cen+.25,HR[i],lw=1.6,c=c,label=f"HeAR {hl[i]}")
ax[6].set_ylim(-.03,1.03); ax[6].set_ylabel("HeAR\n500 ms gate"); ax[6].legend(loc="upper left",fontsize=8,ncol=2)
ax[6].set_xlabel("time (s)"); ax[6].set_xlim(0,dur)
cb=fig.colorbar(im1,ax=ax[6],orientation="horizontal",pad=.30,fraction=.05,aspect=60)
cb.set_label("YAMNet presence probability",fontsize=8)
fig.text(.5,.005,"spans from the envelope alone, floor set by YAMNet Silence: peaks >= floor+18 dB, onset at peak-15 dB, "
  "offset at 0.7 of event range, 120 ms hangover. HeAR shown for Cough and Breathe only. No labels used.",
  ha="center",fontsize=9,c="#7a4a00")
fig.savefig("/Users/satra/Downloads/preprocess_derivatives.png",dpi=115,bbox_inches="tight")
print(f"floor {floor:.2f} dB | {len(sp)} spans: "+", ".join(f"{a:.2f}-{b:.2f}" for a,b in sp))
