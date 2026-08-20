import numpy as np, warnings, torch
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d=np.load("/Users/satra/.claude/jobs/295c3f8a/tmp/stage1.npz")
h=np.load("/Users/satra/.claude/jobs/295c3f8a/tmp/hear.npz")
fs=int(d["fs"]); dur=float(d["dur"]); x=d["x"]; xp=d["xp"]
import numpy as _np
spans=_np.load("/Users/satra/.claude/jobs/295c3f8a/tmp/spans_v2b.npy"); Edb=d["Edb"]; floor=float(d["floor"]); t=d["t"]
labels=[str(l) for l in h["labels"]]; R=h["R"]; cen=h["centres"]

# level + squim on the PLAIN signal, as the spec requires
peak=20*np.log10(np.abs(x).max()); rms=20*np.log10(np.sqrt((x**2).mean()))
try:
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    m=SQUIM_OBJECTIVE.get_model()
    with torch.no_grad(): st,pe,si=m(torch.from_numpy(x.astype(np.float32)).unsqueeze(0))
    sq=f"STOI {st.item():.3f}   PESQ {pe.item():.2f}   SI-SDR {si.item():+.1f} dB"
except Exception as e: sq=f"unavailable ({type(e).__name__})"
try:
    import pyloudnorm as pyln; lu=f"{pyln.Meter(fs).integrated_loudness(x):+.1f} LUFS"
except Exception: lu="LUFS n/a"

fig,ax=plt.subplots(6,1,figsize=(16,17),sharex=True,
                    gridspec_kw=dict(height_ratios=[1.1,1.5,2.0,2.0,2.0,2.2],hspace=0.16))
fig.suptitle("PREPROCESS derivatives, aligned — 16 kHz, pre-emphasis a=0.97 (switchable)\n"
             f"level (plain): peak {peak:.1f} dBFS   RMS {rms:.1f} dBFS   {lu}      squim (plain): {sq}",
             fontsize=11.5,y=0.985)
tw=np.arange(len(x))/fs
ax[0].plot(tw,x,lw=.3,color="#888",label="plain")
ax[0].plot(tw,xp,lw=.3,color="#1f4e79",label="pre-emphasised")
ax[0].set_ylabel("waveform"); ax[0].legend(loc="upper right",fontsize=8,ncol=2); ax[0].set_ylim(-1,1)

ax[1].plot(t,Edb,lw=.8,color="#b03030")
ax[1].axhline(floor,ls=":",c="k",lw=.8); ax[1].axhline(floor+8,ls="--",c="k",lw=.8)
ax[1].text(0.02,floor+1,"floor (10th pct)",fontsize=7); ax[1].text(0.02,floor+9,"+8 dB propose gate",fontsize=7)
ax[1].set_ylabel("Hilbert envelope\n40 Hz LP, dB"); ax[1].set_ylim(floor-6,3)
for k,(s0,e0,pk) in enumerate(spans):
    for a_ in ax:
        a_.axvspan(s0,e0,color="#ffb000",alpha=.16,zorder=0)
    ax[1].annotate(f"S{k}\n{(e0-s0)*1000:.0f} ms\n{pk:.0f} dB",((s0+e0)/2,floor-3.2),ha="center",fontsize=7.5,color="#7a4a00")

for a_,(F,T,S,name,ylim) in zip(ax[2:4],[
        (d["f_wb"],d["t_wb"],d["Swb"],"wideband spectrogram\n5 ms win / 5 ms hop",8000),
        (d["f_nb"],d["t_nb"],d["Snb"],"narrowband spectrogram\n20 ms win / 5 ms hop",8000)]):
    a_.pcolormesh(T,F,np.clip(S,-80,0),shading="auto",cmap="magma",vmin=-80,vmax=0)
    a_.set_ylabel(name); a_.set_ylim(0,ylim)

cf=d["cf"]
ax[4].pcolormesh(d["t_g"],np.arange(len(cf)),np.clip(d["Gdb"],-80,0),shading="auto",cmap="viridis",vmin=-80,vmax=0)
ticks=[0,9,19,29,39]; ax[4].set_yticks(ticks); ax[4].set_yticklabels([f"{cf[i]:.0f}" for i in ticks],fontsize=8)
ax[4].set_ylabel("gammatone\n40 ERB ch (Hz)")

im=ax[5].pcolormesh(cen+0.25,np.arange(len(labels)),R,shading="auto",cmap="cividis",vmin=0,vmax=1)
ax[5].set_yticks(np.arange(len(labels))); ax[5].set_yticklabels(labels,fontsize=8.5)
ax[5].set_ylabel("HeAR event detector\n500 ms gate / 100 ms hop")
ax[5].set_xlabel("time (s)"); ax[5].set_xlim(0,dur)
cb=fig.colorbar(im,ax=ax[5],orientation="horizontal",pad=0.22,fraction=.05,aspect=60)
cb.set_label("HeAR presence probability (independent, not a distribution)",fontsize=8)
fig.text(0.5,0.008,"shaded = event spans from the envelope alone: peaks >= floor+18 dB, onset peak-anchored at peak-15 dB, "
  "offset at 0.7 of event range with a 120 ms hangover. No labels used.",
  ha="center",fontsize=9,color="#7a4a00")
fig.savefig("/Users/satra/Downloads/preprocess_derivatives_v2.png",dpi=115,bbox_inches="tight")
print("wrote /Users/satra/Downloads/preprocess_derivatives_v2.png")
print(f"spans: "+", ".join(f"{s:.2f}-{e:.2f}s" for s,e,_ in spans))
print("HeAR peak windows:", {l:f"{cen[np.nanargmax(R[i])]+0.25:.2f}s@{R[i].max():.2f}" for i,l in enumerate(labels) if R[i].max()>0.5})
