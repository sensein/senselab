import numpy as np, soundfile as sf, warnings, json, sys
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import resample_poly, hilbert, butter, filtfilt, stft, gammatone, lfilter

SRC=sys.argv[1]; OUT=sys.argv[2]
fs=16000; A=0.97
y,sr=sf.read(SRC,dtype="float32",always_2d=True)
x=resample_poly(y.mean(axis=1), fs, sr).astype(np.float64)
xp=np.empty_like(x); xp[0]=x[0]; xp[1:]=x[1:]-A*x[:-1]
dur=len(x)/fs
print(f"{sr}->{fs} Hz, {dur:.2f} s")

# ---- envelope (PREPROCESS energy_envelope)
b,a=butter(4,40/(fs/2),"low")
E=np.maximum(filtfilt(b,a,np.abs(hilbert(xp))),1e-12); E/=E.max()
Edb=20*np.log10(E); floor=np.percentile(Edb,10)
t=np.arange(len(E))/fs

# ---- spans from the envelope alone: propose on a coarse gate, then per-event rules
def envelope_spans(Edb,floor,fs,propose_db=8.0,onset_drop=15.0,off_frac=0.7,hang_ms=120,min_ms=60):
    hot=Edb>floor+propose_db
    d=np.diff(hot.astype(int)); starts=np.where(d==1)[0]+1; ends=np.where(d==-1)[0]+1
    if hot[0]: starts=np.r_[0,starts]
    if hot[-1]: ends=np.r_[ends,len(hot)-1]
    hang=int(hang_ms*fs/1000); spans=[]
    for s0,e0 in zip(starts,ends):
        if e0-s0 < min_ms*fs/1000: continue
        pk=s0+int(np.argmax(Edb[s0:e0]))
        i=pk; th_on=Edb[pk]-onset_drop
        while i>0 and Edb[i]>th_on: i-=1
        th_off=Edb[pk]-off_frac*(Edb[pk]-floor); j=pk
        while j<len(Edb)-1:
            w=Edb[j:j+hang]
            if len(w) and w.max()<=th_off: break
            j+=1
        spans.append((i/fs,j/fs,Edb[pk]-floor))
    merged=[]
    for s0,e0,pkdb in spans:
        if merged and s0<=merged[-1][1]: merged[-1]=(merged[-1][0],max(merged[-1][1],e0),max(merged[-1][2],pkdb))
        else: merged.append((s0,e0,pkdb))
    return merged
spans=envelope_spans(Edb,floor,fs)
print(f"envelope proposed {len(spans)} spans")

# ---- spectrograms
f_wb,t_wb,Zwb=stft(xp,fs,nperseg=int(0.005*fs),noverlap=int(0.005*fs)-int(0.005*fs),boundary=None)
f_wb,t_wb,Zwb=stft(xp,fs,nperseg=80,noverlap=80-80,boundary=None)     # 5ms win, 5ms hop
f_nb,t_nb,Znb=stft(xp,fs,nperseg=320,noverlap=320-80,boundary=None)   # 20ms win, 5ms hop
def db(Z,ref=None):
    M=np.abs(Z); m=20*np.log10(M+1e-10); return m-m.max()

# ---- gammatone filterbank
def erb_space(lo,hi,n):
    E=lambda f: 21.4*np.log10(4.37e-3*f+1)
    Ei=lambda e:(10**(e/21.4)-1)/4.37e-3
    return Ei(np.linspace(E(lo),E(hi),n))
cf=erb_space(80,7800,40)
hop_g=int(0.005*fs)
G=np.zeros((len(cf),len(xp)//hop_g))
for k,fc in enumerate(cf):
    bb,aa=gammatone(fc,'iir',fs=fs)
    v=np.abs(hilbert(lfilter(bb,aa,xp)))
    n=len(v)//hop_g
    G[k]=v[:n*hop_g].reshape(n,hop_g).mean(axis=1)
Gdb=20*np.log10(G+1e-10); Gdb-=Gdb.max()
t_g=np.arange(G.shape[1])*hop_g/fs
print("gammatone done")
np.savez(OUT+".npz",t=t,Edb=Edb,floor=floor,spans=np.array(spans),
         f_wb=f_wb,t_wb=t_wb,Swb=db(Zwb),f_nb=f_nb,t_nb=t_nb,Snb=db(Znb),
         cf=cf,t_g=t_g,Gdb=Gdb,dur=dur,x=x,xp=xp,fs=fs)
print("saved stage 1 ->",OUT+".npz")
