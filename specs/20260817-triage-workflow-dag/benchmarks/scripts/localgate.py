import numpy as np, soundfile as sf, warnings, sys
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from labels import label
from scipy.signal import resample_poly, hilbert, butter, filtfilt, find_peaks
fs=16000; bb,aa=butter(4,40/(fs/2),"low")
def envdbfs(x):
    """envelope in dBFS -- absolute, NOT normalised by the file maximum"""
    xp=np.empty_like(x); xp[0]=x[0]; xp[1:]=x[1:]-0.97*x[:-1]
    E=np.maximum(filtfilt(bb,aa,np.abs(hilbert(xp))),1e-12)
    return 20*np.log10(E)
def load(f,snr=None,click=False,seed=17):
    y,sr=sf.read(f,dtype="float32",always_2d=True)
    x=resample_poly(y.mean(axis=1),fs,sr).astype(np.float64)
    sp=label("speech")
    if snr is not None:
        sig=(x[int(sp["span_lo"]*fs):int(sp["span_hi"]*fs)]**2).mean()
        rng=np.random.default_rng(seed); w=rng.standard_normal(len(x)); W=np.fft.rfft(w)
        fq=np.fft.rfftfreq(len(x),1/fs); W[1:]/=np.sqrt(fq[1:]); nz=np.fft.irfft(W,len(x)); nz/=np.sqrt((nz**2).mean())
        x=x+nz*np.sqrt(sig/(10**(snr/10)))
    if click:
        i=int(1.0*fs); x=x.copy(); x[i:i+int(.03*fs)]+=0.95*np.sign(np.random.default_rng(3).standard_normal(int(.03*fs)))
    return x
def local_floor(Edb, win_s=3.0, q=10):
    """rolling low percentile -- a floor that tracks the recording, in dBFS"""
    w=int(win_s*fs); step=int(0.1*fs); n=len(Edb)
    cs=np.arange(0,n,step); vals=[]
    for c in cs:
        a,b=max(0,c-w//2),min(n,c+w//2); vals.append(np.percentile(Edb[a:b],q))
    return np.interp(np.arange(n), cs, vals)
def spans_local(Edb, K=18.0, drop=15.0, frac=.7, hang_ms=120, min_ms=50):
    fl=local_floor(Edb)
    pk,_=find_peaks(Edb-fl, height=K, distance=int(.150*fs)); out=[]
    for p in pk:
        th=Edb[p]-drop; i=p
        while i>0 and Edb[i]>th: i-=1
        tho=Edb[p]-frac*(Edb[p]-fl[p]); hang=int(hang_ms*fs/1000); j=p
        while j<len(Edb)-1:
            w2=Edb[j:j+hang]
            if len(w2) and w2.max()<=tho: break
            j+=1
        if j-i>=min_ms*fs/1000: out.append([i/fs,j/fs])
    out.sort(); mg=[]
    for s0,e0 in out:
        if mg and s0<=mg[-1][1]: mg[-1][1]=max(mg[-1][1],e0)
        else: mg.append([s0,e0])
    return mg
LAB="/Users/satra/Downloads/streaming-audio-2026-07-30T04-21-56-487Z.wav"
sp=label("speech"); a_,b_=sp["span_lo"],sp["span_hi"]
def report(tag,x,K=18.):
    Edb=envdbfs(x); mg=spans_local(Edb,K=K)
    hit=[s for s in mg if s[0]<b_ and s[1]>a_]
    if hit:
        s0,e0=max(hit,key=lambda s:min(s[1],b_)-max(s[0],a_))
        iou=max(0,min(e0,b_)-max(s0,a_))/(max(e0,b_)-min(s0,a_))
        print(f"  {tag:34s} {len(mg):3d} spans  speech span {s0:6.2f}-{e0:6.2f}  IoU {iou:.2f}")
    else:
        print(f"  {tag:34s} {len(mg):3d} spans  speech span {'NONE':>13s}   {'-':>7s}")
print("LOCAL floor (rolling 3 s 10th pct) in dBFS -- gate = local_floor + K\n")
for K in (18.,12.,8.):
    print(f"  K = {K:.0f} dB")
    for snr in (None,20,10,5,0,-5):
        report(f"   SNR {'orig' if snr is None else f'{snr:+d} dB'}", load(LAB,snr=snr), K=K)
    report("   orig + injected click", load(LAB,click=True), K=K)
    print()
import sys; sys.exit(0)
print("\nfor comparison, the committed global peak-anchored rule under the same click:")
def spans_global(x,K=25.):
    xp=np.empty_like(x); xp[0]=x[0]; xp[1:]=x[1:]-0.97*x[:-1]
    E=np.maximum(filtfilt(bb,aa,np.abs(hilbert(xp))),1e-12); Edb=20*np.log10(E/E.max())
    fl=np.percentile(Edb,10)
    pk,_=find_peaks(Edb,height=-K,distance=int(.150*fs)); out=[]
    for p in pk:
        th=Edb[p]-15.; i=p
        while i>0 and Edb[i]>th: i-=1
        tho=Edb[p]-.7*(Edb[p]-fl); hang=int(.120*fs); j=p
        while j<len(Edb)-1:
            w2=Edb[j:j+hang]
            if len(w2) and w2.max()<=tho: break
            j+=1
        if j-i>=.05*fs: out.append([i/fs,j/fs])
    out.sort(); mg=[]
    for s0,e0 in out:
        if mg and s0<=mg[-1][1]: mg[-1][1]=max(mg[-1][1],e0)
        else: mg.append([s0,e0])
    return mg
for tag,xx in (("orig",load(LAB)),("orig + click",load(LAB,click=True))):
    mg=spans_global(xx); hit=[s for s in mg if s[0]<b_ and s[1]>a_]
    print(f"  peak-25dB {tag:24s} {len(mg):3d} spans  speech {'FOUND' if hit else 'LOST'}")
