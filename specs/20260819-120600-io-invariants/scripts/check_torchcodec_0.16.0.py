import inspect, os, subprocess, warnings
import numpy as np, soundfile as sf, torch, torchcodec
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder
warnings.filterwarnings("error", category=UserWarning)   # make any warning loud
W="/tmp/tcnew/w"; os.makedirs(W, exist_ok=True); SR=22050
print(f"torchcodec {torchcodec.__version__}  torch {torch.__version__}  libsndfile {sf.__libsndfile_version__}")

print("\n== 1. encode API signature: is there any encoding / bit-depth control now?")
print("  AudioEncoder.__init__", inspect.signature(AudioEncoder.__init__))
print("  to_file             ", inspect.signature(AudioEncoder.to_file))
print("  AudioDecoder.__init__", inspect.signature(AudioDecoder.__init__))

print("\n== 2. does encode still silently clamp, and still choose PCM_16 for .wav?")
flat = torch.full((1,1000), 1.5)
p=os.path.join(W,"flat.wav")
try:
    AudioEncoder(samples=flat, sample_rate=SR).to_file(p); wmsg="no warning"
except UserWarning as e: wmsg=f"WARNED: {e}"
back = AudioDecoder(p).get_all_samples().data
print(f"  input flat 1.5 -> subtype={sf.info(p).subtype} peak={back.abs().max().item()} distinct={back.unique().numel()}  [{wmsg}]")

print("\n== 3. extension -> codec map (spot check)")
x=(torch.rand(1,2000)*0.5)
for ext in ["wav","flac","ogg","wv","mp3"]:
    q=os.path.join(W,f"e.{ext}")
    if os.path.exists(q): os.remove(q)
    try:
        AudioEncoder(samples=x, sample_rate=SR).to_file(q)
        pr=subprocess.run(["ffprobe","-v","error","-select_streams","a:0","-show_entries","stream=codec_name,sample_fmt","-of","default=nw=1:nk=1",q],capture_output=True,text=True).stdout.split()
        try: si=f"{sf.info(q).format}/{sf.info(q).subtype}"
        except Exception: si="libsndfile CANNOT READ"
        print(f"  .{ext:<5} {'/'.join(pr):<22} {si}")
    except Exception as e: print(f"  .{ext:<5} {type(e).__name__}: {str(e)[:70]}")

print("\n== 4. float round-trip exactness through torchcodec .wav / .wv")
z=(np.random.default_rng(3).uniform(-0.9,0.9,2000)).astype(np.float32)
for ext in ["wav","wv"]:
    q=os.path.join(W,f"r.{ext}")
    AudioEncoder(samples=torch.from_numpy(z.reshape(1,-1)), sample_rate=SR).to_file(q)
    a=AudioDecoder(q).get_all_samples().data.numpy()[0]
    print(f"  .{ext:<4} exact={a.tobytes()==z.tobytes()} maxdiff={np.abs(a-z).max():.3e}")

print("\n== 5. CHUNK == SLICE (each vs its own full decode, alignment searched +/-3000)")
def align(full,ch,nom,win=3000):
    n=ch.shape[-1]; best=(None,np.inf,False)
    for d in range(-win,win+1):
        s=nom+d
        if s<0 or s+n>full.shape[-1]: continue
        seg=full[:,s:s+n]
        if seg.shape!=ch.shape: continue
        v=float(np.max(np.abs(seg.astype(np.float64)-ch.astype(np.float64))))
        if v<best[1]: best=(d,v,bool(seg.tobytes()==ch.astype(np.float32).tobytes()))
    return best
sig=(np.random.default_rng(23).integers(-32768,32768,SR*6)/32768.0).astype(np.float32)
def ffw(p,x,sr,c):
    subprocess.run(["ffmpeg","-hide_banner","-v","error","-y","-f","f32le","-ar",str(sr),"-ac","1","-i","-"]+c+[p],
                   input=np.ascontiguousarray(x.astype("<f4")).tobytes(),check=True)
F={}
sf.write(os.path.join(W,"c16.wav"),sig,SR,format="WAV",subtype="PCM_16"); F["wav PCM_16"]=("c16.wav",SR)
sf.write(os.path.join(W,"cf.wav"),sig,SR,format="WAV",subtype="FLOAT");  F["wav FLOAT32"]=("cf.wav",SR)
sf.write(os.path.join(W,"c24.flac"),sig,SR,format="FLAC",subtype="PCM_24"); F["flac PCM_24"]=("c24.flac",SR)
ffw(os.path.join(W,"c.mp3"),sig,SR,["-c:a","libmp3lame","-b:a","192k"]); F["mp3 192k"]=("c.mp3",SR)
ffw(os.path.join(W,"c.m4a"),sig,SR,["-c:a","aac","-b:a","128k"]);        F["m4a aac"]=("c.m4a",SR)
ffw(os.path.join(W,"c.opus"),sig,48000,["-c:a","libopus"]);              F["opus 48k"]=("c.opus",48000)
for name,(f,fsr) in F.items():
    p=os.path.join(W,f); full=AudioDecoder(p).get_all_samples().data.numpy()
    cells=[]
    for tag,a0,n0 in [("t=0",0,2048),("odd",5001,2048),("mid",44877,4096)]:
        a=int(round(a0*fsr/SR)); n=int(round(n0*fsr/SR))
        s=AudioDecoder(p).get_samples_played_in_range(start_seconds=a/fsr, stop_seconds=(a+n)/fsr)
        ch=s.data.numpy(); d,m,ex=align(full,ch,a)
        cells.append(f"{tag}: n={ch.shape[1]} off={d} d={m:.1e} {'EXACT' if ex else 'NO'}")
    print(f"  {name:<13} full_n={full.shape[1]:<7} " + " | ".join(cells))

print("\n== 6. contiguity: concat 2048-sample chunks vs full decode")
for name,(f,fsr) in F.items():
    p=os.path.join(W,f); full=AudioDecoder(p).get_all_samples().data.numpy()
    parts=[]; a=0
    while a<full.shape[1]:
        n=min(2048, full.shape[1]-a)
        parts.append(AudioDecoder(p).get_samples_played_in_range(start_seconds=a/fsr, stop_seconds=(a+n)/fsr).data.numpy()); a+=2048
    cat=np.concatenate(parts,axis=1); k=min(cat.shape[1],full.shape[1])
    print(f"  {name:<13} concat_n={cat.shape[1]:<7} full_n={full.shape[1]:<7} "
          f"bitexact={cat.shape==full.shape and cat.tobytes()==full.tobytes()} lost={full.shape[1]-cat.shape[1]}")

print("\n== 7. #1601/#1614 regression: chunked decode WITH resampling vs one go")
p=os.path.join(W,"cf.wav")
for target in [16000, 8000]:
    full=AudioDecoder(p, sample_rate=target).get_all_samples().data.numpy()
    parts=[]; t=0.0; dur=1.2
    while t < 6.0:
        s=AudioDecoder(p, sample_rate=target).get_samples_played_in_range(start_seconds=t, stop_seconds=min(t+dur,6.0))
        parts.append(s.data.numpy()); t+=dur
    cat=np.concatenate(parts,axis=1); k=min(cat.shape[1],full.shape[1])
    print(f"  resample->{target}: full_n={full.shape[1]} chunked_n={cat.shape[1]} "
          f"bitexact={cat.shape==full.shape and cat.tobytes()==full.tobytes()} maxdiff={np.abs(cat[:,:k]-full[:,:k]).max():.3e}")

print("\n== 8. decode range-transparency (float WAV, peak 4.0) and amplitude convention")
oor=np.concatenate([np.full(200,3.0),np.full(200,-4.0),np.linspace(-4,4,400)]).astype(np.float32)
q=os.path.join(W,"oor.wav"); sf.write(q,oor,SR,format="WAV",subtype="FLOAT")
a=AudioDecoder(q).get_all_samples().data.numpy()[0]
print(f"  float WAV peak 4.0 -> read peak={np.abs(a).max():.5f} bitexact={a.tobytes()==oor.tobytes()}")
ints=np.array([-32768,-32767,0,32767],dtype=np.int16); q2=os.path.join(W,"fs.wav")
sf.write(q2,ints,SR,format="WAV",subtype="PCM_16")
print("  int16 full-scale ->", AudioDecoder(q2).get_all_samples().data.numpy()[0].tolist(), "( /32768 convention expected )")
