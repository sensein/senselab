# ruff: noqa  -- benchmark harness run inside the subprocess venvs, not part of the package.
"""Rebuild an AMI IHM 'headset mix' timeline from utterance clips, plus utterance references."""

import io, json, sys
import numpy as np, pyarrow.parquet as pq, soundfile as sf

src, outdir = sys.argv[1], sys.argv[2]
meetings = sys.argv[3].split(",")
top_speakers = int(sys.argv[4]) if len(sys.argv) > 4 else 0
t = pq.read_table(src, filters=[("meeting_id", "in", meetings)])
df = t.to_pandas()
for m in meetings:
    d = df[df.meeting_id == m].sort_values("begin_time")
    suffix = "ihmmix"
    if top_speakers:
        talk = (d.end_time - d.begin_time).groupby(d.speaker_id).sum().sort_values(ascending=False)
        d = d[d.speaker_id.isin(talk.index[:top_speakers])]
        suffix = f"top{top_speakers}mix"
    end = float(d.end_time.max()) + 1.0
    mix = np.zeros(int(end * 16000) + 16000, dtype=np.float32)
    refs = []
    for _, r in d.iterrows():
        a = r["audio"]
        wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        assert sr == 16000
        if wav.ndim > 1:
            wav = wav.mean(1)
        s = int(round(float(r.begin_time) * 16000))
        mix[s : s + len(wav)] += wav[: len(mix) - s]
        refs.append({"start": float(r.begin_time), "end": float(r.end_time), "speaker": r.speaker_id, "text": r.text})
    mix /= max(1e-6, np.abs(mix).max()) / 0.9
    sf.write(f"{outdir}/{m}_{suffix}.wav", mix, 16000, subtype="PCM_16")
    json.dump(refs, open(f"{outdir}/{m}_{suffix}_refs.json", "w"))
    print(m, len(mix) / 16000, len(refs))
