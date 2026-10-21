# ruff: noqa  -- benchmark harness run inside the subprocess venvs, not part of the package.
"""Benchmark Qwen3-ASR (+ forced aligner) on one file. Runs inside the qwen-asr venv."""

import argparse, json, os, resource, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("--model", required=True)
ap.add_argument("--aligner", default=None)
ap.add_argument("--device", default="cpu")
ap.add_argument("--dtype", default="float32")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--max-new-tokens", type=int, default=512)
ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "0")))
ap.add_argument("--out", default=None)
ap.add_argument("--chunk-sec", type=float, default=0)
ap.add_argument("--device-map", action="store_true")
ap.add_argument("--empty-cache", action="store_true")
ap.add_argument("--pad-chunks", action="store_true")
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fpsampler import Sampler

SAMPLER = Sampler()
import torch

if a.threads:
    torch.set_num_threads(a.threads)
from qwen_asr import Qwen3ASRModel

if a.chunk_sec:
    import qwen_asr.inference.qwen3_asr as _q

    _q.MAX_FORCE_ALIGN_INPUT_SECONDS = a.chunk_sec
    _q.MAX_ASR_INPUT_SECONDS = a.chunk_sec
    if a.pad_chunks:
        import numpy as _np

        _split = _q.split_audio_into_chunks
        _target = int((a.chunk_sec + 5.0) * 16000)

        def _padded(wav, sr, max_chunk_sec, **kw):
            out = []
            for w, off in _split(wav, sr, max_chunk_sec, **kw):
                if len(w) < _target:
                    w = _np.concatenate([w, _np.zeros(_target - len(w), dtype=w.dtype)])
                out.append((w, off))
            return out

        _q.split_audio_into_chunks = _padded
dt = getattr(torch, a.dtype)
t0 = time.perf_counter()
kw = {"dtype": dt}
if a.device_map:
    kw["device_map"] = a.device
if a.aligner:
    kw["forced_aligner"] = a.aligner
    kw["forced_aligner_kwargs"] = {"dtype": dt, **({"device_map": a.device} if a.device_map else {})}
asr = Qwen3ASRModel.from_pretrained(a.model, max_inference_batch_size=a.batch, max_new_tokens=a.max_new_tokens, **kw)
asr.model = asr.model.to(a.device)
if asr.forced_aligner is not None:
    asr.forced_aligner.model = asr.forced_aligner.model.to(a.device)
    asr.forced_aligner.device = asr.forced_aligner.model.device
t_load = time.perf_counter() - t0
if a.empty_cache and a.device == "mps":
    _gen = asr.model.generate

    def _generate(*args, **kwargs):
        try:
            return _gen(*args, **kwargs)
        finally:
            torch.mps.empty_cache()

    asr.model.generate = _generate
    if asr.forced_aligner is not None:
        _align = asr.forced_aligner.align

        def _aligned(*args, **kwargs):
            try:
                return _align(*args, **kwargs)
            finally:
                torch.mps.empty_cache()

        asr.forced_aligner.align = _aligned
mem_after_load = torch.mps.driver_allocated_memory() / 1e9 if a.device == "mps" else None
import soundfile as sf

dur = sf.info(a.audio).duration
SAMPLER.mark("loaded")
t0 = time.perf_counter()
r = asr.transcribe(audio=[a.audio], language="English", return_time_stamps=bool(a.aligner))[0]
if a.device == "mps":
    torch.mps.synchronize()
t_run = time.perf_counter() - t0
words = [[s.text, s.start_time, s.end_time] for s in (r.time_stamps or [])] if a.aligner else []
out = {
    "device": a.device,
    "dtype": a.dtype,
    "batch": a.batch,
    "chunk_sec": a.chunk_sec,
    "aligner": bool(a.aligner),
    "threads": torch.get_num_threads(),
    "audio_s": dur,
    "load_s": t_load,
    "run_s": t_run,
    "rtf": t_run / dur,
    "maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
    "n_words": len(r.text.split()),
    "text": r.text,
    "words": words,
    "loadavg": os.getloadavg(),
    "device_map": a.device_map,
    "empty_cache": a.empty_cache,
    "pad_chunks": a.pad_chunks,
    "mps_driver_gb_after_load": mem_after_load,
    "mps_driver_gb_end": (torch.mps.driver_allocated_memory() / 1e9 if a.device == "mps" else None),
}
series = SAMPLER.stop()
out["footprint_series_gb"] = series
out["footprint_peak_dump"] = getattr(SAMPLER, "best_dump", None)
vals = [v for _, v in series if v is not None]
out["footprint_peak_gb"] = max(vals) if vals else None
i = next((k for k, (lab, v) in enumerate(series) if lab == "loaded"), 0)
run_vals = [v for _, v in series[i:] if v is not None]
out["footprint_run_peak_gb"] = max(run_vals) if run_vals else None
out["footprint_run_last_gb"] = run_vals[-1] if run_vals else None
if a.out:
    json.dump(out, open(a.out, "w"))
print(
    json.dumps(
        {k: v for k, v in out.items() if k not in ("text", "words", "footprint_series_gb", "footprint_peak_dump")}
    )
)
print(r.text[:400], file=sys.stderr)
