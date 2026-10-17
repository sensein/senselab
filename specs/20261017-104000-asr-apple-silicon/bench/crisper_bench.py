# ruff: noqa  -- benchmark harness run inside the subprocess venvs, not part of the package.
"""Benchmark CrisperWhisper 2.0 (transformers backend) on one file. Runs inside the crisperwhisper venv."""

import argparse
import json
import os
import resource
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("--model", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--compute-type", default="float32")
ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "0")))
ap.add_argument("--out", default=None)
ap.add_argument("--empty-cache", action="store_true")
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fpsampler import Sampler

SAMPLER = Sampler()

import torch  # noqa: E402

if a.threads:
    torch.set_num_threads(a.threads)
from crisperwhisper import CrisperWhisperModel  # noqa: E402

if a.device == "mps":
    from transformers.generation import utils as _gu

    if hasattr(_gu, "DeferredStopCheck"):
        _gu.DeferredStopCheck.is_supported = staticmethod(lambda *args, **kw: False)

t0 = time.perf_counter()
m = CrisperWhisperModel(a.model, backend="transformers", device=a.device, compute_type=a.compute_type)
t_load = time.perf_counter() - t0
if a.empty_cache and a.device == "mps":
    _eng = m._engine
    _fx = _eng.extract_features_with_mel

    def _efx(*args, **kw):
        torch.mps.empty_cache()
        return _fx(*args, **kw)

    _eng.extract_features_with_mel = _efx
import soundfile as sf  # noqa: E402

dur = sf.info(a.audio).duration
SAMPLER.mark("loaded")
t0 = time.perf_counter()
r = m.transcribe(a.audio, language="en", word_timestamps=True, longform_strategy="continuation", max_new_tokens=256)
if a.device == "mps":
    torch.mps.synchronize()
t_run = time.perf_counter() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
out = {
    "device": a.device,
    "compute_type": a.compute_type,
    "threads": torch.get_num_threads(),
    "audio_s": dur,
    "load_s": t_load,
    "run_s": t_run,
    "rtf": t_run / dur,
    "maxrss_gb": rss,
    "n_words": len(r.words or []),
    "text": r.text,
    "words": [[w.word, w.start, w.end] for w in (r.words or [])],
    "loadavg": os.getloadavg(),
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
print(r.text[:600], file=sys.stderr)
