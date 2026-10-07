#!/usr/bin/env python3
r"""Run the three probe files through senselab's actual production ClearerVoice venv.

Uses ``CLEARVOICE_VENV``/``CLEARVOICE_REQUIREMENTS`` (not a pinned throwaway experiment venv) and
reports the same four quantities, for comparison against the pinned-torch experiment cells.

Usage:
    uv run python frcrn_production_venv_check_2026_09_08.py --inputs a.wav b.wav c.wav \\
        --out-dir OUT --out-json OUT/result.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import soundfile as sf


def main() -> None:
    """Parse CLI args, run FRCRN via the production venv, and write the JSON result."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--max-lag-ms", type=float, default=200.0)
    args = ap.parse_args()

    from senselab.audio.data_structures import Audio
    from senselab.audio.tasks.speech_enhancement.api import enhance_audios
    from senselab.audio.tasks.speech_enhancement.residual import compute_residual
    from senselab.utils.clearvoice import CLEARVOICE_REQUIREMENTS, CLEARVOICE_VENV, ensure_venv
    from senselab.utils.data_structures import HFModel
    from senselab.utils.subprocess_venv import _clean_subprocess_env, venv_python

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    in_paths = [Path(p).expanduser() for p in args.inputs]

    venv_dir = ensure_venv(CLEARVOICE_VENV, CLEARVOICE_REQUIREMENTS, python_version="3.11")
    python = venv_python(venv_dir)
    probe = (
        "import json, platform, torch, torchaudio, numpy, scipy, soundfile;"
        "print(json.dumps({'torch': torch.__version__, 'torchaudio': torchaudio.__version__,"
        "'numpy': numpy.__version__, 'scipy': scipy.__version__, 'soundfile': soundfile.__version__,"
        "'machine': platform.machine(), 'cuda_available': torch.cuda.is_available()}))"
    )
    r = subprocess.run([python, "-c", probe], capture_output=True, text=True, env=_clean_subprocess_env())
    env_info = json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr}
    print("production venv env:", env_info)

    audios = [Audio(filepath=str(p)) for p in in_paths]
    model = HFModel(path_or_uri="alibabasglab/FRCRN_SE_16K")
    enhanced = enhance_audios(audios, model=model)

    results: dict[str, Any] = {"env": env_info, "venv_dir": str(venv_dir), "runs": []}
    for in_path, enh in zip(in_paths, enhanced, strict=True):
        key = in_path.stem
        enh_path = out_dir / f"{key}__production_venv__enhanced.wav"
        enh.save_to_file(str(enh_path), subtype="PCM_16", out_of_range="warn")

        ref, sr_ref = sf.read(str(in_path), dtype="float64", always_2d=True)
        ref = ref.mean(axis=1)
        sig, sr_sig = sf.read(str(enh_path), dtype="float64", always_2d=True)
        sig = sig.mean(axis=1)
        comp = compute_residual(ref, sig, sr_ref, max_lag_ms=args.max_lag_ms)
        metrics = {
            "signal_energy_fraction": comp.signal_energy_fraction,
            "residual_energy_fraction": comp.residual_energy_fraction,
            "gain_db": comp.gain_db,
            "correlation_signal": comp.correlation_signal,
            "lag_ms": comp.lag_ms,
        }
        results["runs"].append({"file": key, **metrics})
        print(
            f"{key}: signal_frac={metrics['signal_energy_fraction']:.4f} "
            f"residual_frac={metrics['residual_energy_fraction']:.4f} "
            f"gain_db={metrics['gain_db']:.2f} corr={metrics['correlation_signal']:.3f}"
        )

    Path(args.out_json).expanduser().write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
