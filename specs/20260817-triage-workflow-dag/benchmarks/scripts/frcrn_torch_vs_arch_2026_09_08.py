#!/usr/bin/env python3
r"""One cell of the FRCRN torch-version x CPU-architecture 2x2.

Disambiguates which one causes FRCRN_SE_16K to null breath recordings locally (torch 2.13, arm64)
while passing them through on the cluster (torch 2.14+cpu, x86_64).

Builds one throwaway ``ensure_venv`` environment pinned to a specific torch version (``clearvoice``
itself and ``numpy`` held fixed across every cell so torch is the only thing that varies within one
architecture), runs FRCRN_SE_16K's own worker script (imported from
``senselab.utils.clearvoice._WORKER_SCRIPT``, not reimplemented) over a fixed set of input files,
and calls ``senselab.audio.tasks.speech_enhancement.residual.compute_residual`` -- the shared library,
not a reimplementation -- to report the fraction of input energy the enhanced output and the residual
each retain, the fitted gain, and the correlation of enhanced with input.

Run once per (architecture, torch version) cell -- twice per cell for the determinism check the
benchmark requires. Local (arm64) cells run directly; the x86_64 cells run this same script on the
cluster, over the same three input files copied there byte-identical.

Usage:
    uv run python frcrn_torch_vs_arch_2026_09_08.py \\
        --venv-name clearvoice-exp214-arm64 \\
        --torch-spec "torch==2.14.*" \\
        --inputs a.wav b.wav c.wav \\
        --repeats 2 \\
        --out-dir ~/scratch/frcrn_cell_out \\
        --out-json ~/scratch/frcrn_cell_out/result.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import soundfile as sf

if TYPE_CHECKING:
    from senselab.utils.clearvoice import ClearVoiceModelSpec


def env_probe(python: str, clean_env: dict) -> dict[str, Any]:
    """Report the experiment venv's actual torch/torchaudio versions and host architecture."""
    probe = (
        "import json, platform, torch, torchaudio;"
        "print(json.dumps({'torch': torch.__version__, 'torchaudio': torchaudio.__version__,"
        "'machine': platform.machine(), 'python': platform.python_version(),"
        "'cuda_available': torch.cuda.is_available()}))"
    )
    r = subprocess.run([python, "-c", probe], capture_output=True, text=True, env=clean_env)
    if r.returncode != 0:
        raise RuntimeError(f"env probe failed: {r.stderr}")
    return json.loads(r.stdout)


def run_batch(
    python: str,
    worker_script: str,
    checkpoint_dir: Path,
    spec: "ClearVoiceModelSpec",
    expected_version: str,
    in_paths: list[Path],
    out_dir: Path,
    staging_dir: str,
    io_dir: str,
    clean_env: dict,
) -> list[str]:
    """Run one FRCRN worker call over every input in ``in_paths``, batched in one subprocess."""
    from senselab.utils.subprocess_venv import parse_subprocess_result

    payload = {
        "mode": "audio",
        "staging_dir": staging_dir,
        "io_dir": io_dir,
        "model_name": spec.name,
        "task": spec.upstream_task,
        "checkpoint_dir": str(checkpoint_dir),
        "expected_version": expected_version,
        "device": None,
        "rms_normalise": spec.rms_normalises_input,
        "in_paths": [str(p) for p in in_paths],
        "out_dir": str(out_dir),
    }
    result = subprocess.run(
        [python, "-c", worker_script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=1800,
        env=clean_env,
    )
    output = parse_subprocess_result(result, venv_label="FRCRN torch/arch experiment")
    # One list per input; FRCRN_SE_16K is single-output, so take entry 0 of each.
    return [paths[0] for paths in output["output_paths"]]


def measure(reference_path: Path, signal_path: Path, max_lag_ms: float) -> dict[str, float]:
    """Compute the four reported quantities via the shared residual library."""
    from senselab.audio.tasks.speech_enhancement.residual import compute_residual

    ref, sr_ref = sf.read(str(reference_path), dtype="float64", always_2d=True)
    ref = ref.mean(axis=1)
    sig, sr_sig = sf.read(str(signal_path), dtype="float64", always_2d=True)
    sig = sig.mean(axis=1)
    if sr_sig != sr_ref:
        raise RuntimeError(f"{reference_path.name}: sampling rate mismatch {sr_ref} vs {sr_sig}")
    comp = compute_residual(ref, sig, sr_ref, max_lag_ms=max_lag_ms)
    return {
        "signal_energy_fraction": comp.signal_energy_fraction,
        "residual_energy_fraction": comp.residual_energy_fraction,
        "gain_db": comp.gain_db,
        "correlation_signal": comp.correlation_signal,
        "lag_ms": comp.lag_ms,
    }


def main() -> None:
    """Parse CLI args, run one cell's worker calls, and write the JSON result."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--venv-name", required=True, help="Unique ensure_venv name for this cell.")
    ap.add_argument("--torch-spec", required=True, help='e.g. "torch==2.13.*"')
    ap.add_argument("--torchaudio-spec", default="torchaudio==2.11.0", help="Pinned torchaudio spec.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input WAV paths, identical across cells.")
    ap.add_argument("--repeats", type=int, default=2, help="Repeat count for the determinism check.")
    ap.add_argument("--out-dir", required=True, help="Directory for enhanced WAV outputs.")
    ap.add_argument("--out-json", required=True, help="Path to write the JSON result.")
    ap.add_argument("--max-lag-ms", type=float, default=200.0, help="compute_residual's search half-window.")
    args = ap.parse_args()

    from senselab.utils.clearvoice import (
        _WORKER_SCRIPT,
        CLEARVOICE_VERSION,
        clearvoice_model_spec,
        stage_clearvoice_checkpoints,
    )
    from senselab.utils.subprocess_venv import (
        _clean_subprocess_env,
        ensure_venv,
        stage_portable_audio_io,
        venv_python,
    )

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    in_paths = [Path(p).expanduser() for p in args.inputs]
    for p in in_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    requirements = ["clearvoice==0.1.2", args.torch_spec, args.torchaudio_spec, "numpy<2.0,>=1.24.3"]
    print(f"Building/reusing venv {args.venv_name!r} with {requirements}", file=sys.stderr)
    venv_dir = ensure_venv(args.venv_name, requirements, python_version="3.11")
    python = venv_python(venv_dir)
    clean_env = _clean_subprocess_env()
    env_info = env_probe(python, clean_env)
    print(f"Venv env: {env_info}", file=sys.stderr)

    spec = clearvoice_model_spec("FRCRN_SE_16K", expected_task="speech_enhancement")
    checkpoint_dir, sha = stage_clearvoice_checkpoints(spec)

    results: dict[str, Any] = {
        "venv_name": args.venv_name,
        "requested_torch_spec": args.torch_spec,
        "requested_torchaudio_spec": args.torchaudio_spec,
        "env": env_info,
        "checkpoint_commit": sha,
        "inputs": [str(p) for p in in_paths],
        "repeats": args.repeats,
        "runs": [],
    }

    with tempfile.TemporaryDirectory(prefix="frcrn-exp-") as staging:
        io_dir = stage_portable_audio_io(staging)
        for repeat in range(args.repeats):
            repeat_dir = out_dir / f"rep{repeat}"
            repeat_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{args.venv_name}] repeat {repeat}: running {len(in_paths)} file(s)...", file=sys.stderr)
            enhanced_paths = run_batch(
                python,
                _WORKER_SCRIPT,
                checkpoint_dir,
                spec,
                CLEARVOICE_VERSION,
                in_paths,
                repeat_dir,
                staging,
                io_dir,
                clean_env,
            )
            for in_path, enh_raw_path in zip(in_paths, enhanced_paths, strict=True):
                key = in_path.stem
                enh_path = repeat_dir / f"{key}__{args.venv_name}__rep{repeat}__enhanced.wav"
                shutil.copyfile(enh_raw_path, enh_path)
                metrics = measure(in_path, enh_path, args.max_lag_ms)
                results["runs"].append(
                    {
                        "repeat": repeat,
                        "file": key,
                        "input_path": str(in_path),
                        "enhanced_path": str(enh_path),
                        **metrics,
                    }
                )
                print(
                    f"  {key}: signal_frac={metrics['signal_energy_fraction']:.4f} "
                    f"residual_frac={metrics['residual_energy_fraction']:.4f} "
                    f"gain_db={metrics['gain_db']:.2f} corr={metrics['correlation_signal']:.3f} "
                    f"lag_ms={metrics['lag_ms']:.2f}",
                    file=sys.stderr,
                )

    out_json = Path(args.out_json).expanduser()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
