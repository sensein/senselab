"""Parse the two ORCD scheduling benchmark logs and recompute the fixed/marginal fits.

Inputs are the raw sbatch stdout files, one per job:
  - gpu-22213820.out (node3805, A100 80GB)
  - cpu-22213836.out (node2803, no GPU)

Each file interleaves plain log lines with ``RESULT {json}`` lines. This script extracts
every RESULT line, reconstructs the batch-size sweep for yamnet and hear on each host, fits
seconds = fixed + marginal * batch_size by ordinary least squares over the raw (not
per-batch-averaged) points, and reports fixed-cost fractions at batch 1 and batch 64.

Usage:
    uv run python orcd_scheduling_2026_09_08.py <gpu.out> <cpu.out>
"""

import json
import sys
from pathlib import Path


def parse_results(path: Path) -> list[dict]:
    """Extract every RESULT {...} JSON record from a raw job log."""
    records = []
    for line in path.read_text().splitlines():
        marker = "RESULT "
        idx = line.find(marker)
        if idx == -1:
            continue
        payload = line[idx + len(marker) :]
        records.append(json.loads(payload))
    return records


def ols_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Ordinary least squares fit of y = a + b*x. Returns (a, b) = (fixed, marginal)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var = sum((x - mean_x) ** 2 for x in xs)
    b = cov / var
    a = mean_y - b * mean_x
    return a, b


def full_calls_by_backend(records: list[dict], backend: str) -> dict[int, list[float]]:
    """batch_size -> list of full_call seconds for one backend."""
    out: dict[int, list[float]] = {}
    for r in records:
        if r.get("backend") == backend and r.get("kind") == "full_call" and r.get("error") is None:
            out.setdefault(r["batch_size"], []).append(r["seconds"])
    return out


def report(label: str, records: list[dict]) -> None:
    """Print full_call fits, warm ensure_venv costs, and any suspect/failed records for one job."""
    print(f"\n=== {label} ===")
    meta = next(r for r in records if r.get("kind") == "meta")
    print(f"host={meta['host']} cuda_available={meta['cuda_available']} batch_sizes={meta['batch_sizes']}")

    for backend in ("yamnet", "hear"):
        calls = full_calls_by_backend(records, backend)
        if not calls:
            continue
        print(f"\n{backend} full_call seconds by batch_size:")
        xs, ys = [], []
        for bs in sorted(calls):
            vals = calls[bs]
            print(f"  batch={bs:3d}  n={len(vals)}  values={vals}  mean={sum(vals) / len(vals):.4f}")
            for v in vals:
                xs.append(bs)
                ys.append(v)
        if len(set(xs)) >= 2:
            fixed, marginal = ols_fit(xs, ys)
            print(f"  OLS fit over {len(xs)} raw points: fixed={fixed:.3f}s marginal={marginal:.4f}s/item")
            for bs in sorted(calls):
                predicted_fixed_frac = fixed / (fixed + marginal * bs)
                print(f"    at batch={bs}: fixed fraction of predicted call = {predicted_fixed_frac:.1%}")

    for backend in ("yamnet", "hear", "crisperwhisper"):
        warm = [r["seconds"] for r in records if r.get("backend") == backend and r.get("kind") == "ensure_venv_warm"]
        if warm:
            print(f"\n{backend} ensure_venv_warm seconds: {warm}")

    for r in records:
        if r.get("kind") == "ensure_venv_warm" and r.get("seconds", 0) > 60:
            print(f"\nSUSPECT mislabelled warm record: {r}")
        if r.get("error"):
            print(
                f"\nfailed full_call: backend={r.get('backend')} batch={r.get('batch_size')} "
                f"seconds={r.get('seconds')} error={r['error'].splitlines()[0]!r}"
            )


def main() -> None:
    """Parse both job logs, print fits and warm costs for each, then compare batch=16 CPU vs GPU."""
    gpu_path, cpu_path = Path(sys.argv[1]), Path(sys.argv[2])
    gpu_records = parse_results(gpu_path)
    cpu_records = parse_results(cpu_path)
    report("GPU (node3805, A100 80GB)", gpu_records)
    report("CPU (node2803)", cpu_records)

    # Cross-host comparison at the one batch size both jobs completed for both backends.
    print("\n=== CPU vs GPU at batch=16 ===")
    for backend in ("yamnet", "hear"):
        gpu_vals = full_calls_by_backend(gpu_records, backend).get(16, [])
        cpu_vals = full_calls_by_backend(cpu_records, backend).get(16, [])
        if gpu_vals and cpu_vals:
            print(
                f"{backend}: GPU {gpu_vals} (mean {sum(gpu_vals) / len(gpu_vals):.2f}s) "
                f"vs CPU {cpu_vals} (mean {sum(cpu_vals) / len(cpu_vals):.2f}s)"
            )


if __name__ == "__main__":
    main()
