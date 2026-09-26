"""Resubmit the not-yet-started slices at half the CPU ask, over three disjoint ranges.

r3's own slices used 2.9 of their 8 cores and 10.9 of their 24 GB (`seff 23604010_0`), so 4 cores
and 14 GB is the ask the work actually takes and backfills roughly twice as readily.

The 37 slices already running keep their 8-core allocation and are excluded here, because two
concurrent tasks in one slice would write the same store.
"""

import pathlib

running = set(range(0, 22)) | set(range(40, 55))
todo = [i for i in range(240) if i not in running]
print(f"{len(todo)} slices to place, {len(running)} already running")

# three contiguous thirds of what is left, one per partition
third = len(todo) // 3
chunks = [todo[:third], todo[third : 2 * third], todo[2 * third :]]


def as_list(values: list[int]) -> str:
    """Slurm's compact array spec for a set of indices.

    Args:
        values: The slice indices, ascending.

    Returns:
        A comma-and-dash list.
    """
    out: list[str] = []
    start = prev = values[0]
    for value in values[1:] + [None]:
        if value == prev + 1:
            prev = value
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        if value is None:
            break
        start = prev = value
    return ",".join(out)


base = pathlib.Path("/tmp/aa/r4-pre.sbatch").read_text()
base = base.replace("#SBATCH --cpus-per-task=8", "#SBATCH --cpus-per-task=4")
base = base.replace("#SBATCH --mem=24G", "#SBATCH --mem=14G")
base = base.replace("export OMP_NUM_THREADS=8", "export OMP_NUM_THREADS=4")
base = base.replace("export OPENBLAS_NUM_THREADS=8", "export OPENBLAS_NUM_THREADS=4")
base = base.replace("export MKL_NUM_THREADS=8", "export MKL_NUM_THREADS=4")

plans = [
    ("pi2", "#SBATCH --partition=pi_satra\n", chunks[0]),
    ("ou2", "#SBATCH --partition=ou_bcs_normal\n#SBATCH --qos=normal\n", chunks[1]),
    ("pre2", "#SBATCH --partition=mit_preemptable\n", chunks[2]),
]
for name, header, chunk in plans:
    s = base.replace("#SBATCH --partition=mit_preemptable\n", header, 1)
    s = s.replace("#SBATCH --array=140-239%128", f"#SBATCH --array={as_list(chunk)}%200")
    s = s.replace("#SBATCH --job-name=triage-r4", f"#SBATCH --job-name=triage-r4-{name}")
    pathlib.Path(f"/tmp/aa/r4-{name}.sbatch").write_text(s)
    print(name, len(chunk), as_list(chunk)[:60])
