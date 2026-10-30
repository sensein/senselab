"""Exclusive-create race on a shared directory, run concurrently from many nodes.

    python race.py run DIR NAMES     # each process: try to win every name, by mkdir and by staged-dir rename
    python race.py check DIR NAMES   # every name must have exactly one winner per primitive
"""
import json, os, socket, sys, uuid
from pathlib import Path

mode, root, names = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
me = f"{socket.gethostname().split('.')[0]}.{os.environ.get('SLURM_ARRAY_TASK_ID', 'x')}.{os.getpid()}"
for sub in ("mkdir", "rename", "tmp", "wins"):
    (root / sub).mkdir(parents=True, exist_ok=True)
if mode == "run":
    won = {"mkdir": [], "rename": []}
    for i in range(names):
        try:
            os.mkdir(root / "mkdir" / str(i))
            won["mkdir"].append(i)
        except FileExistsError:
            pass
        token = uuid.uuid4().hex
        staged = root / "tmp" / token
        staged.mkdir()
        (staged / "owner").write_text(token)
        try:
            os.rename(staged, root / "rename" / str(i))
        except OSError:
            pass
        try:
            if (root / "rename" / str(i) / "owner").read_text() == token:
                won["rename"].append(i)
            else:
                (staged / "owner").unlink(); staged.rmdir()
        except FileNotFoundError:
            pass
    (root / "wins" / f"{me}.json").write_text(json.dumps(won))
else:
    counts = {"mkdir": [0] * names, "rename": [0] * names}
    files = list((root / "wins").glob("*.json"))
    for f in files:
        for k, v in json.loads(f.read_text()).items():
            for i in v:
                counts[k][i] += 1
    hosts = sorted({f.name.split(".")[0] for f in files})
    out = {"processes": len(files), "hosts": hosts}
    for k, c in counts.items():
        out[k] = {"names": names, "exactly_one": sum(x == 1 for x in c), "zero": sum(x == 0 for x in c), "more": sum(x > 1 for x in c)}
    print(json.dumps(out, indent=1))
    sys.exit(0 if all(out[k]["exactly_one"] == names for k in counts) else 1)
