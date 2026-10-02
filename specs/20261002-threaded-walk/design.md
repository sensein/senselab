# Threaded directory walks

`senselab.utils.fastio` holds a multithreaded `os.walk` adapted from TensorFlow's `fastio.walk`
(Apache-2.0), as published at <https://gist.github.com/satra/0e02cd7554672120cf8073df3986f302>.

## Why

On the ORCD network filesystems one metadata call (`stat`, `readdir`, `open`) costs about a
millisecond, and a serial walk issues them one at a time. The r9 run tree holds about 2.2 M files in
62,550 run directories; the derivatives finalize ran six serial passes over it (`find -L` twice,
`find` twice, `find -type l`, `grep -rl` over every store) and took more than three hours, almost all
of it waiting on the filesystem. A pool of threads keeps many calls in flight, so the latency
overlaps instead of adding up. The CPU work per entry is small, and `os.scandir` and `stat` release
the GIL, so threads suffice.

## What it offers

- `walk(top, threads, follow_symlinks, keep, on_error)`: `os.walk`'s `(path, dirs, files)` in no
  particular order, `dirs` and `files` sorted. A symlink to a directory is listed in `dirs` either
  way and descended into only with `follow_symlinks`, as `os.walk(followlinks=...)` does.
- `map_files(top, fn, pattern, ...)`: applies `fn` to each matching file inside the workers, for
  per-file `stat` or read work.
- `find(top, pattern, ...)`: the matching paths, sorted, so a caller that iterated
  `sorted(root.rglob(pattern))` keeps its order. `rglob` in Python 3.12 does not descend through
  symlinked directories; `find` defaults to the same.
- `ordered_map(fn, items, threads)`: a thread pool that keeps input order, for a fixed-depth scan
  split by top-level directory (`recording_dirs`).

Listing errors are collected and raised as `WalkError` after the walk, or handed to `on_error`; the
gist printed and dropped them. An exception from a mapped function stops the walk and reaches the
caller. Closing a generator early stops and joins the pool.

## Where it is used

| Call site | Before | After |
| --- | --- | --- |
| `recording_vectors.recording_dirs` (parquet scan, review-page extract) | nested sorted `iterdir` + `exists`, serial | the same per participant, participants in a thread pool, order kept |
| `measure_stats.readings` | `sorted(root.rglob("store.jsonl"))` | `fastio.find` |
| `corpus_report.decisions` | two sorted `rglob`s | two `fastio.find`s |
| `replay_diff.read_rows` | `sorted(root.rglob("*.jsonl"))` | `fastio.find` |
| `scripts/triage_rerender.run_dirs` | `sorted(rglob("store.jsonl"))` | `fastio.find` |

`scripts/triage_recording_vectors.py` and `scripts/free_speech_review_page.py extract` take
`--walk-threads` (default `fastio.DEFAULT_THREADS`, 32). `speaker_vectors` shards by subject, so
each task lists 1/64 of the tree and was left serial.

## Measurement

Indicative only: one run on a shared `mit_quicktest` node (node1600, 8 CPUs, 64 threads), over the
r9 tree at `ee9374d2`. Each method was timed on its own disjoint 1/64 of the subjects (24 subjects
each), so neither ran on a metadata cache the other had warmed; equality was checked by re-running
the threaded method on the serial method's subset.

| | serial | threaded | ratio |
| --- | --- | --- | --- |
| full walk (`os.walk` vs `fastio.walk`) | 13,777 files in 4.6 s, 3,026 files/s | 15,030 files in 0.7 s, 21,477 files/s | 7.1× |
| `recording_dirs` (`_recordings_of` per participant) | 977 runs in 0.7 s, 1,451 runs/s | 962 runs in 0.1 s, 12,219 runs/s | 8.4× |

On the serial subset, the threaded walk found the same 5,913 directories and 13,777 files, and the
threaded `recording_dirs` returned the same run directories in the same order.

At those rates a full serial walk of the 2.2 M-file tree takes about 12 minutes per pass and a
threaded one under 2. The three-hour finalize was six such passes plus a full read of every store,
so the walk alone is not all of it; reading the stores is the rest.
