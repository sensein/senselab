# ruff: noqa  -- benchmark harness run inside the subprocess venvs, not part of the package.
"""Sample this process's macOS phys footprint every few seconds in a thread."""

import os, re, subprocess, threading, time


class Sampler:
    def __init__(self, every=3.0):
        self.series = []
        self._stop = False
        self._t = threading.Thread(target=self._run, args=(every,), daemon=True)
        self._t.start()

    def _run(self, every):
        t0 = time.time()
        while not self._stop:
            try:
                out = subprocess.run(["footprint", str(os.getpid())], capture_output=True, text=True).stdout
                m = re.search(r"Footprint: ([\d.]+) (KB|MB|GB)", out)
                if m:
                    v = float(m.group(1)) * {"KB": 1e-6, "MB": 1e-3, "GB": 1.0}[m.group(2)]
                    self.series.append((round(time.time() - t0, 1), round(v, 2)))
                    if v > getattr(self, "best", 0):
                        self.best = v
                        self.best_dump = out[:3000]
            except Exception:
                pass
            time.sleep(every)

    def mark(self, label):
        self.series.append((label, None))

    def stop(self):
        self._stop = True
        return self.series
