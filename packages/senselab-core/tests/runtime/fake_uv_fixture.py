"""A stand-in `uv` executable: run as `python fake_uv_fixture.py <uv args>`; builds no real env, downloads nothing.

`venv` makes the venv layout, `pip install` writes a `*.dist-info` per pin, `python dir` prints a directory.
Behaviour comes from the environment: `FAKE_UV_LOG` (each call's argv appended as a JSON line),
`FAKE_UV_UNMANAGED` (`venv` fails as uv does with no managed interpreter), `FAKE_UV_NO_WHEEL` (an index install
fails as uv does when no wheel matches), `FAKE_UV_DELAY` (seconds each `pip install` takes) and
`FAKE_UV_PYTHON_DIR` (what `python dir` prints).
"""

import json
import os
import sys
import time
from pathlib import Path


def _site(python: str) -> Path:
    return next((Path(python).parent.parent / "lib").glob("python*/site-packages"))


def _dist_info(site: Path, spec: str, local: str) -> None:
    name, _, version = spec.partition("==")
    (site / f"{name}-{version}{local}.dist-info").mkdir()


def _venv(args: list[str]) -> int:
    version = args[args.index("--python") + 1]
    if os.environ.get("FAKE_UV_UNMANAGED"):
        sys.stderr.write(f"error: No interpreter found for Python {version} in managed installations\n")
        return 2
    target = Path(args[-1])
    (target / "bin").mkdir(parents=True)
    (target / "bin" / "python").write_text("")
    (target / "lib" / f"python{version}" / "site-packages").mkdir(parents=True)
    (target / "pyvenv.cfg").write_text(f"version_info = {version.rstrip('t')}.4\n")
    return 0


def _pip_install(args: list[str]) -> int:
    time.sleep(float(os.environ.get("FAKE_UV_DELAY", "0")))
    site = _site(args[args.index("--python") + 1])
    if "--index-url" in args:
        if os.environ.get("FAKE_UV_NO_WHEEL"):
            sys.stderr.write("error: No matching distribution found for `torch==2.8.0`\n")
            return 1
        local = "+" + args[args.index("--index-url") + 1].rstrip("/").rsplit("/", 1)[-1]
        for spec in (a for a in args if "==" in a):
            _dist_info(site, spec, local)
    else:
        lock = Path(args[args.index("--requirement") + 1])
        for line in lock.read_text().splitlines():
            if "==" in line and not line.startswith("#"):
                _dist_info(site, line.split()[0], "")
    return 0


def main(args: list[str]) -> int:
    """Act as `uv <args>` for the three commands the env builder runs."""
    log = os.environ.get("FAKE_UV_LOG")
    if log:
        with open(log, "a") as out:
            out.write(json.dumps(args) + "\n")
    if args[:1] == ["venv"]:
        return _venv(args)
    if args[:2] == ["pip", "install"]:
        return _pip_install(args)
    if args[:2] == ["python", "dir"]:
        print(os.environ["FAKE_UV_PYTHON_DIR"])
        return 0
    sys.stderr.write(f"fake uv: unsupported {args}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
