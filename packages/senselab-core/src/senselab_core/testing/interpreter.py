"""A test-only interpreter: this interpreter on this process's import path, for workers of a source tree.

A test declares it as a worker's interpreter (or an env's `python`) when the backends it starts live outside any
installed distribution, such as test fixtures. It is a script whose shebang runs this interpreter under `-I`; it
prepends the `sys.path` of this process's latest call (kept in `paths.json` beside it) and runs the `-m` module it
is given as `__main__`.
"""

import functools
import json
import sys
import tempfile
from pathlib import Path

_BODY = """import json, runpy, sys
from pathlib import Path
sys.path[:0] = json.loads(Path(__file__).with_name("paths.json").read_text())
flag, run, module, *args = sys.argv[1:]
if (flag, run) != ("-I", "-m"):
    raise SystemExit(f"a test interpreter runs `-I -m <module>`, got {sys.argv[1:]!r}")
sys.argv = [module, *args]
runpy.run_module(module, run_name="__main__", alter_sys=True)
"""


@functools.cache
def _interpreter() -> Path:
    path = Path(tempfile.mkdtemp(prefix="senselab-test-python-")) / "python"
    path.write_text(f"#!{sys.executable} -I\n" + _BODY)
    path.chmod(0o755)
    return path


def source_tree_python() -> str:
    """Return the path of the test interpreter, written once per process, with this process's `sys.path` now."""
    path = _interpreter()
    path.with_name("paths.json").write_text(json.dumps([p for p in sys.path if p]))
    return str(path)
