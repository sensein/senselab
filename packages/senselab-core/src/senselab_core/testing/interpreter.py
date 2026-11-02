"""A test-only interpreter: this interpreter on this process's import path, for workers of a source tree.

A test declares it as a worker's interpreter (or an env's `python`) when the backends it starts live outside any
installed distribution, such as test fixtures. It is a script whose shebang runs this interpreter under `-I`; it
prepends this process's `sys.path` and runs the `-m` module it is given as `__main__`.
"""

import functools
import json
import sys
import tempfile
from pathlib import Path

_BODY = """import runpy, sys
sys.path[:0] = {paths}
flag, run, module, *args = sys.argv[1:]
if (flag, run) != ("-I", "-m"):
    raise SystemExit(f"a test interpreter runs `-I -m <module>`, got {{sys.argv[1:]!r}}")
sys.argv = [module, *args]
runpy.run_module(module, run_name="__main__", alter_sys=True)
"""


@functools.cache
def source_tree_python() -> str:
    """Return the path of the test interpreter, written once per process under the system temporary directory."""
    path = Path(tempfile.mkdtemp(prefix="senselab-test-python-")) / "python"
    paths = json.dumps([p for p in sys.path if p])
    path.write_text(f"#!{sys.executable} -I\n" + _BODY.format(paths=paths))
    path.chmod(0o755)
    return str(path)
