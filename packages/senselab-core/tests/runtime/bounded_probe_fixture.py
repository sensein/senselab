"""Self-bounding probes: a finite stand-in for an endless source, and a subprocess under hard limits.

`FiniteSource` serves at most `expected + slack` bytes and then raises, whatever the reader asks for.
`run_bounded` runs a probe function in a child interpreter under `RLIMIT_FSIZE`, `RLIMIT_AS` (where the
platform accepts one below the current mapping; macOS does not) and a timeout, and returns what the probe
returned. A probe is `<module>:<function>`, the module loaded from `<where>/<module>.py` in the child; it
takes string arguments and returns a JSON value. Run as a script, this module is that child.
"""

import importlib.util
import io
import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from typing import Any

MIB = 1 << 20
DEFAULT_FSIZE = 64 * MIB
DEFAULT_ADDRESS_SPACE = 4 << 30
DEFAULT_TIMEOUT_S = 120.0
HERE = Path(__file__).resolve().parent


class ProbeOverrun(RuntimeError):
    """A finite source was read past its bound."""


class FiniteSource(io.RawIOBase):
    """Zeros, at most `expected + slack` bytes of them; a read after the last raises `ProbeOverrun`.

    Attributes:
        served: Bytes handed out so far.
        limit: The most it will ever hand out.
    """

    def __init__(self, expected: int, *, slack: int = MIB) -> None:
        """Serve up to `expected + slack` bytes."""
        super().__init__()
        self.limit = expected + slack
        self.served = 0

    def readable(self) -> bool:
        """Return True."""
        return True

    def read(self, size: int | None = -1) -> bytes:
        """Return up to `size` zeros (a chunk of at most 1 MiB when `size` is negative or None)."""
        if self.served >= self.limit:
            raise ProbeOverrun(f"read past the probe's bound of {self.limit} bytes")
        want = MIB if size is None or size < 0 else size
        n = min(want, self.limit - self.served)
        self.served += n
        return b"\0" * n


class ProbeFailed(AssertionError):
    """The bounded child did not return a report: it was killed, timed out or raised."""


def _limits(fsize: int, address_space: int) -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (address_space, address_space))
    except (ValueError, OSError):
        pass


def run_bounded(
    target: str,
    *args: str,
    where: Path = HERE,
    fsize: int = DEFAULT_FSIZE,
    address_space: int = DEFAULT_ADDRESS_SPACE,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> Any:  # noqa: ANN401 - whatever JSON the probe returned
    """Run `target` (`module:function` importable from `where`) with `args` in a bounded child; return its value.

    Raises:
        ProbeFailed: the child exited without a report (a limit hit, an uncaught error); the message holds its
            exit code and the end of its stderr.
        subprocess.TimeoutExpired: the child ran past `timeout` and was killed.
    """
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(where), target, json.dumps(list(args))],
        capture_output=True,
        text=True,
        timeout=timeout,
        preexec_fn=lambda: _limits(fsize, address_space),
        env={k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")},
    )
    lines = proc.stdout.strip().splitlines()
    if proc.returncode != 0 or not lines:
        raise ProbeFailed(f"{target} ended with exit {proc.returncode} and no report:\n{proc.stderr[-4000:]}")
    return json.loads(lines[-1])


def _main(argv: list[str]) -> None:
    where, target, raw = argv
    sys.path[:0] = [where, str(HERE)]
    module, _, function = target.partition(":")
    spec = importlib.util.spec_from_file_location(module, Path(where) / f"{module}.py")
    assert spec is not None and spec.loader is not None, f"no probe module {module} in {where}"
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[module] = loaded
    spec.loader.exec_module(loaded)
    value = getattr(loaded, function)(*json.loads(raw))
    sys.stdout.write("\n" + json.dumps(value) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    _main(sys.argv[1:])
