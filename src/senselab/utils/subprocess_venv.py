"""Runtime subprocess venv manager for isolated backend dependencies.

Uses uv to create and manage isolated virtual environments for backends
that conflict with the core senselab installation (e.g., different torch
versions, Python version requirements). Functions in isolated backends
are called via subprocess with JSON IPC.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

from filelock import FileLock

logger = logging.getLogger("senselab")

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "senselab" / "venvs"


def _cache_dir() -> Path:
    """Return the directory for cached subprocess venvs."""
    cache = Path(os.environ.get("SENSELAB_VENV_CACHE", str(_DEFAULT_CACHE_DIR)))
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _find_uv() -> str:
    """Find the uv binary."""
    uv = shutil.which("uv")
    if uv:
        return uv
    # Check common locations
    for candidate in [
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
    ]:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    )


def ensure_venv(
    name: str,
    requirements: list[str],
    python_version: Optional[str] = None,
) -> Path:
    """Create or reuse an isolated virtual environment.

    Args:
        name: Unique identifier for this venv (e.g., "coqui", "ppgs").
        requirements: List of pip install specs (e.g., ["coqui-tts~=0.27"]).
        python_version: Python version to use (e.g., "3.11"). Defaults to
            the current interpreter's version.

    Returns:
        Path to the venv directory.
    """
    venv_dir = _cache_dir() / name
    lock_path = _cache_dir() / f"{name}.lock"
    marker = venv_dir / ".senselab-installed"

    with FileLock(str(lock_path), timeout=600):
        # Check if venv already exists and is valid
        if marker.is_file():
            stored = json.loads(marker.read_text())
            if stored.get("requirements") == sorted(requirements):
                logger.debug("Reusing existing venv: %s", venv_dir)
                return venv_dir

        # Create fresh venv
        uv = _find_uv()
        py_ver = python_version or f"{sys.version_info.major}.{sys.version_info.minor}"

        logger.info("Creating isolated venv '%s' with Python %s", name, py_ver)

        # Remove old venv if exists
        if venv_dir.exists():
            shutil.rmtree(venv_dir)

        # Create venv
        try:
            subprocess.run(
                [uv, "venv", "--python", py_ver, str(venv_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("Failed to create venv '%s': %s", name, exc.stderr)
            raise

        # Install requirements
        if requirements:
            try:
                subprocess.run(
                    [uv, "pip", "install", "--python", str(venv_dir / "bin" / "python"), *requirements],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                logger.error("Failed to install requirements in venv '%s': %s", name, exc.stderr)
                raise

        # Write marker with installed requirements
        marker.write_text(json.dumps({
            "requirements": sorted(requirements),
            "python_version": py_ver,
        }))

        logger.info("Venv '%s' ready at %s", name, venv_dir)
        return venv_dir


def run_in_venv(
    name: str,
    requirements: list[str],
    script: str,
    args: Optional[dict[str, object]] = None,
    python_version: Optional[str] = None,
    timeout: int = 600,
) -> object:
    """Execute a Python script in an isolated venv via subprocess.

    The script receives args as JSON on stdin and must print its result
    as JSON to stdout. Stderr is captured for error reporting.

    Args:
        name: Venv identifier.
        requirements: Pip install specs for this venv.
        script: Python script to execute (as a string).
        args: Dictionary of arguments, serialized as JSON to stdin.
        python_version: Python version for the venv.
        timeout: Maximum execution time in seconds.

    Returns:
        The deserialized JSON result from the script's stdout.

    Raises:
        RuntimeError: If the script fails or returns non-JSON output.
    """
    venv_dir = ensure_venv(name, requirements, python_version)
    python = str(venv_dir / "bin" / "python")

    input_json = json.dumps(args or {})

    try:
        result = subprocess.run(
            [python, "-c", script],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Isolated venv '{name}' timed out after {timeout}s"
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"Isolated venv '{name}' failed (exit {result.returncode}):\n{result.stderr}"
        )

    stdout = result.stdout.strip()
    if not stdout:
        return None

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Isolated venv '{name}' returned non-JSON output:\n{stdout}\nstderr:\n{result.stderr}"
        ) from exc


def run_function_in_venv(
    name: str,
    requirements: list[str],
    module: str,
    function: str,
    input_files: Optional[dict[str, str]] = None,
    output_file: Optional[str] = None,
    args: Optional[dict[str, object]] = None,
    python_version: Optional[str] = None,
    timeout: int = 600,
) -> object:
    """Execute a function from a module in an isolated venv.

    For large data (audio, tensors), use input_files and output_file
    to pass file paths instead of serializing data through JSON.

    Args:
        name: Venv identifier.
        requirements: Pip install specs.
        module: Module path (e.g., "TTS.api").
        function: Function name in the module.
        input_files: Dict mapping argument names to file paths for large data.
        output_file: Path where the function should write its output.
        args: JSON-serializable arguments.
        python_version: Python version for the venv.
        timeout: Maximum execution time in seconds.

    Returns:
        The function's return value (deserialized from JSON), or the
        output_file path if output_file was specified.
    """
    script = f'''
import json
import sys

args = json.loads(sys.stdin.read())
input_files = args.pop("__input_files__", {{}})
output_file = args.pop("__output_file__", None)

from {module} import {function}
result = {function}(**args)

if output_file:
    # Write result to file — handle tensors, arrays, and plain data
    try:
        import torch
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu()
    except ImportError:
        pass
    import numpy as np
    if hasattr(result, 'numpy') or isinstance(result, np.ndarray):
        arr = result.numpy() if hasattr(result, 'numpy') else result
        np.save(output_file, arr)
    else:
        with open(output_file, 'w') as f:
            json.dump(result, f)
    print(json.dumps({{"output_file": output_file}}))
else:
    print(json.dumps(result, default=str))
'''

    full_args = dict(args or {})
    if input_files:
        full_args["__input_files__"] = input_files
    if output_file:
        full_args["__output_file__"] = output_file

    return run_in_venv(
        name=name,
        requirements=requirements,
        script=script,
        args=full_args,
        python_version=python_version,
        timeout=timeout,
    )
