"""This is the installation script for the senselab AI extension."""

import json
import shutil
import sys
from importlib.resources import files
from pathlib import Path
from typing import Tuple

# ---- Paths -----------------------------------------------------------------


def _share_jupyter_dir() -> Path:
    # <venv or env>/share/jupyter
    return Path(sys.prefix) / "share" / "jupyter"


def _extensions_dir() -> Path:
    # <venv>/share/jupyter/nbi_extensions/senselab_agent
    return _share_jupyter_dir() / "nbi_extensions" / "senselab_agent"


def _extension_dst_path() -> Path:
    return _extensions_dir() / "extension.json"


def _nbi_config_dst_path() -> Path:
    # <venv>/share/jupyter/nbi-config.json
    return _share_jupyter_dir() / "nbi-config.json"


# ---- Installers ------------------------------------------------------------


def _copy_json(src: Path, dst: Path, *, force: bool) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        # do nothing if present and not forcing
        return dst
    shutil.copyfile(src, dst)
    # sanity check: ensure it's valid JSON
    json.loads(dst.read_text(encoding="utf-8"))
    return dst


def install_extension_json(*, force: bool = False) -> Path:
    """Copy vendored extension.json into the ENV's share/jupyter path.

    This is so that Notebook-Intelligence can discover it.

    Args:
        force (bool): Whether to force the installation, overwriting existing files.

    Return:
        Path: The destination path for the nbi-config.json file.
    """
    src = files("senselab.agentic_interface.senselab_agent") / "extension.json"
    if not src.is_file():
        raise FileNotFoundError("Bundled extension.json not found inside package")
    assert isinstance(src, Path)
    return _copy_json(src, _extension_dst_path(), force=force)


def install_nbi_config_json(*, force: bool = False) -> Path:
    """Copy vendored nbi-config.json into <env-prefix>/share/jupyter/nbi-config.json.

    This is so global Notebook-Intelligence settings are available by default.

    Args:
        force (bool): Whether to force the installation, overwriting existing files.

    Return:
        Path: The destination path for the nbi-config.json file.
    """
    # Adjust the resource package/path below if your nbi-config.json lives elsewhere.
    src = files("senselab.agentic_interface") / "nbi-config.json"
    if not src.is_file():
        raise FileNotFoundError("Bundled nbi-config.json not found inside package")
    assert isinstance(src, Path)
    return _copy_json(src, _nbi_config_dst_path(), force=force)


def install(*, force: bool = False) -> Tuple[Path, Path]:
    """Install both extension.json and nbi-config.json.

    Return:
        Tuple[Path, Path]: The destination paths (extension_dst, nbi_config_dst).
    """
    ext_path = install_extension_json(force=force)
    cfg_path = install_nbi_config_json(force=force)
    return ext_path, cfg_path


def ensure_installed() -> Tuple[Path, Path]:
    """Idempotent helper; copies only if missing.

    Return:
        Tuple[Path, Path]: The destination paths (extension_dst, nbi_config_dst).
    """
    ext_dst = _extension_dst_path()
    cfg_dst = _nbi_config_dst_path()

    if not ext_dst.exists():
        install_extension_json(force=False)
    else:
        # validate JSON even if present (fails fast on corruption)
        json.loads(ext_dst.read_text(encoding="utf-8"))

    if not cfg_dst.exists():
        install_nbi_config_json(force=False)
    else:
        json.loads(cfg_dst.read_text(encoding="utf-8"))

    return ext_dst, cfg_dst
