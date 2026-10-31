"""Remove unimportable lazy modules from ``sys.modules`` before collection hooks walk it."""

import sys
import types


def prune_unimportable_lazy_modules() -> list[str]:
    """Delete every non-``ModuleType`` entry of ``sys.modules`` whose attribute access raises ``ImportError``.

    Returns:
        The names of the removed entries.
    """
    removed: list[str] = []
    for name, module in list(sys.modules.items()):
        if module is None or type(module) is types.ModuleType:
            continue
        try:
            getattr(module, "__file__", None)
        except ImportError:
            del sys.modules[name]
            removed.append(name)
        except Exception:  # noqa: BLE001, S110 - only ImportError marks an unimportable lazy module
            pass
    return removed
