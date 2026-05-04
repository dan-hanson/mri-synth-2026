"""Shared sys.path bootstrap for entry-point scripts.

Any script that lives outside `src/` (training, inference, eval, dataset
smoke tests, etc.) needs `src/` on sys.path before it can `import
mri_missing.*`. Doing this in one place means we only have one piece of
path logic to maintain.

Usage — put this as the FIRST import in any entry-point script:

    import _path_bootstrap  # noqa: F401  (must come before mri_missing.*)
    from mri_missing.config import load_config
    ...
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_MARKERS = (".git", "pyproject.toml", "setup.py", "configs")


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        for marker in _MARKERS:
            if (candidate / marker).exists():
                return candidate
    return Path.cwd()


PROJECT_ROOT = _find_project_root(_HERE)
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
