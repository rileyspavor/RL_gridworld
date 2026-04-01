"""Ensure the local Coverage Gridworld package is importable.

Training and evaluation scripts rely on the vendored ``coverage-gridworld``
directory. This module adds that directory to ``sys.path`` so imports resolve
without requiring package installation.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PACKAGE_ROOT = PROJECT_ROOT / "coverage-gridworld"


def ensure_env_package_path() -> None:
    """Prepend the bundled environment package path to ``sys.path``.

    This operation is idempotent and only applies when the local environment
    directory exists.
    """
    env_path = str(ENV_PACKAGE_ROOT)
    if ENV_PACKAGE_ROOT.exists() and env_path not in sys.path:
        sys.path.insert(0, env_path)


ensure_env_package_path()
