from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PACKAGE_ROOT = PROJECT_ROOT / "coverage-gridworld"

if ENV_PACKAGE_ROOT.exists():
    env_path = str(ENV_PACKAGE_ROOT)
    if env_path not in sys.path:
        sys.path.insert(0, env_path)
