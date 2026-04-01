"""Core package for Coverage Gridworld reinforcement-learning experiments.

The package is organized around pluggable observation/reward variants and a
config-driven training pipeline. Public exports here expose runtime variant
selection helpers used by environment customization hooks.
"""

from project_rl.customization import current_variants, set_custom_variants

__all__ = ["set_custom_variants", "current_variants"]
