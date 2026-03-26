from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


METRIC_FIELDS = (
    "episodes",
    "mean_coverage",
    "success_rate",
    "death_rate",
    "timeout_rate",
    "mean_reward",
    "mean_length",
    "min_coverage",
    "max_coverage",
)


@dataclass(slots=True)
class DashboardPaths:
    root: Path
    runs_dir: Path
    configs_dir: Path
    results_dir: Path
    report_dir: Path
    queue_file: Path
    static_dir: Path

    @classmethod
    def from_root(cls, root: Path, queue_file: str) -> "DashboardPaths":
        package_dir = Path(__file__).resolve().parent
        queue_path = Path(queue_file)
        if not queue_path.is_absolute():
            queue_path = root / queue_path
        return cls(
            root=root,
            runs_dir=root / "runs",
            configs_dir=root / "configs" / "experiments",
            results_dir=root / "results",
            report_dir=root / "report",
            queue_file=queue_path,
            static_dir=package_dir / "static",
        )
