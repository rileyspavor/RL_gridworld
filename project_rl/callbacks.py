"""Stable-Baselines callbacks used during experiment training.

The callback defined here periodically evaluates the current policy on a
separate environment builder, logs metrics, and optionally checkpoints the best
model by mean coverage.
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from project_rl.config import save_json
from project_rl.metrics import evaluate_model, evaluation_text


class PeriodicEvalCallback(BaseCallback):
    """Run periodic evaluation passes while PPO training is in progress.

    The callback writes rolling evaluation artifacts into the run directory so
    later scripts can build reports and learning curves without rerunning
    evaluation.

    Args:
        env_builder: Zero-argument callable that constructs an evaluation env.
        eval_episodes: Number of episodes per evaluation pass.
        eval_freq: Timestep interval between evaluations.
        eval_schedule: Optional explicit list of timestep checkpoints at which
            evaluation should run.
        deterministic: Whether policy actions are deterministic for evaluation.
        output_dir: Directory for evaluation JSON and model checkpoints.
        save_best_model: Whether to save a checkpoint on coverage improvement.
        verbose: Stable-Baselines callback verbosity level.
    """
    def __init__(
        self,
        env_builder,
        eval_episodes: int,
        eval_freq: int,
        deterministic: bool,
        output_dir: str | Path,
        save_best_model: bool = True,
        verbose: int = 1,
        eval_schedule: list[int] | None = None,
    ):
        super().__init__(verbose=verbose)
        self.env_builder = env_builder
        self.eval_episodes = int(eval_episodes)
        self.eval_freq = int(eval_freq)
        self.eval_schedule = sorted({int(item) for item in (eval_schedule or []) if int(item) > 0})
        self._next_schedule_index = 0
        self.deterministic = bool(deterministic)
        self.output_dir = Path(output_dir)
        self.save_best_model = bool(save_best_model)
        self.best_mean_coverage = float("-inf")
        self.evaluations: list[dict] = []

    def _should_evaluate(self) -> bool:
        """Return True when the current timestep reaches an evaluation trigger."""
        if self.eval_schedule:
            if self._next_schedule_index >= len(self.eval_schedule):
                return False
            next_timestep = self.eval_schedule[self._next_schedule_index]
            if self.num_timesteps < next_timestep:
                return False
            self._next_schedule_index += 1
            return True

        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return False
        return True

    def _on_step(self) -> bool:
        """Evaluate and log metrics when the configured interval is reached.

        Returns:
            ``True`` to allow training to continue.
        """
        if not self._should_evaluate():
            return True

        evaluation = evaluate_model(
            self.model,
            env_builder=self.env_builder,
            episodes=self.eval_episodes,
            deterministic=self.deterministic,
        )
        record = {
            "timesteps": int(self.num_timesteps),
            **evaluation,
        }
        self.evaluations.append(record)

        save_json(self.output_dir / "latest_evaluation.json", record)
        save_json(self.output_dir / "evaluations.json", {"items": self.evaluations})

        summary = evaluation["summary"]
        if self.verbose:
            print(f"[eval @ {self.num_timesteps}] {evaluation_text(summary)}")

        self.logger.record("eval/mean_reward", summary["mean_reward"])
        self.logger.record("eval/mean_coverage", summary["mean_coverage"])
        self.logger.record("eval/success_rate", summary["success_rate"])
        self.logger.record("eval/death_rate", summary["death_rate"])

        if self.save_best_model and summary["mean_coverage"] > self.best_mean_coverage:
            self.best_mean_coverage = summary["mean_coverage"]
            self.model.save(self.output_dir / "best_model")
            save_json(self.output_dir / "best_model_metrics.json", record)

        return True
