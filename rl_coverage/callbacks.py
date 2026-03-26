from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from rl_coverage.config import save_json
from rl_coverage.metrics import evaluate_model, evaluation_text


class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        env_builder,
        eval_episodes: int,
        eval_freq: int,
        deterministic: bool,
        output_dir: str | Path,
        save_best_model: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.env_builder = env_builder
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.output_dir = Path(output_dir)
        self.save_best_model = save_best_model
        self.best_mean_coverage = float("-inf")
        self.evaluations: list[dict] = []

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True

        evaluation = evaluate_model(
            self.model,
            env_builder=self.env_builder,
            episodes=self.eval_episodes,
            deterministic=self.deterministic,
        )
        record = {
            "timesteps": self.num_timesteps,
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
