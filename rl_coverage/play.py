from __future__ import annotations

import argparse
import random
import time

from rl_coverage.env_factory import make_env


ACTION_LOOKUP = {
    "a": 0,
    "s": 1,
    "d": 2,
    "w": 3,
    "e": 4,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual or random play for Coverage Gridworld.")
    parser.add_argument("--config", default=None, help="Optional experiment config used to build the environment.")
    parser.add_argument("--env-id", default="sneaky_enemies", help="Environment id when no config is provided.")
    parser.add_argument("--policy", choices=["human", "random"], default="human")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay between random-policy steps.")
    return parser.parse_args(argv)


def human_action() -> int:
    raw = input("action [WASD, E=stay, 0-4]: ").strip().lower()
    if raw.isdigit() and raw in {"0", "1", "2", "3", "4"}:
        return int(raw)
    return ACTION_LOOKUP.get(raw, 4)


def random_action() -> int:
    return random.randint(0, 4)


def build_default_config(env_id: str) -> dict:
    return {
        "environment": {
            "id": env_id,
            "map_suite": None,
            "render_mode": "human",
            "activate_game_status": True,
        },
        "reward": {"name": "native", "params": {}},
        "observation": {"name": "native", "params": {}},
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.config is not None:
        from rl_coverage.config import load_config

        config = load_config(args.config)
        config["environment"]["render_mode"] = "human"
        config["environment"]["activate_game_status"] = True
    else:
        config = build_default_config(args.env_id)

    env = make_env(config, render_mode="human")
    choose_action = human_action if args.policy == "human" else random_action

    try:
        for _ in range(args.episodes):
            observation, _ = env.reset()
            done = False
            while not done:
                action = choose_action()
                observation, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                if args.policy == "random" and args.sleep > 0:
                    time.sleep(args.sleep)
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
