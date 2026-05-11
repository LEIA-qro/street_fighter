# src/scripts/train.py
import os, argparse, importlib
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parents[1]))

from core import config
from core.env_tools import SFv2_make_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=["ppo", "sac", "dqn"])
    parser.add_argument("--env",  default="v2",  choices=["v1", "v2"])
    parser.add_argument("--steps", type=int, default=config.STARTING_TOTAL_TIMESTEPS)
    args = parser.parse_args()

    # Dynamic dispatch — adding a new algorithm requires zero changes here
    module  = importlib.import_module(f"agents.{args.algo}")
    agent   = module.build_agent()          # factory function in agents/ppo/__init__.py
    env_fn  = lambda rank: SFv2_make_env(rank, version=args.env)
    save_dir = os.path.join(config.get_directory()["production"], args.algo)
    os.makedirs(save_dir, exist_ok=True)

    agent.train(env_fn, save_dir, args.steps)

if __name__ == "__main__":
    main()