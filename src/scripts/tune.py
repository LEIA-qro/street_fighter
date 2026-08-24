import argparse
import importlib
import os
import sys
from pathlib import Path

# Add src directory to path to handle imports correctly
sys.path.insert(0, str(Path(__file__).parents[1]))

from core import config
from core.env_tools import SFv2_make_env, failsafe_env

def main():
    parser = argparse.ArgumentParser(description="Street Fighter II RL Hyperparameter Tuning via Optuna")
    parser.add_argument("--algo", required=True, choices=["ppo", "sac", "dqn"], help="RL algorithm to tune")
    parser.add_argument("--env", default="v2", choices=["v1", "v2", "v3"], help="Environment version")
    parser.add_argument("--trials", type=int, default=2, help="Number of Optuna trials to run")
    parser.add_argument("--study_name", type=str, default="ppo_sf2_tuning", help="Unique name for the Optuna study")
    
    # Advanced Tuning Settings
    parser.add_argument("--load_zip", type=str, default=None)
    parser.add_argument("--load_pkl", type=str, default=None)
    parser.add_argument("--phase", type=str, default="0")
    parser.add_argument("--timesteps", type=int, default=50000, help="Timesteps per trial")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    config.generate_lua_config()

    # GPU Verification
    import torch
    # Performance Optimization: Restrict PyTorch to 2 CPU threads during tuning
    # This prevents it from hijacking logical cores alongside active EmuHawk emulators.
    torch.set_num_threads(2)

    if torch.cuda.is_available():
        print(f"[Hardware] Tuning on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[Hardware] WARNING: No GPU detected for tuning. Running on CPU.")

    # Device Auto-Logic
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Suppress SB3 GPU Warnings for MlpPolicy
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

    print(f"[CLI] Initializing Hyperparameter Tuning for {args.algo.upper()}...")
    print(f"[CLI] Environment: SF2 {args.env.upper()}")
    print(f"[CLI] Target Trials: {args.trials}")
    print(f"[CLI] Study Name: {args.study_name}")
    print(f"[CLI] Compute Device: {device}")

    # Dynamic dispatch — loads the build_agent() factory from agents/{algo}/__init__.py
    try:
        module = importlib.import_module(f"agents.{args.algo}")
        agent = module.build_agent()
        
        # Factory function for environment creation
        env_fn = lambda rank: SFv2_make_env(rank, version=args.env)
        
        # Run the tuning process
        agent.tune(
            env_fn, 
            args.trials, 
            study_name=args.study_name, 
            load_zip=args.load_zip, 
            load_pkl=args.load_pkl, 
            start_phase=args.phase, 
            timesteps=args.timesteps,
            device=device
        )
        
    except ImportError as e:
        print(f"[ERROR] Could not load agent module for {args.algo}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[CRITICAL] Tuning process failed: {e}")
        sys.exit(1)
    finally:
        failsafe_env()

if __name__ == "__main__":
    main()
