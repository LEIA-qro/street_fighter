import os, argparse, importlib, time
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parents[1]))

from core import config
from core.env_tools import SFv2_make_env, failsafe_env

def main():
    config.generate_lua_config()

    # GPU Verification
    import torch
    if torch.cuda.is_available():
        print(f"[Hardware] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[Hardware] WARNING: No GPU detected. Running on CPU.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=["ppo", "sac", "dqn"])
    parser.add_argument("--env",  default="v2",  choices=["v1", "v2"])
    parser.add_argument("--steps", type=int, default=config.STARTING_TOTAL_TIMESTEPS)
    parser.add_argument("--load_zip", type=str, default=None)
    parser.add_argument("--load_pkl", type=str, default=None)
    parser.add_argument("--phase", type=str, default="0")
    parser.add_argument("--device", type=str, default="auto")

    # Advanced Hyperparameter Overrides
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--clip_range", type=float, default=0.0)
    args = parser.parse_args()

    # Device Auto-Logic
    device = args.device
    if device == "auto":
        device = "cpu" if args.algo.lower() == "ppo" else "cuda"
    
    # Suppress SB3 GPU Warnings for MlpPolicy
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

    # Dynamic dispatch
    module  = importlib.import_module(f"agents.{args.algo}")
    agent   = module.build_agent()
    env_fn  = lambda rank: SFv2_make_env(rank, version=args.env)

    save_dir = os.path.join(config.get_directory()["production"], args.algo)
    os.makedirs(save_dir, exist_ok=True)

    total_target_steps = args.steps
    steps_completed = 0

    current_load_zip = args.load_zip
    current_load_pkl = args.load_pkl
    current_phase = args.phase
    original_phase_choice = args.phase  # Preserve special names like RYU_ONLY

    max_retries = 10
    retry_count = 0

    while steps_completed < total_target_steps:
        try:
            remaining_steps = total_target_steps - steps_completed
            if remaining_steps <= 0:
                break

            print(f"\n[RETRY LOOP] Attempt {retry_count + 1} | Goal: {total_target_steps} | Done: {steps_completed} | Remaining: {remaining_steps} | Device: {device}")

            # Check for existing crash saves if not the first attempt or if explicitly resuming
            if retry_count > 0:
                crash_zip = os.path.join(save_dir, config.MODEL_NAME + "_CRASH_SAVE.zip")
                crash_pkl = os.path.join(save_dir, config.MODEL_NAME + "_vecnormalize_CRASH_SAVE.pkl")
                if os.path.exists(crash_zip):
                    # Relativize for consistency with initial args
                    current_load_zip = os.path.relpath(crash_zip, config.PROJECT_ROOT).replace("\\", "/")
                    print(f"[RETRY LOOP] Found crash save, resuming from: {current_load_zip}")
                if os.path.exists(crash_pkl):
                    current_load_pkl = os.path.relpath(crash_pkl, config.PROJECT_ROOT).replace("\\", "/")

            agent.train(
                env_fn, 
                save_dir, 
                remaining_steps, 
                load_zip=current_load_zip, 
                load_pkl=current_load_pkl, 
                start_phase=current_phase, 
                lr=args.lr, 
                ent_coef=args.ent_coef, 
                clip_range=args.clip_range,
                device=device
            )

            # If we reached here, training finished normally
            print("[RETRY LOOP] Training completed successfully.")
            break

        except (RuntimeError, ConnectionResetError, Exception) as e:
            # We catch generic Exception because Agent.train might raise something else on socket death
            retry_count += 1
            if retry_count > max_retries:
                print(f"[RETRY LOOP] Max retries ({max_retries}) exceeded. Giving up.")
                sys.exit(1)

            print(f"[RETRY LOOP] Recoverable error detected: {e}")
            print("[RETRY LOOP] Cooling down for 10 seconds before restart...")
            failsafe_env() # Ensure everything is dead
            time.sleep(10)

            # Read progress from curriculum_state.json if available
            state_path = os.path.join(save_dir, "curriculum_state.json")
            if os.path.exists(state_path):
                import json
                try:
                    with open(state_path, "r") as f:
                        state_data = json.load(f)
                        # Sync steps and phase
                        steps_completed = state_data.get("num_timesteps", steps_completed)

                        # Preserve special phase names if that was the original choice
                        if original_phase_choice in ["RYU_ONLY", "CUSTOM"]:
                            current_phase = original_phase_choice
                        else:
                            current_phase = str(state_data.get("current_phase", current_phase))

                        print(f"[RETRY LOOP] Progress synced from disk: {steps_completed} steps, Phase {current_phase}")

                        # BREAK IF ALREADY DONE
                        if steps_completed >= total_target_steps:
                            print("[RETRY LOOP] Target steps already reached. Exiting.")
                            break
                except Exception as ex:
                    print(f"[RETRY LOOP] Could not read curriculum state: {ex}")
        except KeyboardInterrupt:
            print("\n[RETRY LOOP] Training stopped by user (Ctrl+C).")
            break

if __name__ == "__main__":
    main()