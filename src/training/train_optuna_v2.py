import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import optuna
import json  # <-- ADD: for saving best params to disk
# import subprocess 
   
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines3.common.evaluation import evaluate_policy


import config
from selective_norm import SelectiveVecNormalize
from env_tools import failsafe_env, SFv2_make_env

directories = config.get_directory()

BEST_PARAMS_PATH = os.path.join(directories["optuna"], "best_params_v2.json")
BEST_REWARD_PATH = os.path.join(directories["optuna"], "best_reward_v2.txt")

def _print_and_save_best(study: optuna.Study):
    """
    FIX: Dedicated reporter called at every save point AND at script end.
    Saves best params to JSON so a crash doesn't lose the results.
    """
    try:
        best = study.best_trial
    except ValueError:
        print("[Optuna] No completed trials yet.")
        return

    print("\n" + "="*60)
    print(f"  BEST TRIAL: #{best.number}")
    print(f"  BEST MEAN REWARD: {best.value:.4f}")
    print("  BEST HYPERPARAMETERS:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print("="*60 + "\n")

    # Persist to disk — survives crashes
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump({
            "trial_number": best.number,
            "value": best.value,
            "params": best.params,
        }, f, indent=2)
    print(f"[Optuna] Best params saved → {BEST_PARAMS_PATH}")

def objective(trial):
    # THE MASSIVE MATRIX SEARCH SPACE
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.01, 0.1)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    n_steps = trial.suggest_categorical("n_steps", [2048, 4096, 8192])
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])

    if n_steps * config.N_ENVS % batch_size != 0:
        raise ValueError(
            f"n_steps ({n_steps}) * N_ENVS ({config.N_ENVS}) "
            f"is not divisible by batch_size ({batch_size}). Skipping."
        )

    print(f"\n--- Starting Trial {trial.number} ---")

    n_envs = config.N_ENVS

    env = None
    model = None

    try:
        env = SubprocVecEnv([SFv2_make_env(i) for i in range(n_envs)])
        env = SelectiveVecNormalize(env, n_continuous_dims=10, n_frames=4)

        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=ent_coef,
            clip_range=clip_range,
            n_epochs=10,
            gamma=0.99,
            target_kl=0.03, 
            # THE FIX: Removed the [] from around the dict()
            policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256])), # THE MASSIVE BRAIN
            verbose=0, 
            tensorboard_log=directories["logs"],
            device="cuda"
        )

        model.learn(
            total_timesteps=150000,
            tb_log_name=f"{config.MODEL_NAME}_Optuna_V2_Trial_{trial.number}"
        )
        
        # --- THE FIX: EXTRACT INTERNAL MEMORY INSTEAD OF EVALUATING ---
        print("Extracting trial performance...")
        ep_info_buffer = list(model.ep_info_buffer)
        
        if len(ep_info_buffer) == 0:
            print(f"[Trial {trial.number}] Zero completed episodes. Pruning.")
            raise optuna.TrialPruned()
        last_n = ep_info_buffer[-20:]  # Last 20 episodes only
        mean_reward = sum(ep["r"] for ep in last_n) / len(last_n)
        print(f"Trial {trial.number} → Mean Reward (last {len(last_n)} eps): {mean_reward:.4f}")


        # --- MOVED SAVING LOGIC INSIDE THE TRY BLOCK ---
        # Save best model (atomically — model and normalizer from the same trial)
        best_reward = -float('inf')
        if os.path.exists(BEST_REWARD_PATH):
            with open(BEST_REWARD_PATH, "r") as f:
                content = f.read().strip()
                if content:  # Only convert if not empty
                    best_reward = float(content)

        if mean_reward > best_reward:
            print(f"!!! NEW BEST V2 MODEL (Reward: {mean_reward:.4f}) !!!")
            model.save(os.path.join(directories["optuna"], "best_ppo_sf2_v2"))
            env.save(os.path.join(directories["optuna"], "best_vec_normalize_v2.pkl"))
            with open(BEST_REWARD_PATH, "w") as f:
                f.write(str(mean_reward))
            # FIX: Save best params every time a new best is found, not just at the end
            _print_and_save_best(trial.study)

        return mean_reward
        
    except optuna.TrialPruned:
        raise  # Re-raise cleanly — don't swallow it
    except Exception as e:
        print(f"\n[ERROR] Trial {trial.number} crashed with: {type(e).__name__}: {e}")
        raise  # Let Optuna mark this as FAIL, not PRUNED

    finally:
        failsafe_env(env=env, model=model)

if __name__ == "__main__":
    print("Initializing Phase 2 Optuna V2 Tuning...")
    study = optuna.create_study(study_name="sf2_ppo_v2", 
                                direction="maximize", 
                                storage=None,  # Optional: add SQLite path here for full crash-resume
                                )

    try:
        study.optimize(objective, 
                       n_trials=config.N_HYPERPARAMETER_TRIALS,
                       gc_after_trial=True  # Force GC between trials for long runs
                       )
    except KeyboardInterrupt:
        print("\nOptuna Optimization forcefully interrupted by user.")
  
    finally:
        # FIX: Always print and save best results, even on crash/interrupt
        print("\n[Optuna] Optimization session ending. Generating final report...")
        _print_and_save_best(study)