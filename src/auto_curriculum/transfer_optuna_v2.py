import os
import optuna
import csv
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.evaluation import evaluate_policy


import config
from env_sf2_v2 import StreetFighterEnvV2
from selective_norm import SelectiveVecNormalize 
from env_nuclear_cleanup import _nuclear_cleanup


directories = config.get_directory()

def make_env(rank):
    def _init():
        env = StreetFighterEnvV2(rank=rank)
        env = Monitor(env)
        return env
    return _init


TRIAL_LOG_PATH = os.path.join(directories["optuna"], "trial_history.csv")

def _log_trial(study_name: str, trial_number: int, params: dict, mean_reward: float):
    file_exists = os.path.isfile(TRIAL_LOG_PATH) and os.path.getsize(TRIAL_LOG_PATH) > 0
    with open(TRIAL_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "study_name", "trial_number",
            "mean_reward", "learning_rate", "ent_coef", "clip_range"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp":     datetime.now().isoformat(),
            "study_name":    study_name,
            "trial_number":  trial_number,
            "mean_reward":   round(mean_reward, 4),
            "learning_rate": params.get("learning_rate"),
            "ent_coef":      params.get("ent_coef"),
            "clip_range":    params.get("clip_range"),
        })


def make_objective(model_path: str, vec_path: str, next_phase_states: list,
                   study_name: str = "transfer_optuna"):  # ADD THIS
    """
    Closure factory — captures checkpoint paths and target states so the
    objective is fully self-contained and re-entrant across Optuna trials.
    """
    def objective(trial):
        # THE SEARCH SPACE
        learning_rate = trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True) # Tighter bounds for fine-tuning
        ent_coef = trial.suggest_float("ent_coef", 0.01, 0.05)                     # Tighter bounds
        clip_range = trial.suggest_float("clip_range", 0.1, 0.2)                   # Tighter bounds

        print(f"\n--- Starting Trial {trial.number} ---")
        print(f"\n--- Trial {trial.number} | "
              f"lr={learning_rate:.2e} ent={ent_coef:.4f} clip={clip_range:.3f} ---")
        
        env   = None  # GUARD: ensures finally block never raises UnboundLocalError
        model = None  # GUARD

        try:
            env = SubprocVecEnv([make_env(i) for i in range(config.N_ENVS)])
            
            # THE FIX: Load the Phase 1 VecNormalize math so the brain isn't blind!
            # vec_path = os.path.join(directories["project_root"], config.TRAINING_PKL_FILE)
            env = SelectiveVecNormalize.load(vec_path, env)
            env.training = True
            env.norm_reward = True

            # THE FIX: Load the Phase 1 Brain and override its hyperparams for this trial
            # model_path = os.path.join(directories["project_root"], config.TRAINING_ZIP_FILE) # Update with your exact Phase 1 file
            
            # Broadcast next-phase states to all subprocess envs
            env.env_method("set_training_states", next_phase_states)

            model = PPO.load(
                model_path,
                env=env,
                device="cuda",
                tensorboard_log=directories["logs"],
                custom_objects={
                    "learning_rate": learning_rate,
                    "ent_coef": ent_coef,
                    "clip_range": clip_range,
                }
            )
            model.target_kl = 0.03

            model.learn(
                total_timesteps=150000,
                tb_log_name=f"{config.MODEL_NAME}_Optuna_V2_Trial_{trial.number}"
            )
            
            # --- THE FIX: EXTRACT INTERNAL MEMORY INSTEAD OF EVALUATING ---
            print("Extracting trial performance...")
            buf = model.ep_info_buffer
            mean_reward = (sum(e["r"] for e in buf) / len(buf)) if buf else -9999.0
            print(f"Trial {trial.number} → Mean Reward: {mean_reward:.2f}")
            # ep_info_buffer = model.ep_info_buffer
            _log_trial(study_name, trial.number, trial.params, mean_reward)

            best_path = os.path.join(directories["optuna"], "best_transfer_reward.txt")
            best_reward = -float("inf")
            
            if os.path.exists(best_path):
                with open(best_path, "r") as f:
                    best_reward = float(f.read().strip())

            if mean_reward > best_reward:
                print(f"!!! NEW BEST TRANSFER MODEL !!! (Reward: {mean_reward:.2f})")
                model.save(os.path.join(directories["optuna"], "best_transfer_ppo"))
                env.save(os.path.join(directories["optuna"], "best_transfer_vecnorm.pkl"))
                with open(best_path, "w") as f:
                    f.write(str(mean_reward))
                
            print(f"Trial {trial.number} finished with Mean Reward: {mean_reward}")
            
            return mean_reward  # Only reachable on clean success

            # -----------------------------------------------

        except Exception as e:
            print(f"\n[WARNING] Trial {trial.number} crashed: {e}")
            raise optuna.exceptions.TrialPruned()

        finally:
            _nuclear_cleanup(model=model, env=env)
    
    return objective

def run_study(model_path: str, vec_path: str, next_phase_states: list,
              n_trials: int = config.N_HYPERPARAMETER_TRIALS,
              study_name: str = "transfer_optuna") -> dict:

    # --- THE FIX: Persistent SQLite storage ---
    # Trials survive crashes. Re-running the same study_name resumes from where it left off.
    storage_path = os.path.join(directories["optuna"], "optuna_studies.db")
    storage_url  = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,   # Resume existing study if process crashed mid-run
    )
    
    # Only run the REMAINING trials, not the full n_trials again
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining_trials = max(0, n_trials - completed_trials)
    
    if remaining_trials == 0:
        print(f"[Optuna] Study '{study_name}' already complete ({completed_trials} trials). Returning best params.")
        return study.best_trial.params

    print(f"[Optuna] Resuming study '{study_name}': {completed_trials} done, {remaining_trials} remaining.")
    
    try:
        study.optimize(
            make_objective(model_path, vec_path, next_phase_states, study_name),
            n_trials=remaining_trials
        )
    except KeyboardInterrupt:
        print("\n[Optuna] Study interrupted by user.")

    best = study.best_trial.params
    print(f"\n[Optuna] Best params: {best}")
    return best


if __name__ == "__main__":

    run_study(
        model_path=os.path.join(directories["project_root"], config.TRAINING_ZIP_FILE),
        vec_path=os.path.join(directories["project_root"], config.TRAINING_PKL_FILE),
        next_phase_states=config.TRAINING_STATES,
    )