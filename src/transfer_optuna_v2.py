import os

import optuna

    
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


def make_objective(model_path: str, vec_path: str, next_phase_states: list):
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

        n_envs = config.N_ENVS
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        
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

        try:
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

            # -----------------------------------------------

        except Exception as e:
            print(f"\n[WARNING] Trial {trial.number} crashed: {e}")
            raise optuna.exceptions.TrialPruned()

        finally:
            _nuclear_cleanup(model=model, env=env)

        return mean_reward
    
    return objective




def run_study(model_path: str, vec_path: str, next_phase_states: list,
              n_trials: int = config.N_HYPERPARAMETER_TRIALS,
              study_name: str = "transfer_optuna") -> dict:
    """
    Public API used by phase_supervisor.py.
    Returns the best hyperparams found as a dict.
    """
    print(f"\n[Optuna] Starting transfer study '{study_name}' — {n_trials} trials")
    print(f"[Optuna] Target states: {next_phase_states}")

    study = optuna.create_study(direction="maximize", study_name=study_name)
    try:
        study.optimize(
            make_objective(model_path, vec_path, next_phase_states),
            n_trials=n_trials
        )
    except KeyboardInterrupt:
        print("\n[Optuna] Study interrupted by user.")

    best = study.best_trial.params
    print(f"\n[Optuna] Best params: {best}")
    return best


if __name__ == "__main__":
    from transfer_optuna_v2 import run_study
    run_study(
        model_path=os.path.join(directories["project_root"], config.TRAINING_ZIP_FILE),
        vec_path=os.path.join(directories["project_root"], config.TRAINING_PKL_FILE),
        next_phase_states=config.TRAINING_STATES,
    )