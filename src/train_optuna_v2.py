import os
import optuna
import subprocess 
import time       
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

import config
# THE V2 IMPORT
from env_sf2_v2 import StreetFighterEnvV2 

directories = config.get_directory()

def make_env(rank):
    def _init():
        env = StreetFighterEnvV2(rank=rank)
        env = Monitor(env)
        return env
    return _init

def objective(trial):
    # THE MASSIVE MATRIX SEARCH SPACE
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.01, 0.1)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    n_steps = trial.suggest_categorical("n_steps", [2048, 4096, 8192])
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])

    if n_steps * config.N_ENVS % batch_size != 0:
        raise optuna.TrialPruned()

    print(f"\n--- Starting Trial {trial.number} ---")

    # FORCE PHASE 2 (Sparring Partners)
    # config.RYU_ONLY_STATES = config.RYU_ONLY_STATES_PHASE_2

    n_envs = config.N_ENVS
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

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
        target_kl=0.03, # THE KL FAILSAFE
        verbose=0, 
        tensorboard_log=directories["logs"],
        device="cuda"
    )

    try:
        model.learn(
            total_timesteps=175000,
            tb_log_name=f"{config.MODEL_NAME}_Optuna_V2_Trial_{trial.number}"
        )
        
        print("Evaluating V2 agent...")
        mean_reward, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        print(f"Trial {trial.number} finished with Mean Reward: {mean_reward}")

    except Exception as e:
        print(f"\n[WARNING] Trial {trial.number} crashed: {e}")
        raise optuna.exceptions.TrialPruned()

    finally:
        try:
            env.close()
        except Exception:
            pass

        print("Executing Failsafe: Purging zombie BizHawk instances...")
        subprocess.run(["taskkill", "/F", "/IM", "EmuHawk.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)

    # YOUR SAVING LOGIC PRESERVED
    best_path = os.path.join(directories["optuna"], "best_reward_v2.txt")
    best_reward = -float('inf')
   
    if os.path.exists(best_path):
        with open(best_path, "r") as f:
            best_reward = float(f.read().strip()) 

    if mean_reward > best_reward:
        print(f"!!! NEW BEST V2 MODEL FOUND !!! (Reward: {mean_reward})")
        model.save(os.path.join(directories["optuna"], "best_ppo_sf2_v2"))
        env.save(os.path.join(directories["optuna"], "best_vec_normalize_v2.pkl"))
        with open(best_path, "w") as f:
            f.write(str(mean_reward))

    return mean_reward

if __name__ == "__main__":
    print("Initializing Phase 2 Optuna V2 Tuning...")
    study = optuna.create_study(direction="maximize")

    try:
        study.optimize(objective, n_trials=config.N_HYPERPARAMETER_TRIALS)
    except KeyboardInterrupt:
        print("\nOptuna Optimization forcefully interrupted by user.")
  
    print("\nOptimization Complete!")
    print("Best Hyperparameters:", study.best_trial.params)