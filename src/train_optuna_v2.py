import os
import multiprocessing
import optuna
import gc
import torch
# import subprocess 
import time       
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.evaluation import evaluate_policy


import config
from env_sf2_v2 import StreetFighterEnvV2 
from selective_norm import SelectiveVecNormalize

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

    n_envs = config.N_ENVS
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
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

    try:
        model.learn(
            total_timesteps=150000,
            tb_log_name=f"{config.MODEL_NAME}_Optuna_V2_Trial_{trial.number}"
        )
        
        # --- THE FIX: EXTRACT INTERNAL MEMORY INSTEAD OF EVALUATING ---
        print("Extracting trial performance...")
        ep_info_buffer = model.ep_info_buffer
        
        if len(ep_info_buffer) > 0:
            # Calculate the average reward of the last 100 completed matches
            mean_reward = sum([ep_info["r"] for ep_info in ep_info_buffer]) / len(ep_info_buffer)
        else:
            # If the agent stood completely still and didn't finish a single match in 150k steps, 
            # it is a totally broken hyperparameter set. Punish it severely.
            mean_reward = -9999.0 
            
        print(f"Trial {trial.number} finished with Mean Reward: {mean_reward}")

        # --- MOVED SAVING LOGIC INSIDE THE TRY BLOCK ---
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
        # -----------------------------------------------

    except Exception as e:
        print(f"\n[WARNING] Trial {trial.number} crashed: {e}")
        raise optuna.exceptions.TrialPruned()

    finally:
        print("Executing Failsafe: Purging zombie instances and VRAM...")
        # 1. Kill Emulators
        os.system("taskkill /F /IM EmuHawk.exe >nul 2>&1")
        time.sleep(2)
        
        # 2. The Thread Sniper
        active_children = multiprocessing.active_children()
        if active_children:
            for child in active_children:
                try:
                    child.kill()
                except Exception:
                    pass
        
        # 3. The VRAM Purge
        try:
            del model
            del env
        except UnboundLocalError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        time.sleep(3)

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