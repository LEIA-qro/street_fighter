 # pip install optuna

import os
import optuna
import subprocess 
import time       
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

import config
from env_sf2 import StreetFighterEnv

directories = config.get_directory()

# 1. Environment Factory for Parallelization
def make_env(rank):
    """
    Utility function for multiprocess env.
    Every environment gets a unique rank to assign a unique TCP Port.
    """
    def _init():
        env = StreetFighterEnv(rank=rank)
        env = Monitor(env)
        return env
    return _init

# 2. The Optuna Objective Function
def objective(trial):
    """
    This function represents a single 'Trial' (one set of hyperparameters).
    Optuna will run this function dozens of times.
    """
    # Define the hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.01, 0.1)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # We must ensure batch_size is a factor of n_steps * n_envs to prevent PyTorch crashes
    if n_steps * config.N_ENVS % batch_size != 0:
        raise optuna.TrialPruned()

    print(f"\n--- Starting Trial {trial.number} ---")

    # Launch 16 independent BizHawk instances on 16 CPU Cores
    n_envs = config.N_ENVS
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # Apply Normalization across all 16 environments
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
   

    # Initialize PPO with Optuna's suggested hyperparameters
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
        verbose=0, # Turn off verbose so Optuna terminal logs stay clean
        tensorboard_log=directories["logs"],
        device="cuda"
    )

    # Train for a fast burst to see if these parameters work (e.g., 150k steps)
    # Because we have 16 envs, this will process 16x faster than single-env
    try:
        model.learn(
            total_timesteps=175000,
            tb_log_name=f"Optuna_{config.MODEL_NAME}_Trial_{trial.number}"
            )
        
        # Evaluate the model
        print("Evaluating agent...")
        # evaluate_policy plays 10 matches and returns the average reward
        mean_reward, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

        print(f"Trial {trial.number} finished with Mean Reward: {mean_reward}")

    except Exception as e:
        print(f"\n[WARNING] Trial {trial.number} crashed: {e}")
        # Prune the trial so Optuna's math engine ignores this hardware failure
        raise optuna.exceptions.TrialPruned()

    finally:
        # Execute the Poison Pill for all 16 environments
        try:
            env.close()
        except Exception as e:
            pass

        # NUCLEAR FAILSAFE: Tell Windows to forcefully kill all EmuHawk instances
        print("Executing Failsafe: Purging zombie BizHawk instances...")
        subprocess.run(
            ["taskkill", "/F", "/IM", "EmuHawk.exe"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
       
        # Give Windows 3 seconds to fully release the RAM and TCP ports
        time.sleep(3)

    # Save the absolute best model
    best_path = os.path.join(directories["optuna"], "best_reward.txt")
    best_reward = -float('inf')
   
    if os.path.exists(best_path):
        with open(best_path, "r") as f:
            best_reward = float(f.read().strip()) 

    if mean_reward > best_reward:
        print(f"!!! NEW BEST MODEL FOUND !!! (Reward: {mean_reward})")
        model.save(os.path.join(directories["optuna"], "best_ppo_sf2"))
        env.save(os.path.join(directories["optuna"], "best_vec_normalize.pkl"))
        with open(best_path, "w") as f:
            f.write(str(mean_reward))

    return mean_reward

# 3. Execution Block
if __name__ == "__main__":
    # Create an Optuna study that maximizes the mean reward
    study = optuna.create_study(direction="maximize")

    # Run 20 different hyperparameter combinations
    try:
        study.optimize(objective, n_trials=config.N_HYPERPARAMETER_TRIALS)
    except KeyboardInterrupt:
        print("\nOptuna Optimization forcefully interrupted by user.")
  

    print("\nOptimization Complete!")
    print("Best Hyperparameters:", study.best_trial.params)
