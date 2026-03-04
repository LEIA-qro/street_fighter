import os
import subprocess
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

import config
from env_sf2 import StreetFighterEnv

LOG_DIR = os.path.join(config.PROJECT_ROOT, "logs")
MODEL_DIR = os.path.join(config.PROJECT_ROOT, "models", "production")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def make_env(rank):
    def _init():
        env = StreetFighterEnv(rank=rank)
        env = Monitor(env)
        return env
    return _init

def train_production():
    print("Initializing 16-Core Production Environment...")
    
    # 1. Launch 16 parallel emulators
    n_envs = config.N_ENVS
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    
    # 2. Apply Normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # 3. Setup aggressive checkpointing (Save every 500k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 500_000 // n_envs), 
        save_path=MODEL_DIR,
        name_prefix="ppo_sf2_prod"
    )
    
    # 4. Initialize PPO with Optuna's WINNING Hyperparameters
    # *** REPLACE THESE VALUES WITH YOUR OPTUNA RESULTS ***
    print("Building Grandmaster Neural Network...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.000112125, # Optuna's exact LR
        n_steps=2048,              # Optuna's exact n_steps
        batch_size=64,             # Optuna's exact batch_size
        ent_coef=0.067344,         # Optuna's exact entropy
        clip_range=0.208978,       # Optuna's exact clip range
        n_epochs=10,
        gamma=0.99,
        tensorboard_log=LOG_DIR,
        verbose=1,
        device="cuda"
    )
    
    # 5. The Grandmaster Training Loop (10 Million Steps)
    # With 16 cores, this will process exponentially faster than before.
    TOTAL_TIMESTEPS = 10_000_000 
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=checkpoint_callback,
            tb_log_name="PPO_Production_Run_1"
        )
        
        # Save Final Production Model
        model.save(os.path.join(MODEL_DIR, "sf2_grandmaster_final"))
        env.save(os.path.join(MODEL_DIR, "vec_normalize_grandmaster.pkl"))
        print("\nProduction Training Complete!")
        
    except KeyboardInterrupt:
        print("\nTraining forcefully interrupted. Executing emergency save...")
        model.save(os.path.join(MODEL_DIR, "sf2_grandmaster_EMERGENCY"))
        env.save(os.path.join(MODEL_DIR, "vec_normalize_EMERGENCY.pkl"))
        
    finally:
        # Nuclear Failsafe Cleanup
        print("Executing Nuclear Cleanup...")
        subprocess.run(["taskkill", "/F", "/IM", "EmuHawk.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)

if __name__ == "__main__":
    train_production()