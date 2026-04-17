import os
import subprocess
import time
from typing import Callable 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import config
from v1.env_sf2 import StreetFighterEnv

directories = config.get_directory()

# --- NEW: The Linear Schedule Function ---
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Progressively drops the learning rate.
    progress_remaining starts at 1.0 and goes to 0.0.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func



# 1. Custom Callback to Synchronize .zip and .pkl saves (FIXED)
class SyncCheckpointCallback(BaseCallback):
    """Saves the model AND the VecNormalize stats simultaneously."""
    def __init__(self, save_freq_steps, save_path, name_prefix, verbose=1):
        super().__init__(verbose)
        self.save_freq_steps = save_freq_steps
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.last_save_step = 0 # Track the exact step we last saved at

    def _on_step(self) -> bool:
        # Initialize the baseline tracker on the very first step of the resumed run
        if self.last_save_step == 0:
            self.last_save_step = self.num_timesteps

        # Delta Check: Have we progressed 100,000 steps since the last save?
        if self.num_timesteps - self.last_save_step >= self.save_freq_steps:
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_model_{self.num_timesteps}_steps")
            vec_path = os.path.join(self.save_path, f"{self.name_prefix}_vecnormalize_{self.num_timesteps}_steps.pkl")
            
            self.model.save(model_path)
            self.training_env.save(vec_path)
            
            if self.verbose > 0:
                print(f"\n[Failsafe Backup] Saved Model and Normalization Stats at {self.num_timesteps} steps.")
            
            # Reset the baseline tracker
            self.last_save_step = self.num_timesteps
            
        return True

def make_env(rank):
    def _init():
        env = StreetFighterEnv(rank=rank)
        env = Monitor(env)
        return env
    return _init

def resume_training():
    print(f"Initializing {config.N_ENVS}-Core Resume Environment...")

    # LOAD PATHS FOR RESUMPTION
    model_load_path = os.path.join(directories["project_root"], config.TRAINING_ZIP_FILE)
    vec_load_path = os.path.join(directories["project_root"], config.TRAINING_PKL_FILE)
    
    # 1. Boot Parallel Emulators
    n_envs = config.N_ENVS
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    
    # 2. Load the VecNormalize Math
    print(f"Loading normalization stats from {config.TRAINING_PKL_FILE}...")
    env = VecNormalize.load(vec_load_path, env)
    
    # CRITICAL: Ensure the environment continues to update its normalization math
    env.training = True
    env.norm_reward = True
    
    LR_BASE = config.LR

    # 3. Load the Brain (PPO)
    print(f"Loading neural network weights from {config.TRAINING_ZIP_FILE}...")
    model = PPO.load(
        model_load_path, 
        env=env, 
        device="cuda", 
        tensorboard_log=directories["logs"],
        custom_objects={
            "learning_rate": linear_schedule(LR_BASE),
            "clip_range": linear_schedule(0.199)
            } 
    )

    # NEW: Surgically inject the stability constraints to prevent KL Explosions
    model.target_kl = 0.03   # The Emergency Brake
    model.n_epochs = 5       # Reduce how many times it loops over the same data
    model.batch_size = 1024  # Increase batch size so it looks at bigger, less noisy chunks of data

    
    # 4. Setup our impenetrable Failsafe Callback (Saves every 100k GLOBAL steps)
    sync_callback = SyncCheckpointCallback(
        save_freq_steps=config.SAVE_FREQ_STEPS, 
        save_path=directories["production"],
        name_prefix=config.MODEL_NAME
    )
    
    try:
        # reset_num_timesteps=False ensures TensorBoard continues the graph smoothly
        model.learn(
            total_timesteps=config.RESUME_PRODUCTION_TIMESTEPS, 
            callback=sync_callback,
            tb_log_name=config.MODEL_NAME,
            reset_num_timesteps=False 
        )
        
        # Save Final Grandmaster
        model.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_FINAL"))
        env.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_vecnormalize_FINAL.pkl"))
        print("\nProduction Training Complete!")
        
    except KeyboardInterrupt:
        print("\n[MANUAL OVERRIDE] Training forcefully interrupted by user.")
        model.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_EMERGENCY"))
        env.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_vecnormalize_EMERGENCY.pkl"))
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Training crashed: {e}")
        model.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_CRASH_SAVE"))
        env.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_vecnormalize_CRASH_SAVE.pkl"))
        
    finally:
        print("Executing Nuclear Cleanup...")
        subprocess.run(["taskkill", "/F", "/IM", "EmuHawk.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)

if __name__ == "__main__":
    resume_training()