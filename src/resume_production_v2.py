import os
import subprocess
import time
import multiprocessing
import gc
import torch
from typing import Callable 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import config
from env_sf2_v2 import StreetFighterEnvV2

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
        env = StreetFighterEnvV2(rank=rank)
        env = Monitor(env)
        return env
    return _init

def resume_training(model_path, vec_path):
    
    print(f"Initializing {config.N_ENVS}-Core Resume Environment...")
    
    # 1. Boot Parallel Emulators
    n_envs = config.N_ENVS
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    
    # 2. Load the VecNormalize Math
    print(f"Loading normalization stats from {vec_path}...")
    env = VecNormalize.load(vec_path, env)
    
    # CRITICAL: Ensure the environment continues to update its normalization math
    env.training = True
    env.norm_reward = True

    # 3. Load the Brain (PPO)
    print(f"Loading neural network weights from {model_path}...")
    model = PPO.load(
        model_path, 
        env=env, 
        device="cuda", 
        tensorboard_log=directories["logs"],
        custom_objects={
            "learning_rate": 0.00014963345069997716,   # Restored from Trial 19
            "clip_range": 0.25225074436550204,         # Restored from Trial 19
            "ent_coef": 0.058045038937880364
        }
    )

    # 5. Restore Golden Architecture Constraints
    model.target_kl = 0.03   # THE FIX: Restored to standard 0.03 
    model.n_epochs = 10      
    model.batch_size = 512

    
    # 4. Setup our impenetrable Failsafe Callback (Saves every 100k GLOBAL steps)
    sync_callback = SyncCheckpointCallback(
        save_freq_steps=config.SAVE_FREQ_STEPS // n_envs, 
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
        return True # Signals the Supervisor that we finished successfully
        
    except KeyboardInterrupt:
        print("\n[MANUAL OVERRIDE] Training forcefully interrupted by user.")
        model.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_EMERGENCY"))
        env.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_vecnormalize_EMERGENCY.pkl"))
        return True # Signals the Supervisor to stop
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Training crashed: {e}")
        model.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_CRASH_SAVE"))
        env.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_vecnormalize_CRASH_SAVE.pkl"))
        return False

        
    finally:
        print("Executing Nuclear Cleanup...")
        
        # 1. OS-LEVEL FIRE AND FORGET
        os.system("taskkill /F /IM EmuHawk.exe >nul 2>&1")
        time.sleep(2)
        
        # 2. THE UPGRADED THREAD SNIPER
        active_children = multiprocessing.active_children()
        if active_children:
            print(f"Force-killing {len(active_children)} zombie Python worker processes...")
            for child in active_children:
                try:
                    child.kill()  
                except Exception:
                    pass
                    
        # 3. THE VRAM/RAM PURGE (NEW)
        print("Purging GPU and System Memory...")
        try:
            # Forcefully delete the massive objects from Python's local memory
            del model
            del env
        except UnboundLocalError:
            pass # Ignores the error if the script crashed before they were created
            
        # Command Python to empty the RAM trash can
        gc.collect() 
        # Command PyTorch to completely flush the GPU VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 

        # Give Windows 5 seconds to finalize the memory flush
        time.sleep(5)

if __name__ == "__main__":
    current_model_path = os.path.join(directories["project_root"], config.TRAINING_ZIP_FILE)
    current_vec_path = os.path.join(directories["project_root"], config.TRAINING_PKL_FILE)

    restart_count = 0
    
    while True:
        success = resume_training(current_model_path, current_vec_path)
        
        if success:
            print("Training session ended cleanly.")
            break # Exit the loop
            
        else:
            restart_count += 1
            print(f"\n--- INITIATING AUTO-RESTART #{restart_count} ---")
            # Override the load paths to point to the newly generated crash files
            current_model_path = os.path.join(directories["production"], f"{config.MODEL_NAME}_CRASH_SAVE.zip")
            current_vec_path = os.path.join(directories["production"], f"{config.MODEL_NAME}_vecnormalize_CRASH_SAVE.pkl")
            # The loop goes back to the top and safely restarts!