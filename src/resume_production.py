import os
import subprocess
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import config
from env_sf2 import StreetFighterEnv

LOG_DIR = os.path.join(config.PROJECT_ROOT, "logs")
MODEL_DIR = os.path.join(config.PROJECT_ROOT, "models", "production")

# 1. Custom Callback to Synchronize .zip and .pkl saves
class SyncCheckpointCallback(BaseCallback):
    """Saves the model AND the VecNormalize stats simultaneously."""
    def __init__(self, save_freq, save_path, name_prefix, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        # Check if it's time to save based on the parallel step count
        if self.num_timesteps % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_model_{self.num_timesteps}_steps")
            vec_path = os.path.join(self.save_path, f"{self.name_prefix}_vecnormalize_{self.num_timesteps}_steps.pkl")
            
            self.model.save(model_path)
            self.training_env.save(vec_path)
            
            if self.verbose > 0:
                print(f"\n[Failsafe Backup] Saved Model and Normalization Stats at {self.num_timesteps} steps.")
        return True

def make_env(rank):
    def _init():
        env = StreetFighterEnv(rank=rank)
        env = Monitor(env)
        return env
    return _init

def resume_training():
    print("Initializing 16-Core Resume Environment...")
    
    # ---------------------------------------------------------
    # ACTION REQUIRED: EXACT FILENAMES MUST BE PROVIDED HERE
    # ---------------------------------------------------------
    # Look in your models/production folder and put the exact name of the latest .zip
    LATEST_ZIP_FILE = "sf2_grandmaster_EMERGENCY.zip" 
    
    # Look for the emergency .pkl file or a previously saved .pkl file
    LATEST_PKL_FILE = "vec_normalize_EMERGENCY.pkl" 
    
    model_load_path = os.path.join(MODEL_DIR, LATEST_ZIP_FILE)
    vec_load_path = os.path.join(MODEL_DIR, LATEST_PKL_FILE)
    
    # 1. Boot Parallel Emulators
    n_envs = config.N_ENVS
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    
    # 2. Load the VecNormalize Math
    print(f"Loading normalization stats from {LATEST_PKL_FILE}...")
    env = VecNormalize.load(vec_load_path, env)
    
    # CRITICAL: Ensure the environment continues to update its normalization math
    env.training = True
    env.norm_reward = True
    
    # 3. Load the Brain (PPO)
    print(f"Loading neural network weights from {LATEST_ZIP_FILE}...")
    model = PPO.load(
        model_load_path, 
        env=env, 
        device="cuda", 
        tensorboard_log=LOG_DIR
        )
    
    # 4. Setup our new impenetrable Failsafe Callback (Saves every 100k steps)
    sync_callback = SyncCheckpointCallback(
        save_freq=max(1, 100_000 // n_envs), 
        save_path=MODEL_DIR,
        name_prefix="sf2_grandmaster"
    )
    
    TOTAL_TIMESTEPS = 9_000_000 # The remaining steps
    
    try:
        # reset_num_timesteps=False ensures TensorBoard continues the graph smoothly
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=sync_callback,
            tb_log_name="PPO_Production_Run_1",
            reset_num_timesteps=False 
        )
        
        # Save Final Grandmaster
        model.save(os.path.join(MODEL_DIR, "sf2_grandmaster_FINAL"))
        env.save(os.path.join(MODEL_DIR, "sf2_grandmaster_vecnormalize_FINAL.pkl"))
        print("\nProduction Training Complete!")
        
    except KeyboardInterrupt:
        # 1. Catch the Manual Stop (Ctrl+C)
        print("\n[MANUAL OVERRIDE] Training forcefully interrupted by user.")
        print("Executing graceful emergency save...")
        model.save(os.path.join(MODEL_DIR, "sf2_grandmaster_EMERGENCY"))
        env.save(os.path.join(MODEL_DIR, "sf2_vecnormalize_EMERGENCY.pkl"))
        
    except Exception as e:
        # 2. Catch the Unpredictable Emulation Crash (e.g., EOFError)
        print(f"\n[CRITICAL ERROR] Training crashed: {e}")
        print("Executing automated crash save...")
        model.save(os.path.join(MODEL_DIR, "sf2_grandmaster_CRASH_SAVE"))
        env.save(os.path.join(MODEL_DIR, "sf2_vecnormalize_CRASH_SAVE.pkl"))
        
    finally:
        # 3. Always nuke the zombies, no matter how the script ended
        print("Executing Nuclear Cleanup...")
        subprocess.run(["taskkill", "/F", "/IM", "EmuHawk.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)

if __name__ == "__main__":
    resume_training()