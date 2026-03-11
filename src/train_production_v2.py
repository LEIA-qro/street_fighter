import os
import multiprocessing
import gc
import torch
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import config
from env_sf2_v2 import StreetFighterEnvV2

directories = config.get_directory()

class SaveOnStepCallback(BaseCallback):
    """Callback to save the model and normalization math every N steps."""
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, config.MODEL_NAME + f"_{self.num_timesteps}_steps")
            vec_path = os.path.join(self.save_path, config.MODEL_NAME + f"_vecnormalize_{self.num_timesteps}_steps.pkl")
            
            self.model.save(model_path)
            self.training_env.save(vec_path)
            if self.verbose > 0:
                print(f"\n[CHECKPOINT] Saved model at {self.num_timesteps} steps!")
        return True
    
def make_env(rank):
    def _init():
        env = StreetFighterEnvV2(rank=rank)
        env = Monitor(env)  # <--- This is the accountant that tracks the score!
        return env
    return _init

def train_production():
    print("Initializing Phase 1: Grandmaster Production Training...")

    n_envs = config.N_ENVS
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Instantiate the Brain with Trial 9 Math
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.00014963345069997716,   # From Trial 19
        n_steps=2048,                           # From Trial 19
        batch_size=512,                         # From Trial 19
        ent_coef=0.058045038937880364,          # From Trial 19
        clip_range=0.25225074436550204,         # From Trial 19
        n_epochs=10,              
        gamma=0.99,
        target_kl=0.03,
        policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256])),
        verbose=1,
        tensorboard_log=directories["logs"],
        device="cuda"
    )

    checkpoint_callback = SaveOnStepCallback(save_freq=config.SAVE_FREQ_STEPS // n_envs, save_path=directories["production"])
    
    # 5. The Grandmaster Training Loop (10 Million Steps)
    # With 16 cores, this will process exponentially faster than before.
    
    try:
        model.learn(
            total_timesteps=config.STARTING_TOTAL_TIMESTEPS, 
            callback=checkpoint_callback,
            tb_log_name=config.MODEL_NAME
        )
        
        # Save Final Production Model
        model.save(os.path.join(directories["production"], config.MODEL_NAME + "_FINAL"))
        env.save(os.path.join(directories["production"], config.MODEL_NAME + "_vecnormalize_FINAL.pkl"))
        print("\nProduction Training Complete!")
        
    except KeyboardInterrupt:
        print("\nTraining forcefully interrupted. Executing emergency save...")
        model.save(os.path.join(directories["production"], config.MODEL_NAME + "_EMERGENCY"))
        env.save(os.path.join(directories["production"], config.MODEL_NAME + "_vecnormalize_EMERGENCY.pkl"))
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Training crashed: {e}")
        model.save(os.path.join(directories["production"], config.MODEL_NAME + "_CRASH_SAVE"))
        env.save(os.path.join(directories["production"], config.MODEL_NAME + "_vecnormalize_CRASH_SAVE.pkl"))
        
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

if __name__ == "__main__":
    train_production()