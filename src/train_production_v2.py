import os
import subprocess
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import CheckpointCallback
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
    
    config.RYU_ONLY_STATES = config.RYU_ONLY_STATES_PHASE_1

    n_envs = config.N_ENVS
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Instantiate the Brain with Trial 9 Math
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=9.86e-05,   # From Optuna
        n_steps=8192,             # From Optuna
        batch_size=512,           # From Optuna
        ent_coef=0.0918,          # From Optuna
        clip_range=0.185,         # From Optuna
        n_epochs=10,              
        gamma=0.99,
        target_kl=0.03,
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
        
    finally:
        # Nuclear Failsafe Cleanup
        print("Executing Failsafe: Purging zombie BizHawk instances...")
        subprocess.run(["taskkill", "/F", "/IM", "EmuHawk.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)

if __name__ == "__main__":
    train_production()